# Copyright (c) 2025, Markus Knoche. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import pytorch_lightning as pl
from losses import LaplaceNLLLoss, VonMisesNLLLoss
from copy import deepcopy
from itertools import compress, chain
from modules import QCNetMapEncoder, DonutLayer
from layers import MLPLayer
from metrics import Brier, MR, minADE, minFDE
import torch.nn.functional as F


class DonutPa(pl.LightningModule):
    """
    DONUT: A Decoder-Only Model for Trajectory Prediction

    Args:
        t_per_tok:
            Number of timesteps that are aggregated into a single token.
        t_hist:
            Length in timesteps of the history that is supplied to the model.
        t_pred:
            Prediction horizon in timesteps.
        num_modes:
            Number of trajectory hypotheses that are generated per agent.
        refine:
            If True an additional refinement decoder is used.
        over_predict:
            If True the decoders also output an auxiliary "over-prediction".
        hidden_dim:
            Dimensionality of all latent feature vectors.
        edge_limit:
            Relative threshold for pruning edges to limit the required memory
            for huge scenes.
        map_enc_radius:
            Local radius (in metres) that is considered by the map encoder.
        map_enc_layers:
            Number of layers that are used for the map encoder.
        dec_attn_order:
            Decoder attention order: (t)emporal, (s)ocial, (r)oad, (m)ode.
        dec_attn_repetitions:
            How often the attention order is repeated.
        dec_radius_r:
            Radius (in metres) of the road attention.
        dec_radius_s:
            Radius (in metres) of the social attention.
        lr:
            Initial learning rate for AdamW.
        weight_decay:
            L2 weight decay used for regularised parameters.
        decay_epochs:
            Number of decay epochs for the cosine scheduler.
    """

    def __init__(
            self,
            t_per_tok,
            t_hist,
            t_pred,
            num_modes,
            refine,
            over_predict,
            hidden_dim,
            edge_limit,
            map_enc_radius,
            map_enc_layers,
            dec_attn_order,
            dec_attn_repetitions,
            dec_radius_r,
            dec_radius_s,
            lr,
            weight_decay,
            decay_epochs,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        level_config = {
            'order': dec_attn_order,
            'repetitions': dec_attn_repetitions,
            'radius_r': dec_radius_r,
            'radius_s': dec_radius_s,
            'edge_limit': edge_limit
        }

        mode_config = {
            'num_modes': num_modes,
            'pred_steps': t_pred // t_per_tok,
        }

        self.t_per_tok = t_per_tok
        self.t_hist = t_hist
        self.t_pred = t_pred
        self.num_modes = num_modes
        self.refine = int(refine)
        self.over_predict = int(over_predict)

        self.map_encoder = QCNetMapEncoder(
            hidden_dim=hidden_dim,
            num_historical_steps=t_hist,
            pl2pl_radius=map_enc_radius,
            num_layers=map_enc_layers
        )

        self.proposer = DonutLayer(
            hidden_dim=hidden_dim,
            t_per_tok=t_per_tok,
            type_count=10,
            over_predict=self.over_predict,
            refine=False,
            has_feature_input=False,
            has_feature_output=self.refine,
            has_prob_output=not self.refine,
            **level_config,
            mode_config=mode_config
        )
        if self.refine:
            self.refiner = DonutLayer(
                hidden_dim=hidden_dim,
                t_per_tok=t_per_tok,
                type_count=10,
                over_predict=self.over_predict,
                refine=True,
                has_feature_input=True,
                has_feature_output=False,
                has_prob_output=True,
                **level_config,
                mode_config=mode_config
            )
        self.to_pi = MLPLayer(input_dim=hidden_dim,
                              hidden_dim=hidden_dim, output_dim=1)

        self.loss_fn_pos = LaplaceNLLLoss(reduction='none')
        self.loss_fn_head = VonMisesNLLLoss(reduction='none')

        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_epochs = decay_epochs

        self.val_metrics = nn.ModuleDict({
            'val_minADE': minADE(max_guesses=num_modes),
            'val_minFDE': minFDE(max_guesses=num_modes),
            'val_Brier': Brier(max_guesses=num_modes),
            'val_MR': MR(max_guesses=num_modes),
        })
        self.train_metrics = nn.ModuleDict({
            'train_minADE': minADE(max_guesses=num_modes),
            'train_minFDE': minFDE(max_guesses=num_modes),
            'train_Brier': Brier(max_guesses=num_modes),
            'train_MR': MR(max_guesses=num_modes),
        })
        self.register_buffer(
            'l_r_mapping', 
            torch.tensor([1.5, 0.0, 0.7, 0.7, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float)
        )

    def forward(self, inp):
        """Run a forward pass.

        The method first encodes the map and the observed agent history. It
        then autoregressively samples the next t_pred steps for every
        agent and returns raw distribution parameters.

        Args:
            inp:
                Batched HeteroData from the DataLoader.

        Returns:
            Tuple[List[Dict[str, List[torch.Tensor]]], torch.Tensor]
            1. A list containing the prediction dictionaries for proposer      (K)List has proposer Dich and Refiner Dich --> Dich has the keys pos, head , scale , conc, each 
            and refiner. The dictionary keys are the parameters of the         value, is a list of 2 values 0 = normal pred and 1 =overprediction , the prediction is a tensor
            Laplace and von Mises distributions for position and heading,       with the shape  [num_agents, num_modes, t_pred, 2] .
            respectively. Each value is again a list with the first entry
            being the normal prediction and the second entry being the
            over-prediction. Position outputs have shape                        (Chitabooran)(The model is forced to scale are conc , so as to get the uncertinity in 
            [num_agents, num_modes, t_pred, 2],                                  Prediction as there are cases where the it cannot predict anything (Car in cross-road)
            heading outputs have shape
            [num_agents, num_modes, t_pred].                                     (IIIIII) pi is the logit of all modes(Do softmax to get the probability spread)

            2. Raw unnormalised mode logits pi with shape
            [num_agents, num_modes].
        """
        map_enc = self.map_encoder(inp)

        x_pos = inp['agent']['position'][:, :self.t_hist, :2].contiguous()
        x_head = inp['agent']['heading'][:, :self.t_hist].contiguous()
        x_mask = inp['agent']['valid_mask'][:, :self.t_hist].contiguous()
        x_batch = inp['agent']['batch']
        x_type = inp['agent']['type'].int()

        pl_x = map_enc['x_pl'][:, 0]
        pl_pos = inp['map_polygon']['position'][:, :2].contiguous()
        pl_head = inp['map_polygon']['orientation'].contiguous()
        pl_batch = inp['map_polygon']['batch']

        num_agents = x_pos.shape[0]
        num_hist = self.t_hist // self.t_per_tok
        num_pred = self.t_pred // self.t_per_tok

        x_pos = x_pos.reshape(num_agents, 1, num_hist, self.t_per_tok, 2)
        x_head = x_head.reshape(num_agents, 1, num_hist, self.t_per_tok)
        x_mask = x_mask.reshape(num_agents, 1, num_hist, self.t_per_tok)
        x_type = x_type.reshape(num_agents, 1, 1)

        # encode history
        xs, proposer_past = self.proposer(
            x_pos[:, :, :-1], x_head[:, :, :-1],
            x_type, x_mask[:, :, :-1], x_batch, 0,
            pl_x, pl_pos, pl_head, pl_batch
        )

        if self.refine:
            xs, refiner_past = self.refiner(
                x_pos[:, :, 1:], x_head[:, :, 1:],
                x_type, x_mask[:, :, 1:], x_batch, 0,
                pl_x, pl_pos, pl_head, pl_batch,
                proposed=(xs['pos'], xs['head']),
                feature_input=xs['feats']
            )

        # make multi-modal
        x_pos = x_pos[:, :, -1:].repeat_interleave(self.num_modes, 1)
        x_head = x_head[:, :, -1:].repeat_interleave(self.num_modes, 1)
        x_mask = x_mask[:, :, -1:].repeat_interleave(self.num_modes, 1)

        pred = [
            {
                'pos': [],
                'head': [],
                'scale': [],
                'conc': []
            }
            for _ in range(self.refine + 1)
        ]

        for pred_step in range(1, num_pred+1):
            xs, proposer_past = self.proposer(
                x_pos, x_head, x_type, x_mask, x_batch, pred_step,
                pl_x, pl_pos, pl_head, pl_batch,
                past=proposer_past
            )
            pred[0]['pos'].append(xs['pos'])
            pred[0]['head'].append(xs['head'])
            pred[0]['scale'].append(xs['scale'])
            pred[0]['conc'].append(1/xs['conc'])
            x_mask = torch.ones_like(x_mask)

            x_pos = xs['pos'][:, :, :, :self.t_per_tok].detach()
            x_pos = x_pos.reshape(
                num_agents, self.num_modes, 1, self.t_per_tok, 2)
            x_head = xs['head'][:, :, :, :self.t_per_tok].detach()
            x_head = x_head.reshape(
                num_agents, self.num_modes, 1, self.t_per_tok)

            if self.refine:
                xs, refiner_past = self.refiner(
                    x_pos, x_head, x_type, x_mask, x_batch, pred_step,
                    pl_x, pl_pos, pl_head, pl_batch,
                    proposed=(xs['pos'], xs['head']),
                    past=refiner_past, feature_input=xs['feats']
                )
                pred[1]['pos'].append(xs['pos'])
                pred[1]['head'].append(xs['head'])
                pred[1]['scale'].append(xs['scale'])
                pred[1]['conc'].append(1/xs['conc'])

                x_mask = torch.ones_like(x_mask)
                x_pos = xs['pos'][:, :, :, :self.t_per_tok].detach()
                x_pos = x_pos.reshape(
                    num_agents, self.num_modes, 1, self.t_per_tok, 2)
                x_head = xs['head'][:, :, :, :self.t_per_tok].detach()
                x_head = x_head.reshape(
                    num_agents, self.num_modes, 1, self.t_per_tok)

        pi = self.to_pi(xs['prob']).squeeze(-1).squeeze(-1)

        for layer_pred in pred:
            for key, val in layer_pred.items():
                val = torch.cat(val, dim=2)
                val = torch.split(val, self.t_per_tok, 3)
                val = [x.flatten(2, 3) for x in val]
                layer_pred[key] = val

        for layer_pred in pred:
            layer_pred['scale'][0] =torch.clamp( 0.1 + \
                torch.cumsum(layer_pred['scale'][0], dim=2),min=1e-2)
            layer_pred['conc'][0] = 1 / \
                torch.clamp((0.02 + torch.cumsum(layer_pred['conc'][0], dim=2)),min=1e-2)
            if self.over_predict:
                layer_pred['scale'][1] = torch.clamp(0.1 + layer_pred['scale'][1],min=1e-2)
                layer_pred['conc'][1] =1 /torch.clamp( (0.02 + layer_pred['conc'][1]),min=1e-2
                                                     )

        return pred, pi

    def compute_reg_loss(
        self,
        best_pred_pos, best_scale_pos, gt_pos,
        best_pred_head, best_conc_head, gt_head,
        pred_mask
    ):

        # x pos
        best = torch.cat([
            best_pred_pos[..., :1],
            best_scale_pos[..., :1]
        ], dim=-1)
        gt = gt_pos[..., :1]
        loss_reg = self.loss_fn_pos(best, gt)

        # y pos
        best = torch.cat([
            best_pred_pos[..., 1:],
            best_scale_pos[..., 1:]
        ], dim=-1)
        gt = gt_pos[..., 1:]
        loss_reg += self.loss_fn_pos(best, gt)

        # head
        best = torch.stack([
            best_pred_head,
            best_conc_head
        ], dim=-1)
        gt = gt_head[..., None]
        loss_reg += self.loss_fn_head(best, gt)

        assert loss_reg.shape[-1] == 1
        loss_reg = pred_mask * loss_reg[..., 0]
        loss_reg = loss_reg.sum(0) / pred_mask.sum(0).clamp_(min=1)
        loss_reg = loss_reg.mean()

        return loss_reg

    def compute_cls_nll(self, pred_pos, scale_pos, gt_pos, pred_head, conc_head, gt_head):
        with torch.no_grad():
            last = torch.cat([
                pred_pos[:, :, -1:, :1],
                scale_pos[:, :, -1:, :1]
            ], dim=-1)
            gt = gt_pos[..., None, -1:, :1]
            nll = self.loss_fn_pos(last, gt)
            last = torch.cat([
                pred_pos[:, :, -1:, 1:],
                scale_pos[:, :, -1:, 1:]
            ], dim=-1)
            gt = gt_pos[..., None, -1:, 1:]
            nll += self.loss_fn_pos(last, gt)
            last = torch.stack([
                pred_head[:, :, -1:],
                conc_head[:, :, -1:]
            ], dim=-1)
            gt = gt_head[..., None, -1:, None]
            nll += self.loss_fn_head(last, gt)
        return nll.detach()

    def compute_pinn_loss(self, pred_pos, pred_head, pred_mask, agent_type, dt=0.1):
        """
        Calculates the physics-informed loss based on the bicycle model constraint.
        Applies constraints ONLY to valid wheeled moving agents and zeroes out 
        the loss for background, static, pedestrian, and unknown tracks.
        
        Args:
            pred_pos: Tensor of shape [batch, num_modes, t_pred, 2] (x, y positions)
            pred_head: Tensor of shape [batch, num_modes, t_pred] (yaw angles)
            pred_mask: Tensor of shape [batch, t_pred] (validity mask)
            agent_type: Tensor of shape [batch] (integer category IDs mapping to AV2 classes)
            dt: Time delta between prediction steps (seconds)
        """
        agent_type = agent_type.flatten().long()
        
        # 1. Define your dataset's integer class mapping standard.
        # Example mapping (adjust indices to match your exact dataloader encoding):
        # 0: Vehicle, 1: Pedestrian, 2: Motorcyclist, 3: Cyclist, 4: Bus, 
        # 5: Static, 6: Background, 7: Construction, 8: Riderless Bicycle, 9: Unknown
        
        # Define wheelbases for ALL categories to keep tensor operations safe
        # (Indices 1, 5, 6, 7, 8, 9 are given dummy values since they will be masked out anyway)
        # Clamp inputs to protect against any unexpected index out-of-bounds exceptions
        safe_agent_type = torch.clamp(agent_type, 0, len(self.l_r_mapping) - 1)
        l_r = self.l_r_mapping[safe_agent_type].view(-1, 1, 1) # [batch, 1, 1]
        
        # 2. Approximate derivatives using finite differences (t to t+1)
        dx = pred_pos[:, :, 1:, 0] - pred_pos[:, :, :-1, 0]
        dy = pred_pos[:, :, 1:, 1] - pred_pos[:, :, :-1, 1]
        
        dhead = pred_head[:, :, 1:] - pred_head[:, :, :-1]
        dhead = torch.atan2(torch.sin(dhead), torch.cos(dhead))
        
        vx = dx / dt
        vy = dy / dt
        v_head = dhead / dt
        
        psi = pred_head[:, :, :-1]
        
        # 3. Compute the kinematic bicycle equation residual
        residual = vy * torch.cos(psi) - vx * torch.sin(psi) - l_r * v_head
        pinn_loss_raw = F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)
        lambda_pinn = 0.1 
        pinn_loss = lambda_pinn * pinn_loss_raw
        
        # 4. INCLUSION MASK: Explicitly isolate valid dynamic wheeled categories
        # Only allow: 0 (Vehicle), 2 (Motorcyclist), 3 (Cyclist), 4 (Bus)
        valid_wheeled_mask = (
            (agent_type == 0) | 
            (agent_type == 2) | 
            (agent_type == 3) | 
            (agent_type == 4)
        ).view(-1, 1, 1).float()
        
        # Combine tracking availability mask with our wheeled physical constraint indicator
        mask = pred_mask[:, 1:].unsqueeze(1) * valid_wheeled_mask
        
        # 5. Calculate final penalized loss
        loss_pinn_masked = pinn_loss * mask
        loss_pinn = loss_pinn_masked.sum() / mask.sum().clamp_(min=1)
        
        return loss_pinn

    def compute_loss(self, data, pred, pi):
      
        pred_mask = data['agent']['predict_mask'][:, -self.t_pred:].clone()

        gt_pos = data['agent']['position'][:, -self.t_pred:, :2]
        gt_head = data['agent']['heading'][:, -self.t_pred:]

        l2_norm = torch.linalg.norm(
            pred[0]['pos'][0] - gt_pos[:, None], dim=-1) * pred_mask[:, None]
        best_mode = l2_norm.mean(-1).argmin(dim=-1)
        index0 = torch.arange(len(best_mode),device=best_mode.device)

        losses = {}
        net_names = ['prop', 'ref'][:self.refine+1]
        over_names = ['', '_over'][:self.over_predict+1]

        for net_i, net_name in enumerate(net_names):
            for over_i, over_name in enumerate(over_names):

                pred_pos = pred[net_i]['pos'][over_i]
                pred_head = pred[net_i]['head'][over_i]
                pred_scale = pred[net_i]['scale'][over_i]
                pred_conc = pred[net_i]['conc'][over_i]

                shift = self.t_per_tok * over_i

                loss_reg = self.compute_reg_loss(
                    pred_pos[index0, best_mode, :self.t_pred-shift],
                    pred_scale[index0, best_mode, :self.t_pred-shift],
                    gt_pos[..., shift:, :],
                    pred_head[index0, best_mode, :self.t_pred-shift],
                    pred_conc[index0, best_mode, :self.t_pred-shift],
                    gt_head[..., shift:],
                    pred_mask[:, shift:]
                )

                losses[f'reg_loss_{net_name}{over_name}'] = loss_reg

        ref_pos = pred[-1]['pos'][0]
        ref_head = pred[-1]['head'][0]
        ref_scale = pred[-1]['scale'][0]
        ref_conc = pred[-1]['conc'][0]

        nll = self.compute_cls_nll(
            ref_pos, ref_scale, gt_pos,
            ref_head, ref_conc, gt_head
        )
        nll = nll[..., 0, 0].detach()

        cls_mask = data['agent']['predict_mask'][:, -1]
        nll = (nll * cls_mask[..., None])

        log_pi = nn.functional.log_softmax(pi, dim=-1)
        cls_loss = -torch.logsumexp(log_pi - nll, dim=-1)
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)

        losses['cls_loss'] = cls_loss

        # Integrate overall PINN Loss 
        # (Computed purely from kinematic constraints, independent of ground reality)
        loss_pinn = self.compute_pinn_loss(
            pred_pos=ref_pos, 
            pred_head=ref_head, 
            pred_mask=pred_mask, 
            agent_type=data['agent']['type']
        )
        losses['pinn_loss'] = loss_pinn
        valid_loss_keys = [k for k in losses.keys() if 'mode_' not in k and k != 'pinn_loss']
        losses['data_loss'] = sum(losses[k] for k in valid_loss_keys)
        
        # total loss is the clean sum of both
        losses['loss'] = losses['data_loss'] + losses['pinn_loss']
        
        # ... [per mode logging loop] ...
        

        with torch.no_grad():
            for m in range(self.num_modes):
                
                # L2 Displacement Error per mode
                mode_l2 = torch.linalg.norm(ref_pos[:, m] - gt_pos, dim=-1) * pred_mask
                losses[f'mode_{m}_l2_dist'] = mode_l2.sum() / pred_mask.sum().clamp_(min=1)

                # NLL regression loss per mode
                losses[f'mode_{m}_reg_nll'] = self.compute_reg_loss(
                    ref_pos[:, m], ref_scale[:, m], gt_pos,
                    ref_head[:, m], ref_conc[:, m], gt_head,
                    pred_mask
                )

                # Per-mode PINN loss 
                # (Computed purely from kinematic constraints, independent of ground reality)
                # Slicing m:m+1 keeps the num_modes dimension = 1 to prevent shape mismatch
                mode_pinn = self.compute_pinn_loss(
                    pred_pos=ref_pos[:, m:m+1], 
                    pred_head=ref_head[:, m:m+1], 
                    pred_mask=pred_mask, 
                    agent_type=data['agent']['type']
                )
                losses[f'mode_{m}_pinn_loss'] = mode_pinn

        return losses

    def eval_pred(self, data, pred, pi, metrics):
        eval_mask = data['agent']['category'] >= 3
        reg_mask = data['agent']['predict_mask'][:, -self.t_pred:]
        valid_mask_eval = reg_mask[eval_mask]
        pred_eval = pred[-1]['pos'][0][eval_mask].detach()
        gt = data['agent']['position'][:, -self.t_pred:, :2]
        gt_eval = gt[eval_mask]
        agent_type = data['agent']['type']
        agent_type_eval = agent_type[eval_mask]
        batch_eval = data['agent']['batch'][eval_mask]

        pi = pi[eval_mask].detach()
        pi_eval = nn.functional.softmax(pi, dim=-1)

        for metric in metrics.values():
            metric.update(pred=pred_eval, target=gt_eval, batch=batch_eval, prob=pi_eval,
                          valid_mask=valid_mask_eval, agent_type=agent_type_eval)
        self.log_dict(
            metrics, prog_bar=True,
            on_step=False, on_epoch=True, batch_size=gt_eval.size(0), sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        pred, pi = self(batch)
        
        # 1. Compute all losses and populate the dictionary
        # (This automatically includes all your mode_{m}_reg_nll and mode_{m}_pinn_loss values)
        losses = self.compute_loss(batch, pred, pi)

        # 2. Isolate the data loss and PINN loss
        data_loss = losses['loss']
        pinn_loss = losses['pinn_loss']
        
        # 3. Combine them so the model trains on BOTH
        total_loss = data_loss + pinn_loss

        # --- GRADIENT VECTOR TRACKING ---
        target_param = list(self.proposer.attentions[-1].parameters())[-1]

        # Compute gradients on the fly (using the isolated losses)
        grad_data = torch.autograd.grad(data_loss, target_param, retain_graph=True, allow_unused=True)[0]
        grad_pinn = torch.autograd.grad(pinn_loss, target_param, retain_graph=True, allow_unused=True)[0]

        if grad_data is not None and grad_pinn is not None:
            if grad_data.abs().sum() > 1e-8 and grad_pinn.abs().sum() > 1e-8:
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad_data.flatten(), 
                    grad_pinn.flatten(), 
                    dim=0
                )
                losses['grad_cosine_similarity'] = cos_sim.detach()
            else:
                losses['grad_cosine_similarity'] = torch.tensor(0.0, device=target_param.device)

        # --- AUTOMATIC LOGGING LOOP ---
        for loss_name, loss_val in losses.items():
            # Show the main loss and pinn_loss on the terminal progress bar
            prog_bar = loss_name in ['loss', 'pinn_loss'] 
            self.log(
                f'train_{loss_name}', loss_val.detach(),
                prog_bar=prog_bar, on_step=True, on_epoch=True, sync_dist=True,
                batch_size=batch.num_graphs
            )
            
        # Log the combined total loss
        self.log(
            'train_total_combined_loss', total_loss.detach(), 
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, 
            batch_size=batch.num_graphs
        )

        # 4. Return the COMBINED loss so the optimizer uses both for the backward pass
        return total_loss
        
    

    def validation_step(self, batch, batch_idx):
        pred, pi = self(batch)
        losses = self.compute_loss(batch, pred, pi)
        for loss_name, loss_val in losses.items():
            prog_bar = loss_name == 'loss'
            self.log(
                f'val_{loss_name}', loss_val.detach(),
                prog_bar=prog_bar, on_step=False, on_epoch=True, sync_dist=True,
                batch_size=batch.num_graphs
            )

        self.eval_pred(batch, pred, pi, self.val_metrics)

        return self.val_metrics

    def test_step(self, data, batch_idx):
        # single
        pred = self(data)
        traj = pred['preds_pos'][-1]
        pi = pred['pi']
        eval_mask = data['agent']['category'] == 3

        traj_eval = traj[eval_mask]
        pi_eval = nn.functional.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        eval_id = list(
            compress(list(chain(*data['agent']['id'])), eval_mask))
        for i in range(data.num_graphs):
            self.test_predictions[data['scenario_id'][i]] = {
                eval_id[i]: (traj_eval[i], pi_eval[i])}

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
            nn.LSTMCell, nn.GRU, nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (
                    module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name,
                      param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.decay_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--t_per_tok', type=int, default=10)
        parser.add_argument('--t_hist', type=int, default=50)
        parser.add_argument('--t_pred', type=int, default=60)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--refine', type=int, default=0)
        parser.add_argument('--over_predict', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--map_enc_layers', type=int, default=1)
        parser.add_argument('--map_enc_radius', type=float, default=50)
        parser.add_argument('--edge_limit', type=float, default=0.9999)
        parser.add_argument('--dec_attn_order', type=str, default='trsm')
        parser.add_argument('--dec_attn_repetitions', type=int, default=1)
        parser.add_argument('--dec_radius_r', type=float, default=50)
        parser.add_argument('--dec_radius_s', type=float, default=50)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--acc_batch_size', type=int, default=64)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--decay_epochs', type=int, default=64)
        return parent_parser