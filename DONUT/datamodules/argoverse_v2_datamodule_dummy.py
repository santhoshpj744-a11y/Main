from typing import Callable, Optional

import torch
import pytorch_lightning as pl

from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from datasets import ArgoverseV2Dataset


class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_raw_dir: Optional[str] = None,
        val_raw_dir: Optional[str] = None,
        test_raw_dir: Optional[str] = None,
        train_processed_dir: Optional[str] = None,
        val_processed_dir: Optional[str] = None,
        test_processed_dir: Optional[str] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        dataset_fraction: float = 0.0001,
        **kwargs
    ) -> None:

        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.persistent_workers = (
            persistent_workers and num_workers > 0
        )

        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir

        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        # 0.0001 = 0.01% of the full dataset
        self.dataset_fraction = dataset_fraction

    # -------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------

    def prepare_data(self):

        ArgoverseV2Dataset(
            self.data_root,
            'train',
            self.train_raw_dir,
            self.train_processed_dir,
            self.train_transform
        )

        ArgoverseV2Dataset(
            self.data_root,
            'val',
            self.val_raw_dir,
            self.val_processed_dir,
            self.val_transform
        )

        ArgoverseV2Dataset(
            self.data_root,
            'test',
            self.test_raw_dir,
            self.test_processed_dir,
            self.test_transform
        )

    # -------------------------------------------------------
    # Select a small subset
    # -------------------------------------------------------

    def _get_subset(self, dataset, name):

        total_size = len(dataset)

        subset_size = max(
            1,
            int(total_size * self.dataset_fraction)
        )

        # Reproducible random selection
        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            total_size,
            generator=generator
        )[:subset_size].tolist()

        subset = Subset(
            dataset,
            indices
        )

        percentage = (
            100.0 * subset_size / total_size
        )

        print(
            f"{name}: "
            f"{subset_size}/{total_size} samples "
            f"({percentage:.4f}%)"
        )

        return subset

    # -------------------------------------------------------
    # Setup
    # -------------------------------------------------------

    def setup(self, stage: Optional[str] = None):

        train_dataset = ArgoverseV2Dataset(
            self.data_root,
            'train',
            self.train_raw_dir,
            self.train_processed_dir,
            self.train_transform
        )

        val_dataset = ArgoverseV2Dataset(
            self.data_root,
            'val',
            self.val_raw_dir,
            self.val_processed_dir,
            self.val_transform
        )

        test_dataset = ArgoverseV2Dataset(
            self.data_root,
            'test',
            self.test_raw_dir,
            self.test_processed_dir,
            self.test_transform
        )

        # Keep only 0.01%
        self.train_dataset = self._get_subset(
            train_dataset,
            "Train"
        )

        self.val_dataset = self._get_subset(
            val_dataset,
            "Validation"
        )

        self.test_dataset = self._get_subset(
            test_dataset,
            "Test"
        )

    # -------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )