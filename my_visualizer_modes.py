# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import io
import math
from pathlib import Path
from typing import Final, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from PIL import Image as img
from PIL.Image import Image

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    TrackCategory,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.0
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.7
_PLOT_BOUNDS_BUFFER_M: Final[float] = 30.0

_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"

_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"
_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"  # Orange (History & Main Car)
_FOCAL_PRED_COLOR: Final[str] = "#FF0000"   # Red (Prediction)
_AV_COLOR: Final[str] = "#007672"
_BOUNDING_BOX_ZORDER: Final[int] = 100

# Hex codes for distinct ghost modes
_GHOST_COLORS: Final[List[str]] = [
    "#32CD32",  # Lime Green
    "#1E90FF",  # Dodger Blue
    "#FF1493",  # Deep Pink
    "#00CED1",  # Dark Turquoise
    "#FFD700",  # Gold
    "#9932CC",  # Dark Orchid (This was overriding your car box!)
]

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def visualize_scenario(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    save_path: Path,
) -> None:
    frames: List[Image] = []
    plot_bounds: _PlotBounds = (0, 0, 0, 0)

    for timestep in range(_OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS):
        _, ax = plt.subplots()

        _plot_static_map_elements(scenario_static_map)
        cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds

        plt.xlim(
            plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.ylim(
            plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M,
            plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M,
        )
        plt.gca().set_aspect("equal", adjustable="box")

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    vid_path = str(save_path.parents[0] / f"{save_path.stem}.mp4")
    video = cv2.VideoWriter(vid_path, fourcc, fps=10, frameSize=frames[0].size)
    for i in range(len(frames)):
        frame_temp = frames[i].copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()


def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )


def _plot_actor_tracks(
    ax: Axes, scenario: ArgoverseScenario, timestep: int
) -> Optional[_PlotBounds]:
    track_bounds = None
    for track in scenario.tracks:
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        actor_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        track_color = _DEFAULT_ACTOR_COLOR
        is_ghost = False

        if track.category == TrackCategory.FOCAL_TRACK:
            is_ghost = "_ghostmode_" in track.track_id
            
            if is_ghost:
                mode_idx = int(track.track_id.split("_ghostmode_")[1])
                target_color = _GHOST_COLORS[mode_idx % len(_GHOST_COLORS)]
                alpha_val = 0.5
            else:
                target_color = _FOCAL_PRED_COLOR
                alpha_val = 1.0
                
                # Only calculate bounds based on main trajectory
                x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
                y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
                track_bounds = (x_min, x_max, y_min, y_max)
            
            # 1. History
            hist_mask = actor_timesteps < _OBS_DURATION_TIMESTEPS
            if hist_mask.any():
                hist_col = target_color if is_ghost else _FOCAL_AGENT_COLOR
                _plot_polylines([actor_trajectory[hist_mask]], color=hist_col, line_width=2, alpha=alpha_val)
            
            # 2. Prediction
            pred_mask = actor_timesteps >= (_OBS_DURATION_TIMESTEPS - 1)
            if pred_mask.sum() > 1:
                _plot_polylines([actor_trajectory[pred_mask]], color=target_color, line_width=2, style="--", alpha=alpha_val)
            
            track_color = _FOCAL_AGENT_COLOR if (timestep < _OBS_DURATION_TIMESTEPS and not is_ghost) else _FOCAL_AGENT_COLOR

        elif track.track_id == "AV":
            track_color = _AV_COLOR
        elif track.object_type in _STATIC_OBJECT_TYPES:
            continue

        # THE FIX: If this track is a ghost prediction, skip drawing the bounding box!
        if is_ghost:
            continue

        # Plot bounding boxes (Only the real vehicles reach this point now)
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax, actor_trajectory[-1], actor_headings[-1], track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
            )
        elif track.object_type in [ObjectType.CYCLIST, ObjectType.MOTORCYCLIST]:
            _plot_actor_bounding_box(
                ax, actor_trajectory[-1], actor_headings[-1], track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
            )
        else:
            plt.plot(
                actor_trajectory[-1, 0], actor_trajectory[-1, 1], "o",
                color=track_color, markersize=4,
            )

    return track_bounds


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    for polyline in polylines:
        plt.plot(
            polyline[:, 0], polyline[:, 1], style,
            linewidth=line_width, color=color, alpha=alpha,
        )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


def _plot_actor_bounding_box(
    ax: Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
) -> None:
    bbox_length, bbox_width = bbox_size
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y), bbox_length, bbox_width,
        angle=np.degrees(heading), color=color, zorder=_BOUNDING_BOX_ZORDER,
    )
    ax.add_patch(vehicle_bounding_box)