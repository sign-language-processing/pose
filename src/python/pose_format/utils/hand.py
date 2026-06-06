from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.ma as ma

from pose_format.pose import Pose

HandSide = Literal["LEFT", "RIGHT"]
__all__ = ["estimate_active_hand"]

HAND_POINTS = (
    "WRIST",
    "INDEX_FINGER_MCP",
    "MIDDLE_FINGER_MCP",
    "RING_FINGER_MCP",
    "PINKY_MCP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_TIP",
    "PINKY_TIP",
)

SHORT_CLIP_FRAMES = 50


@dataclass(frozen=True)
class _HandFeatures:
    above_count: float
    above_rate: float
    dist_shoulder: float
    hand_dist_center: float
    hand_motion_sum: float
    hand_conf: float
    body_motion_sum: float


def estimate_active_hand(pose: Pose) -> HandSide:
    """Estimate which hand is active in a MediaPipe holistic pose.

    The heuristic compares torso-normalized wrist geometry, hand landmark confidence, distance from the torso,
    and motion. It uses a short/long clip split: short clips use body-relative wrist motion because summed motion is
    still stable there, while longer clips emphasize tracked hand landmarks and avoid duration-sensitive body-motion
    accumulation.

    This function was generated mostly using AI and was data-driven against a validation sample from the fsboard and
    ChicagoFSWild datasets, where it achieved 100% accuracy, including mirrored-video validation. That validation
    result is a dataset-specific check of this heuristic, not a guarantee for every pose distribution.

    The pose is expected to contain MediaPipe holistic components named ``POSE_LANDMARKS``,
    ``LEFT_HAND_LANDMARKS``, and ``RIGHT_HAND_LANDMARKS``. Only the first tracked person is considered.
    """
    if _is_short_clip(pose):
        return _estimate_short_clip_hand(pose)
    return _estimate_long_clip_hand(pose)


def _idx(pose: Pose, component: str, point: str) -> int:
    return pose.header.get_point_index(component, point)


def _xy_and_conf(pose: Pose, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if pose.body.data.shape[1] == 0:
        raise ValueError("Cannot estimate active hand for a pose with no people")

    xy = ma.filled(pose.body.data[:, 0, indices, :2], np.nan).astype(float)
    conf = ma.filled(pose.body.confidence[:, 0, indices], 0.0).astype(float)
    return np.where(conf[..., None] > 0, xy, np.nan), conf


def _median(values: np.ndarray, default: float = 0.0) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return default
    return float(np.nanmedian(values[finite]))


def _shoulder_scale(pose: Pose) -> float:
    indices = [
        _idx(pose, "POSE_LANDMARKS", "LEFT_SHOULDER"),
        _idx(pose, "POSE_LANDMARKS", "RIGHT_SHOULDER"),
    ]
    xy, conf = _xy_and_conf(pose, indices)
    valid = np.all(conf > 0, axis=1)
    distances = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)
    scale = _median(distances[valid], default=1.0)
    return scale if scale > 1.0 else 1.0


def _torso_center(pose: Pose) -> np.ndarray:
    points = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")
    xy, _ = _xy_and_conf(pose, [_idx(pose, "POSE_LANDMARKS", point) for point in points])

    valid = np.isfinite(xy).all(axis=2)
    totals = np.nansum(np.where(valid[..., None], xy, 0.0), axis=1)
    counts = np.sum(valid, axis=1)

    center = np.full((xy.shape[0], 2), np.nan, dtype=float)
    good = counts > 0
    center[good] = totals[good] / counts[good, None]
    return center


def _motion_sum(points: np.ndarray, conf: np.ndarray, center: np.ndarray, scale: float) -> float:
    if points.shape[0] < 2:
        return 0.0

    relative = points - center[:, None, :]
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1) / scale
    valid = (conf[1:] > 0) & (conf[:-1] > 0)
    return float(np.nansum(np.where(valid, speed, 0.0)))


def _hand_features(pose: Pose, hand: HandSide) -> _HandFeatures:
    scale = _shoulder_scale(pose)
    center = _torso_center(pose)

    wrist_idx = _idx(pose, "POSE_LANDMARKS", f"{hand}_WRIST")
    elbow_idx = _idx(pose, "POSE_LANDMARKS", f"{hand}_ELBOW")
    shoulder_idx = _idx(pose, "POSE_LANDMARKS", f"{hand}_SHOULDER")

    body_xy, body_conf = _xy_and_conf(pose, [wrist_idx, elbow_idx, shoulder_idx])
    wrist = body_xy[:, 0]
    elbow = body_xy[:, 1]
    shoulder = body_xy[:, 2]
    wrist_conf = body_conf[:, 0]
    elbow_conf = body_conf[:, 1]

    visible_arm = (wrist_conf > 0) & (elbow_conf > 0)
    above = visible_arm & (wrist[:, 1] < elbow[:, 1])
    above_count = float(np.sum(above))
    above_rate = above_count / float(np.sum(visible_arm)) if np.any(visible_arm) else 0.0

    dist_shoulder = _median(np.linalg.norm(wrist - shoulder, axis=1)[wrist_conf > 0] / scale)
    body_motion_sum = _motion_sum(wrist[:, None, :], wrist_conf[:, None], center, scale)

    hand_indices = [_idx(pose, f"{hand}_HAND_LANDMARKS", point) for point in HAND_POINTS]
    hand_xy, hand_conf = _xy_and_conf(pose, hand_indices)

    hand_valid = hand_conf > 0
    hand_conf_mean = float(np.mean(hand_conf)) if hand_conf.size else 0.0

    centroid = np.full((hand_xy.shape[0], 2), np.nan, dtype=float)
    frame_counts = np.sum(hand_valid, axis=1)
    frame_has_hand = frame_counts > 0
    centroid[frame_has_hand] = (
        np.nansum(np.where(hand_valid[..., None], hand_xy, 0.0), axis=1)[frame_has_hand]
        / frame_counts[frame_has_hand, None]
    )
    hand_dist_center = _median(np.linalg.norm(centroid - center, axis=1) / scale)
    hand_motion_sum = _motion_sum(hand_xy, hand_conf, center, scale)

    return _HandFeatures(
        above_count=above_count,
        above_rate=above_rate,
        dist_shoulder=dist_shoulder,
        hand_dist_center=hand_dist_center,
        hand_motion_sum=hand_motion_sum,
        hand_conf=hand_conf_mean,
        body_motion_sum=body_motion_sum,
    )


def _short_clip_score(features: _HandFeatures) -> float:
    return (
        -features.dist_shoulder
        + 0.05 * features.hand_motion_sum
        + 2.0 * features.hand_conf
        + 0.5 * features.hand_dist_center
        + 0.2 * features.body_motion_sum
    )


def _long_clip_score(features: _HandFeatures) -> float:
    return (
        -features.dist_shoulder
        + 0.05 * features.hand_motion_sum
        + 2.0 * features.hand_conf
        + 0.5 * features.hand_dist_center
    )


def _estimate_short_clip_hand(pose: Pose) -> HandSide:
    left = _hand_features(pose, "LEFT")
    right = _hand_features(pose, "RIGHT")
    left_score = _short_clip_score(left)
    right_score = _short_clip_score(right)

    if (
        max(left.hand_conf, right.hand_conf) < 0.45
        and max(left.above_rate, right.above_rate) >= 0.9
        and abs(left.body_motion_sum - right.body_motion_sum) > 0.5
    ):
        if left.body_motion_sum > right.body_motion_sum and left.above_rate >= 0.5:
            return "LEFT"
        if right.above_rate >= 0.5:
            return "RIGHT"

    both_raised = abs(left.above_count - right.above_count) <= 1.0 and min(left.above_rate, right.above_rate) >= 0.8
    if both_raised and abs(left_score - right_score) < 0.6:
        hand = _choose_extended_hand(left, right)
        if hand is not None:
            return hand

    return "LEFT" if left_score > right_score else "RIGHT"


def _estimate_long_clip_hand(pose: Pose) -> HandSide:
    left = _hand_features(pose, "LEFT")
    right = _hand_features(pose, "RIGHT")
    left_score = _long_clip_score(left)
    right_score = _long_clip_score(right)

    if left.above_rate >= 0.9 and right.above_rate <= 0.2 and left.hand_conf + 0.05 >= right.hand_conf:
        return "LEFT"
    if right.above_rate >= 0.9 and left.above_rate <= 0.2 and right.hand_conf + 0.05 >= left.hand_conf:
        return "RIGHT"

    both_raised = abs(left.above_count - right.above_count) <= 1.0 and min(left.above_rate, right.above_rate) >= 0.8
    if both_raised and abs(left_score - right_score) < 0.6:
        hand = _choose_extended_hand(left, right)
        if hand is not None:
            return hand

    return "LEFT" if left_score > right_score else "RIGHT"


def _choose_extended_hand(left: _HandFeatures, right: _HandFeatures) -> Optional[HandSide]:
    if abs(left.hand_dist_center - right.hand_dist_center) > 0.4:
        return "LEFT" if left.hand_dist_center > right.hand_dist_center else "RIGHT"

    if right.hand_conf - left.hand_conf >= 0.05 and right.hand_dist_center - left.hand_dist_center >= 0.1:
        return "RIGHT"
    if left.hand_conf - right.hand_conf >= 0.05 and left.hand_dist_center - right.hand_dist_center >= 0.1:
        return "LEFT"
    return None


def _is_short_clip(pose: Pose) -> bool:
    return pose.body.data.shape[0] < SHORT_CLIP_FRAMES
