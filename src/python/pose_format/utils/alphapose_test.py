import json
import tempfile
import pytest
import numpy as np

from pose_format.utils.alphapose import load_alphapose_wholebody_from_json
from pose_format.utils.alphapose_133 import (
    load_alphapose_wholebody_from_json as load_alphapose_133_from_json,
)


def _make_flat_keypoints(n: int, value: float = 1.0):
    return [value] * (n * 3)


def _make_json_format_a(n_keypoints: int, n_frames: int = 3):
    return [
        {"image_id": f"frame_{i:04d}.jpg", "keypoints": _make_flat_keypoints(n_keypoints)}
        for i in range(n_frames)
    ]


def _make_json_format_b(n_keypoints: int, n_frames: int = 3, fps=30.0, width=640, height=480):
    return {
        "frames": _make_json_format_a(n_keypoints, n_frames),
        "metadata": {"fps": fps, "width": width, "height": height},
    }


def _write_json(data) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.flush()
    return f.name


# ---------------------------------------------------------------------------
# Auto-detecting loader (alphapose.py)
# ---------------------------------------------------------------------------

def test_auto_detect_136_shape():
    path = _write_json(_make_json_format_a(136, n_frames=3))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.data.shape == (3, 1, 136, 2)
    assert pose.body.confidence.shape == (3, 1, 136)


def test_auto_detect_133_shape():
    path = _write_json(_make_json_format_a(133, n_frames=3))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.data.shape == (3, 1, 133, 2)
    assert pose.body.confidence.shape == (3, 1, 133)


def test_auto_detect_136_component_names():
    path = _write_json(_make_json_format_a(136))
    pose = load_alphapose_wholebody_from_json(path)
    assert [c.name for c in pose.header.components] == ["BODY_136", "FACE_136", "LEFT_HAND_136", "RIGHT_HAND_136"]


def test_auto_detect_133_component_names():
    path = _write_json(_make_json_format_a(133))
    pose = load_alphapose_wholebody_from_json(path)
    assert [c.name for c in pose.header.components] == ["BODY_133", "FACE_133", "LEFT_HAND_133", "RIGHT_HAND_133"]


def test_auto_detect_rejects_unknown_count():
    path = _write_json(_make_json_format_a(100))
    with pytest.raises(ValueError):
        load_alphapose_wholebody_from_json(path)


# ---------------------------------------------------------------------------
# 136-keypoint loader: defaults and metadata
# ---------------------------------------------------------------------------

def test_load_136_defaults():
    path = _write_json(_make_json_format_a(136))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.fps == 24
    assert pose.header.dimensions.width == 1000
    assert pose.header.dimensions.height == 1000


def test_load_136_metadata_override():
    path = _write_json(_make_json_format_b(136, fps=60.0, width=1920, height=1080))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.fps == 60.0
    assert pose.header.dimensions.width == 1920
    assert pose.header.dimensions.height == 1080


# ---------------------------------------------------------------------------
# 133-keypoint strict loader (alphapose_133.py)
# ---------------------------------------------------------------------------

def test_load_133_shape():
    path = _write_json(_make_json_format_a(133, n_frames=3))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.data.shape == (3, 1, 133, 2)
    assert pose.body.confidence.shape == (3, 1, 133)


def test_load_133_defaults():
    path = _write_json(_make_json_format_a(133))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.fps == 24
    assert pose.header.dimensions.width == 1000
    assert pose.header.dimensions.height == 1000


def test_load_133_metadata_override():
    path = _write_json(_make_json_format_b(133, fps=25.0, width=720, height=576))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.fps == 25.0
    assert pose.header.dimensions.width == 720
    assert pose.header.dimensions.height == 576


def test_load_133_rejects_136():
    path = _write_json(_make_json_format_a(136))
    with pytest.raises(ValueError, match="136-keypoint"):
        load_alphapose_133_from_json(path)


def test_load_133_component_names():
    path = _write_json(_make_json_format_a(133))
    pose = load_alphapose_133_from_json(path)
    assert [c.name for c in pose.header.components] == ["BODY_133", "FACE_133", "LEFT_HAND_133", "RIGHT_HAND_133"]


# ---------------------------------------------------------------------------
# Frame sorting
# ---------------------------------------------------------------------------

def test_frame_sort_order():
    # Image IDs where the frame number is NOT the first integer ("v2/frame_NNN.jpg").
    # A correct sort uses the last integer; an incorrect one would sort by "2" instead.
    def kpts(sentinel):
        # x=sentinel, y=sentinel, conf=1.0 for every keypoint (non-zero conf avoids masking)
        return [v for _ in range(136) for v in (sentinel, sentinel, 1.0)]

    data = [
        {"image_id": "v2/frame_0003.jpg", "keypoints": kpts(3.0)},
        {"image_id": "v2/frame_0001.jpg", "keypoints": kpts(1.0)},
        {"image_id": "v2/frame_0002.jpg", "keypoints": kpts(2.0)},
    ]
    path = _write_json(data)
    pose = load_alphapose_wholebody_from_json(path)
    # Correctly sorted: frame_0001 → [0], frame_0002 → [1], frame_0003 → [2]
    assert pose.body.data[0, 0, 0, 0] == pytest.approx(1.0)
    assert pose.body.data[1, 0, 0, 0] == pytest.approx(2.0)
    assert pose.body.data[2, 0, 0, 0] == pytest.approx(3.0)
