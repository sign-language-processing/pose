import json
import tempfile
import pytest
import numpy as np

from pose_format.utils.alphapose import load_alphapose_wholebody_from_json, reorder_136_kpts
from pose_format.utils.alphapose_133 import (
    load_alphapose_wholebody_from_json as load_alphapose_133_from_json,
    reorder_133_kpts,
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
# 136-keypoint (default) loader tests
# ---------------------------------------------------------------------------

def test_load_alphapose_136_json_shape():
    path = _write_json(_make_json_format_a(136, n_frames=3))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.data.shape == (3, 1, 136, 2)
    assert pose.body.confidence.shape == (3, 1, 136)


def test_load_alphapose_136_json_defaults():
    path = _write_json(_make_json_format_a(136))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.fps == 24
    assert pose.header.dimensions.width == 1000
    assert pose.header.dimensions.height == 1000


def test_load_alphapose_136_json_metadata_override():
    path = _write_json(_make_json_format_b(136, fps=60.0, width=1920, height=1080))
    pose = load_alphapose_wholebody_from_json(path)
    assert pose.body.fps == 60.0
    assert pose.header.dimensions.width == 1920
    assert pose.header.dimensions.height == 1080


def test_load_alphapose_136_rejects_133():
    path = _write_json(_make_json_format_a(133))
    with pytest.raises(ValueError, match="133-keypoint"):
        load_alphapose_wholebody_from_json(path)


def test_load_alphapose_136_rejects_unknown_count():
    path = _write_json(_make_json_format_a(100))
    with pytest.raises(ValueError):
        load_alphapose_wholebody_from_json(path)


def test_load_alphapose_136_component_names():
    path = _write_json(_make_json_format_a(136))
    pose = load_alphapose_wholebody_from_json(path)
    names = [c.name for c in pose.header.components]
    assert names == ["BODY_136", "FACE_136", "LEFT_HAND_136", "RIGHT_HAND_136"]


# ---------------------------------------------------------------------------
# 133-keypoint loader tests
# ---------------------------------------------------------------------------

def test_load_alphapose_133_json_shape():
    path = _write_json(_make_json_format_a(133, n_frames=3))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.data.shape == (3, 1, 133, 2)
    assert pose.body.confidence.shape == (3, 1, 133)


def test_load_alphapose_133_json_defaults():
    path = _write_json(_make_json_format_a(133))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.fps == 24
    assert pose.header.dimensions.width == 1000
    assert pose.header.dimensions.height == 1000


def test_load_alphapose_133_json_metadata_override():
    path = _write_json(_make_json_format_b(133, fps=25.0, width=720, height=576))
    pose = load_alphapose_133_from_json(path)
    assert pose.body.fps == 25.0
    assert pose.header.dimensions.width == 720
    assert pose.header.dimensions.height == 576


def test_load_alphapose_133_rejects_136():
    path = _write_json(_make_json_format_a(136))
    with pytest.raises(ValueError, match="136-keypoint"):
        load_alphapose_133_from_json(path)


def test_load_alphapose_133_component_names():
    path = _write_json(_make_json_format_a(133))
    pose = load_alphapose_133_from_json(path)
    names = [c.name for c in pose.header.components]
    assert names == ["BODY_133", "FACE_133", "LEFT_HAND_133", "RIGHT_HAND_133"]


# ---------------------------------------------------------------------------
# Keypoint reordering tests
# ---------------------------------------------------------------------------

def test_reorder_136_kpts_body_placement():
    xy = np.arange(136 * 2).reshape(136, 2).astype(float)
    conf = np.arange(136, dtype=float)
    xy_out, conf_out = reorder_136_kpts(xy, conf)
    # body is indices 0-25, should be first 26 rows
    np.testing.assert_array_equal(xy_out[:26], xy[0:26])
    np.testing.assert_array_equal(conf_out[:26], conf[0:26])


def test_reorder_136_kpts_face_placement():
    xy = np.arange(136 * 2).reshape(136, 2).astype(float)
    conf = np.arange(136, dtype=float)
    xy_out, conf_out = reorder_136_kpts(xy, conf)
    # face is indices 26-93, should be next 68 rows
    np.testing.assert_array_equal(xy_out[26:94], xy[26:94])
    np.testing.assert_array_equal(conf_out[26:94], conf[26:94])


def test_reorder_136_kpts_hands_placement():
    xy = np.arange(136 * 2).reshape(136, 2).astype(float)
    conf = np.arange(136, dtype=float)
    xy_out, conf_out = reorder_136_kpts(xy, conf)
    # left hand 94-114, right hand 115-135
    np.testing.assert_array_equal(xy_out[94:115], xy[94:115])
    np.testing.assert_array_equal(xy_out[115:136], xy[115:136])


def test_reorder_133_kpts_body_placement():
    xy = np.arange(133 * 2).reshape(133, 2).astype(float)
    conf = np.arange(133, dtype=float)
    xy_out, conf_out = reorder_133_kpts(xy, conf)
    # body is indices 0-22
    np.testing.assert_array_equal(xy_out[:23], xy[0:23])
    np.testing.assert_array_equal(conf_out[:23], conf[0:23])


def test_reorder_133_kpts_face_placement():
    xy = np.arange(133 * 2).reshape(133, 2).astype(float)
    conf = np.arange(133, dtype=float)
    xy_out, conf_out = reorder_133_kpts(xy, conf)
    # face is indices 23-90
    np.testing.assert_array_equal(xy_out[23:91], xy[23:91])
    np.testing.assert_array_equal(conf_out[23:91], conf[23:91])


def test_reorder_133_kpts_hands_placement():
    xy = np.arange(133 * 2).reshape(133, 2).astype(float)
    conf = np.arange(133, dtype=float)
    xy_out, conf_out = reorder_133_kpts(xy, conf)
    # left hand 91-111, right hand 112-132
    np.testing.assert_array_equal(xy_out[91:112], xy[91:112])
    np.testing.assert_array_equal(xy_out[112:133], xy[112:133])
