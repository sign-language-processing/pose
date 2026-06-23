import sys
from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma
import pytest

# Stub the MMPose package and its dependencies before our module is imported.
# mmposewholebody.py does `from mmpose.apis import MMPoseInferencer` at module level,
# so sys.modules must be populated before the first import of that module.
for _mod in ["mmpose", "mmpose.apis", "mmcv", "mmengine", "mmdet"]:
    sys.modules.setdefault(_mod, MagicMock())

from pose_format.utils import mmposewholebody  # noqa: E402 — must come after the stubs above
from pose_format.utils.mmposewholebody import estimate_mmpose_wholebody  # noqa: E402
from pose_format.utils.cocowholebody133_header import cocowholebody_components  # noqa: E402

NUM_KEYPOINTS = 133


# ---------------------------------------------------------------------------
# Header / components tests — no MMPose installation required
# ---------------------------------------------------------------------------

def test_components_total_keypoints():
    assert sum(len(c.points) for c in cocowholebody_components()) == NUM_KEYPOINTS


def test_components_names():
    names = [c.name for c in cocowholebody_components()]
    assert names == ["BODY", "FACE", "LEFT_HAND", "RIGHT_HAND"]


def test_components_point_format():
    for c in cocowholebody_components():
        assert c.format == "XYC"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_result(num_keypoints: int = NUM_KEYPOINTS):
    """Single-frame MMPose result with one detected person."""
    return {
        "predictions": [[{
            "keypoints": np.random.rand(num_keypoints, 2).tolist(),
            "keypoint_scores": np.random.rand(num_keypoints).tolist(),
        }]]
    }


def _empty_result():
    """Single-frame MMPose result with no detected person."""
    return {"predictions": []}


def _make_inferencer(results):
    """Return a patched MMPoseInferencer class whose instance yields `results`."""
    fake_instance = MagicMock()
    fake_instance.return_value = iter(results)
    return MagicMock(return_value=fake_instance)


# ---------------------------------------------------------------------------
# Loader tests (MMPoseInferencer is mocked)
# ---------------------------------------------------------------------------

def test_load_shape(monkeypatch, tmp_path):
    """Output Pose has the right frame/keypoint shape."""
    monkeypatch.setattr(mmposewholebody, "MMPoseInferencer",
                        _make_inferencer([_fake_result(), _fake_result(), _fake_result()]))
    pose = estimate_mmpose_wholebody(str(tmp_path / "video.mp4"), fps=25.0, width=1280, height=720)

    assert pose.body.data.shape == (3, 1, NUM_KEYPOINTS, 2)
    assert pose.body.fps == 25.0
    assert pose.header.dimensions.width == 1280
    assert pose.header.dimensions.height == 720


def test_load_component_names(monkeypatch, tmp_path):
    monkeypatch.setattr(mmposewholebody, "MMPoseInferencer",
                        _make_inferencer([_fake_result()]))
    pose = estimate_mmpose_wholebody(str(tmp_path / "video.mp4"))

    assert [c.name for c in pose.header.components] == ["BODY", "FACE", "LEFT_HAND", "RIGHT_HAND"]


def test_empty_frame_is_masked(monkeypatch, tmp_path):
    """Frames with no detection are present in the output but fully masked."""
    results = [_fake_result(), _empty_result(), _fake_result()]
    monkeypatch.setattr(mmposewholebody, "MMPoseInferencer", _make_inferencer(results))
    pose = estimate_mmpose_wholebody(str(tmp_path / "video.mp4"))

    # All three frames must be present so frame count matches the video.
    assert pose.body.data.shape[0] == 3

    # Frame 1 (index 1) must be fully masked; frames 0 and 2 must not be.
    assert pose.body.data[1].mask.all(), "empty frame should be fully masked"
    assert not pose.body.data[0].mask.all(), "detected frame should not be fully masked"
    assert not pose.body.data[2].mask.all(), "detected frame should not be fully masked"


def test_all_empty_frames(monkeypatch, tmp_path):
    """A video where no person is ever detected produces a fully masked Pose."""
    results = [_empty_result(), _empty_result()]
    monkeypatch.setattr(mmposewholebody, "MMPoseInferencer", _make_inferencer(results))
    pose = estimate_mmpose_wholebody(str(tmp_path / "video.mp4"))

    assert pose.body.data.shape[0] == 2
    assert pose.body.data.mask.all()


def test_version_default(monkeypatch, tmp_path):
    monkeypatch.setattr(mmposewholebody, "MMPoseInferencer",
                        _make_inferencer([_fake_result()]))
    pose = estimate_mmpose_wholebody(str(tmp_path / "video.mp4"))
    assert pose.header.version == 0.2
