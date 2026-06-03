import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np

from pose_format.utils.holistic import load_holistic, mirror_horizontal, FLIPPED_FACE_POINTS

# Committed test image, originally from:
# https://raw.githubusercontent.com/sign/image-to-human-avatar/main/assets/examples/flux/masked.png
IMAGE_PATH = Path(__file__).parent / "data" / "fake_human.png"
SIZE = 512

HOLISTIC_CONFIG = {"model_complexity": 2, "refine_face_landmarks": True, "static_image_mode": True}


def _load_image() -> np.ndarray:
    from PIL import Image

    image = Image.open(IMAGE_PATH).convert("RGB").resize((SIZE, SIZE))
    return np.asarray(image)


def _run(frame: np.ndarray, config: dict = HOLISTIC_CONFIG):
    return load_holistic([frame], fps=1, width=SIZE, height=SIZE,
                         additional_holistic_config=dict(config), pose_workers=1)


# POSE_WORLD_LANDMARKS is left unchanged by mirror_horizontal (mediapipe's world landmarks do not
# mirror with the image), and is too noisy per-joint to compare point-by-point, so it is excluded.
FLIPPING_COMPONENTS = ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]


class TestMirrorHorizontal(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            frame = _load_image()
        except Exception as e:  # pylint: disable=broad-except
            raise unittest.SkipTest(f"Could not load test image: {e}")

        cls.frame = frame
        flipped_frame = np.ascontiguousarray(frame[:, ::-1, :])
        cls.original = _run(frame)
        cls.reference = _run(flipped_frame)        # holistic on the actually-flipped image
        cls.mirrored = mirror_horizontal(cls.original)

    def _assert_matches_flipped(self, mirrored_pose, reference_pose, max_px=10.0):
        # For every flipping component, every point of our mirror must land where holistic landed on
        # the actually-flipped image. A wrong hand swap or face/body index would be off by tens of px.
        header = mirrored_pose.header
        mirrored = mirrored_pose.body.data.filled(0)[0, 0]
        reference = reference_pose.body.data.filled(0)[0, 0]
        mirrored_conf = mirrored_pose.body.confidence[0, 0]
        reference_conf = reference_pose.body.confidence[0, 0]

        for name in FLIPPING_COMPONENTS:
            component = next(c for c in header.components if c.name == name)
            checked = 0
            for point in component.points:
                index = header.get_point_index(name, point)
                if mirrored_conf[index] <= 0.5 or reference_conf[index] <= 0.5:
                    continue
                diff = np.abs(mirrored[index, :2] - reference[index, :2]).max()
                self.assertLess(diff, max_px, f"{name}/{point}: {diff:.2f}px from flipped-image run")
                checked += 1
            self.assertGreater(checked, 0, f"{name}: no confident points to compare")

    def test_rejects_non_holistic_pose(self):
        from pose_format.utils.generic import fake_openpose_pose

        with self.assertRaises(NotImplementedError):
            mirror_horizontal(fake_openpose_pose(num_frames=1))

    def test_face_permutation_is_involution(self):
        self.assertEqual(len(FLIPPED_FACE_POINTS), 478)
        for i, flipped in enumerate(FLIPPED_FACE_POINTS):
            self.assertEqual(FLIPPED_FACE_POINTS[flipped], i)

    def test_header_is_unchanged(self):
        self.assertEqual([c.name for c in self.mirrored.header.components],
                         [c.name for c in self.original.header.components])

    def test_does_not_mutate_input(self):
        # mirror is its own inverse, so applying it twice restores the original data
        restored = mirror_horizontal(self.mirrored)
        np.testing.assert_allclose(restored.body.data.filled(0), self.original.body.data.filled(0))

    def test_confidence_is_a_permutation(self):
        original_conf = self.original.body.confidence[0, 0]
        mirrored_conf = self.mirrored.body.confidence[0, 0]
        np.testing.assert_array_equal(np.sort(original_conf), np.sort(mirrored_conf))

    def test_world_landmarks_unchanged(self):
        index = self.mirrored.header.get_point_index("POSE_WORLD_LANDMARKS", "NOSE")
        np.testing.assert_array_equal(
            self.mirrored.body.data.filled(0)[:, :, index:],
            self.original.body.data.filled(0)[:, :, index:],
        )

    def test_matches_holistic_on_flipped_image(self):
        self._assert_matches_flipped(self.mirrored, self.reference)

    def test_matches_holistic_on_flipped_image_without_refine(self):
        config = {**HOLISTIC_CONFIG, "refine_face_landmarks": False}
        original = _run(self.frame, config)
        reference = _run(np.ascontiguousarray(self.frame[:, ::-1, :]), config)
        mirrored = mirror_horizontal(original)

        face = next(c for c in mirrored.header.components if c.name == "FACE_LANDMARKS")
        self.assertEqual(len(face.points), 468)
        self._assert_matches_flipped(mirrored, reference)


if __name__ == "__main__":
    unittest.main()
