import tempfile
import os
from unittest import TestCase

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


class TestPoseVisualizer(TestCase):
    """
    Test cases for PoseVisualizer functionality.
    """

    def test_save_gif(self):
        """
        Test saving pose visualization as GIF.
        """
        with open("tests/data/mediapipe.pose", "rb") as f:
            pose = Pose.read(f.read())

        v = PoseVisualizer(pose)

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_gif:
            v.save_gif(temp_gif.name, v.draw())
            self.assertTrue(os.path.exists(temp_gif.name))
            self.assertGreater(os.path.getsize(
                temp_gif.name), 0)

    def test_save_png(self):
        """
        Test saving pose visualization as PNG.
        """
        with open("tests/data/mediapipe_long.pose", "rb") as f:
            pose = Pose.read(f.read())

        v = PoseVisualizer(pose)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_png = os.path.join(temp_dir, 'example.png')
            v.save_png(temp_png, v.draw(transparency=True))
            self.assertTrue(os.path.exists(temp_png))
            self.assertGreater(os.path.getsize(temp_png), 0)

    def test_save_mp4(self):
        """
        Test saving pose visualization as MP4 video.
        """
        with open("tests/data/mediapipe_hand_normalized.pose", "rb") as f:
            pose = Pose.read(f.read())

        v = PoseVisualizer(pose)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_mp4:
            v.save_video(temp_mp4.name, v.draw())
            self.assertTrue(os.path.exists(temp_mp4.name))
            self.assertGreater(os.path.getsize(temp_mp4.name), 0)
