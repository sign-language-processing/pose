from unittest import TestCase

from pose_format.pose import Pose


class TestPose(TestCase):
    def test_should_be_callable(self):
        assert callable(Pose)
