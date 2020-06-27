from unittest import TestCase

from pose_format.pose import Pose


class TestPose(TestCase):
    def test_call_value_should_be_angle(self):
        assert (callable(Pose))