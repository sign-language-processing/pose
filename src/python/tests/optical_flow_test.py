import tempfile

from unittest import TestCase

from matplotlib.testing.compare import compare_images
from pose_format.pose import Pose

from pose_format.numpy.representation.distance import DistanceRepresentation
from pose_format.utils.optical_flow import OpticalFlowCalculator


class TestOpticalFlow(TestCase):
    def test_optical_flow(self):
        calculator = OpticalFlowCalculator(fps=30, distance=DistanceRepresentation())

        with open('data/mediapipe.pose', 'rb') as f:
            pose = Pose.read(f.read())
            pose = pose.get_components(["POSE_LANDMARKS", "RIGHT_HAND_LANDMARKS", "LEFT_HAND_LANDMARKS"])

        flow = calculator(pose.body.data)
        flow = flow.squeeze(axis=1)
        print(flow.shape)

        # Matplotlib heatmap
        import matplotlib.pyplot as plt
        plt.imshow(flow.T)
        plt.tight_layout()

        fp = tempfile.NamedTemporaryFile()
        plt.savefig(fp.name, format='png')

        self.assertTrue(compare_images('data/optical_flow.png', fp.name, 0.001) is None)

