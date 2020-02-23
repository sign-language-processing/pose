from json import load
import imgaug.augmenters as iaa
from tqdm import tqdm

from format.python.src.custom_augment.piecewise_affine_kp import PiecewiseAffineKP
from format.python.src.pose import Pose, PoseReader
from format.python.src.utils.openpose import BODY_POINTS

buffer = open("video/sample.pose", "rb").read()
p = PoseReader(buffer).read()
p.focus_pose()

seq = iaa.Sequential([
    # iaa.HorizontalFlip(0.5),  # 50% of poses should be flipped left/right
    PiecewiseAffineKP(scale=(0.01, 0.05)),  # Distort keypoints
    iaa.Affine(
        rotate=(-5, 5),  # Rotate up to 10 degrees each way
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # Stretch or squash up to 50% each direction independently
        shear={"x": (-10, 10), "y": (-10, 10)}  # Shear X/Y up to 16 degrees independently
    ),
    iaa.PerspectiveTransform(scale=(0.0, 0.1)) # Change perspective
])

html = []

for i in tqdm(list(range(50))):
    new_p = p.augment2d(seq)
    new_p.focus_pose()

    new_p.normalize(
        dist_p1=("pose_keypoints_2d", BODY_POINTS.index("RShoulder")),
        dist_p2=("pose_keypoints_2d", BODY_POINTS.index("LShoulder")),
        scale_factor=500)
    new_p.save("video/augmented/" + str(i) + ".pose")

    html.append('<pose-viewer src="sample-data/video/augmented/' + str(i) + '.pose"></pose-viewer>')

print("\n".join(html))
