import imgaug.augmenters as iaa
import imgaug as ia

from tqdm import tqdm
from lib.python.pose_format import PoseReader, Pose
from lib.python.pose_format.custom_augment.piecewise_affine_kp import PiecewiseAffineKP

buffer = open("1.pose", "rb").read()
p = PoseReader(buffer).read()

for _ in tqdm(list(range(10))):
    p.normalize(
        dist_p1=("pose_keypoints_2d", 2),  # RShoulder
        dist_p2=("pose_keypoints_2d", 5),  # LShoulder
    )

for _ in tqdm(list(range(10))):
    list(p.to_vectors(["angle", "distance"]))

seq = iaa.Sequential([
    iaa.HorizontalFlip(0.5),  # 50% of poses should be flipped left/right
    PiecewiseAffineKP(scale=(0.01, 0.05)),  # Distort keypoints
    iaa.Affine(
        rotate=(-5, 5),  # Rotate up to 10 degrees each way
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # Stretch or squash up to 50% each direction independently
        shear={"x": (-10, 10), "y": (-10, 10)}  # Shear X/Y up to 16 degrees independently
    ),
    iaa.PerspectiveTransform(scale=(0.0, 0.1))  # Change perspective
])

for _ in tqdm(list(range(10))):
    p.augment2d(seq)

