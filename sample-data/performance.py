import imgaug.augmenters as iaa
from tqdm import tqdm
from lib.python.pose_format import Pose
import numpy as np
import numpy.ma as ma
from lib.python.pose_format.vectorizer import SequenceVectorizer, DistanceVectorizer, AngleVectorizer

iterations = 20

buffer = open("1.pose", "rb").read()
p = Pose.read(buffer)

# print("Unpack file V0.0")
# for _ in tqdm(range(iterations), total=iterations):
#     Pose.read(buffer)

p.write("test.pose")
buffer = open("test.pose", "rb").read()

print("Unpack file V0.1")
for _ in tqdm(range(iterations), total=iterations):
    Pose.read(buffer)

print("Save File")
for _ in tqdm(range(iterations), total=iterations):
    p.write("test.pose")

print("Focus Pose")
for _ in tqdm(range(iterations), total=iterations):
    p.focus()

print("Normalize Pose")
info = p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
)
for _ in tqdm(range(iterations), total=iterations):
    p.normalize(info)

print("Vectorize Distance")
vectorizer = SequenceVectorizer([DistanceVectorizer()])
for _ in tqdm(range(iterations), total=iterations):
    p.to_vectors(vectorizer)

print("Vectorize Angle")
vectorizer = SequenceVectorizer([AngleVectorizer()])
for _ in tqdm(range(iterations), total=iterations):
    p.to_vectors(vectorizer)

print("Vectorize")
vectorizer = SequenceVectorizer([DistanceVectorizer(), AngleVectorizer()])
for _ in tqdm(range(iterations), total=iterations):
    p.to_vectors(vectorizer)

vectors = p.to_vectors(vectorizer)

print("Augment Local")
for _ in tqdm(range(iterations), total=iterations):
    p.augment_vectors(vectors)

print("Augment Affine")
for _ in tqdm(range(iterations), total=iterations):
    p.augment2d()

print("Augment imgaug empty")
augmentor = iaa.Sequential([])
for _ in tqdm(range(iterations), total=iterations):
    p.augment2d_imgaug(augmentor)

print("Augment imgaug affine")
augmentor = iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    rotate=(-25, 25),
    shear=(-8, 8)
)
for _ in tqdm(range(iterations), total=iterations):
    p.augment2d_imgaug(augmentor)
