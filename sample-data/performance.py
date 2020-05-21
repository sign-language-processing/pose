import itertools

import imgaug.augmenters as iaa
import torch
from tqdm import tqdm
from lib.python.pose_format import Pose
import numpy as np
import numpy.ma as ma
from lib.python.pose_format.vectorizer import SequenceVectorizer, DistanceVectorizer, AngleVectorizer, ZeroVectorizer

iterations = 200

buffer = open("1.pose", "rb").read()
p = Pose.read(buffer)
#
# print("Unpack file V0.0")
# for _ in tqdm(range(iterations), total=iterations):
#     Pose.read(buffer)
#
# p.write("test.pose")
buffer = open("test.pose", "rb").read()

print("Unpack file V0.1")
for _ in tqdm(range(iterations), total=iterations):
    Pose.read(buffer)

info = p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
)

#
# print("Save File")
# for _ in tqdm(range(iterations), total=iterations):
#     p.write("test.pose")
#
# print("Focus Pose")
# for _ in tqdm(range(iterations), total=iterations):
#     p.focus()
#
#
# # Fake all limbs
# all_limbs = list(itertools.product(range(21), range(21)))
# for component in p.header.components:
#     component.limbs = all_limbs
#
# print("Normalize Pose")
# for _ in tqdm(range(iterations), total=iterations):
#     p.normalize(info)
#
# print("Vectorize Nothing")
# vectorizer = SequenceVectorizer([ZeroVectorizer()])
# for _ in tqdm(range(iterations), total=iterations):
#     p.to_vectors(vectorizer)
#
# print("Vectorize Distance")
# vectorizer = SequenceVectorizer([DistanceVectorizer()])
# for _ in tqdm(range(iterations), total=iterations):
#     p.to_vectors(vectorizer)
#
# print("Vectorize Angle")
# vectorizer = SequenceVectorizer([AngleVectorizer()])
# for _ in tqdm(range(iterations), total=iterations):
#     p.to_vectors(vectorizer)
#
# print("Vectorize")
# vectorizer = SequenceVectorizer([DistanceVectorizer(), AngleVectorizer()])
# for _ in tqdm(range(iterations), total=iterations):
#     p.to_vectors(vectorizer)
#
#
# vectors = p.to_vectors(vectorizer)
#
# print("Get Specific Components")
# for _ in tqdm(range(iterations), total=iterations):
#     p.get_components(["hand_left_keypoints_2d", "hand_right_keypoints_2d"])


p.torch()

print("Torch")
test = torch.zeros(10000)
for _ in tqdm(range(iterations), total=iterations):
    div = torch.div(test, test)

print("Torch out")
test = torch.zeros(10000)
for _ in tqdm(range(iterations), total=iterations):
    div = torch.div(test, test, out=test)

print("Torch fix")
test = torch.zeros(10000)
for _ in tqdm(range(iterations), total=iterations):
    div = torch.div(test, test, out=test)
    div[div != div] = 0

# print("Torch no grad")
# with torch.no_grad():
#     for _ in tqdm(range(iterations), total=iterations):
#         a = torch.matmul(torch_arr, torch_transform)


# print("Interpolate to same FPS")
# for _ in tqdm(range(iterations), total=iterations):
#     p.interpolate(new_fps=p.body.fps)
#
# print("Interpolate to half FPS")
# for _ in tqdm(range(iterations), total=iterations):
#     p.interpolate(new_fps=p.body.fps / 2)
#
# print("Augment Local")
# for _ in tqdm(range(iterations), total=iterations):
#     p.augment_vectors(vectors)
#
print("Augment Affine")
for _ in tqdm(range(iterations), total=iterations):
    p.augment2d()
#
# print("Augment imgaug empty")
# augmentor = iaa.Sequential([])
# for _ in tqdm(range(iterations), total=iterations):
#     p.augment2d_imgaug(augmentor)
#
# print("Augment imgaug affine")
# augmentor = iaa.Affine(
#     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#     rotate=(-25, 25),
#     shear=(-8, 8)
# )
# for _ in tqdm(range(iterations), total=iterations):
#     p.augment2d_imgaug(augmentor)
