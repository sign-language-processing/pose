import imgaug.augmenters as iaa
from tqdm import tqdm
from lib.python.pose_format import Pose
import numpy as np
import numpy.ma as ma

from lib.python.pose_format.vectorizer import SequenceVectorizer, DistanceVectorizer, AngleVectorizer

iterations = 10

buffer = open("1.pose", "rb").read()
p = Pose.read(buffer)

print("Unpack file V0.0")
for _ in tqdm(range(iterations), total=iterations):
    Pose.read(buffer)

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

print("Vectorize")
aggregator = SequenceVectorizer([DistanceVectorizer()])
for _ in tqdm(range(iterations), total=iterations):
    p.to_vectors(aggregator)


#
# print("Augment")
# for _ in tqdm(range(iterations), total=iterations):
#     seq = iaa.Sequential([])
#     p.augment2d(seq)
