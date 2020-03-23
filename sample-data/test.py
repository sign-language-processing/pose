from lib.python.pose_format import Pose
import imgaug.augmenters as iaa
import numpy as np
import numpy.ma as ma

from lib.python.pose_format.pose_visualizer import PoseVisualizer
from lib.python.pose_format.vectorizer import SequenceVectorizer, DistanceVectorizer, AngleVectorizer

buffer = open("1.pose", "rb").read()
p0 = Pose.read(buffer)
PoseVisualizer(p0).draw("v0.0")

p0.write("test.pose")

buffer = open("test.pose", "rb").read()
p = Pose.read(buffer)
# PoseVisualizer(p).draw("v0.1")
#
#
# # Focus Pose
# p.focus()
#
# # Normalize
# info = p.header.normalization_info(
#     p1=("pose_keypoints_2d", "RShoulder"),
#     p2=("pose_keypoints_2d", "LShoulder")
# )
# p.normalize(info)
#
# # Vectorize
# aggregator = SequenceVectorizer([DistanceVectorizer(), AngleVectorizer()])
# vectors = p.to_vectors(aggregator)
#
# # Augment Local
# vectors = p.augment_vectors(vectors)

# Augment Global
p.augment2d()

# Augment Global ImgAug
p.augment2d_imgaug(iaa.Sequential([]))
