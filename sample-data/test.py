from lib.python.pose_format import Pose
import imgaug.augmenters as iaa
import numpy as np
import numpy.ma as ma

from lib.python.pose_format.vectorizer import SequenceVectorizer, DistanceVectorizer, AngleVectorizer

a = np.zeros((93, 1, 137))
print(np.stack((a,a), axis=3).shape)

buffer = open("1.pose", "rb").read()

p = Pose.read(buffer)

p.write("test.pose")
buffer = open("test.pose", "rb").read()
p = Pose.read(buffer)


print(p.body.data.shape)
print(p.body.confidence.shape)

# Focus Pose
p.focus()

# Normalize
info = p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
)
p.normalize(info)

# Vectorize
aggregator = SequenceVectorizer([DistanceVectorizer()])
p.to_vectors(aggregator)

#
# # Augment
# seq = iaa.Sequential([])
# p.augment2d(seq)
