from lib.python.pose_format import PoseReader
import imgaug.augmenters as iaa

buffer = open("1.pose", "rb").read()
p = PoseReader(buffer).read()

# Focus Pose
p.focus_pose()

# Normalize
p.normalize(
    dist_p1=("pose_keypoints_2d", 2),  # RShoulder
    dist_p2=("pose_keypoints_2d", 5),  # LShoulder
)

# Vectorize
list(p.to_vectors(["angle", "distance"]))

# Augment
seq = iaa.Sequential([])
p.augment2d(seq)
