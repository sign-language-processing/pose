from json import load
import imgaug.augmenters as iaa
from tqdm import tqdm

from lib.python.pose_format import Pose

buffer = open("imgs/video_000000000000_keypoints.pose", "rb").read()
p = Pose.read(buffer)

info = p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
)

html = []

for i in tqdm(list(range(50))):
    new_p = p.augment2d(rotation_std=0.3, scale_std=0.3, shear_std=0.3)
    new_p.normalize(info, scale_factor=500)
    new_p.focus()

    new_p.write("imgs/augmented/" + str(i) + ".pose")

    html.append('<pose-viewer src="sample-data/imgs/augmented/' + str(i) + '.pose"></pose-viewer>')

print("\n".join(html))


