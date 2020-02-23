# Get JSON File
import json
from os import listdir, path

from format.python.src.main import Pose

json_dir = "json"
json_files = list(listdir(json_dir))

frames = [json.load(open(path.join(json_dir, file_name), "r")) for file_name in json_files]

poses = [Pose.from_openpose_json(frame) for frame in frames]

# Save imgs
for f_name, pose in zip(json_files, poses):
    bin_f_name = ".".join(f_name.split(".")[:-1]) + ".pose"
    pose.save(path.join("imgs", bin_f_name))


# Save video
video = Pose(poses[0].header, fps=8)
for pose in poses:
    video.add_frame(pose.body["frames"][0])
video.save(path.join("video", "sample.pose"))