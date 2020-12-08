import numpy as np
from numpy import ma

import pose_format.utils.siren as siren
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

buffer = open("/home/nlp/amit/PhD/PoseFormat/sample-data/1.pose", "rb").read()
p = Pose.read(buffer)
print("Poses loaded")

info = p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
)
p = p.normalize(info, scale_factor=1)
p.body.zero_filled()

net = siren.get_pose_siren(p, total_steps=2000, learning_rate=1e-4, steps_til_summary=100, cuda=True)

new_fps = 12
coords = siren.PoseDataset.get_coords(time=len(p.body.data) / p.body.fps, fps=new_fps)
pred = net(coords).cpu().numpy()

p.body.fps = new_fps
p.body.data = ma.array(pred)
p.body.confidence = np.ones(shape=tuple(pred.shape[:3]))

p.normalize(info, scale_factor=500)
p.focus()

v = PoseVisualizer(p)
v.save_video("reconstructed.mp4", v.draw())
