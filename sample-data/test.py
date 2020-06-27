from pose_format.pose_visualizer import PoseVisualizer

from pose_format import Pose

import numpy as np

from pose_format.torch.pose_body import TorchPoseBody



pose = "/home/nlp/amit/PhD/SpeakerDetection/detector/data/dgs-korpus/poses/95fd3ab2eab442954ed92be5aeeb91d2.pose"
video = "/home/nlp/amit/PhD/meta-scholar/datasets/SLCrawl/versions/SpreadTheSign/videos/655_es-mx_0.mp4"

buffer = open(pose, "rb").read()


p = Pose.read(buffer, TorchPoseBody)


print("flattening...")
print("fps", p.body.fps)
print("frames", len(p.body.data))
f = p.body.flatten()
print(f.shape)
print(f)





# visualizer = PoseVisualizer(p)
# # frame = next(iter(visualizer.draw_on_video(video)))
# # visualizer.save_frame("test.png", frame)
#
# frames = list(visualizer.draw_on_video(video))
# visualizer.save_video("test.mp4", frames)
#
#
# for body in [TorchPoseBody, NumPyPoseBody]:
#     p = Pose.read(buffer, body)
#
#     print("original shape", p.body.data.shape)
#
#     for i in tqdm(list(range(20))):
#         flat = p.body.flatten()
#
#     print("flat shape", flat.shape)
#     print(flat)
#     print("\n\n\n")



#
# # Normalize
# info = p.header.normalization_info(
#     p1=("pose_keypoints_2d", "RShoulder"),
#     p2=("pose_keypoints_2d", "LShoulder")
# )
# p.normalize(info, scale_factor=500)
#
# p.write("654es_mx.pose")


# p.get_components(["hand_left_keypoints_2d", "hand_right_keypoints_2d"])

# PoseVisualizer(p).draw("v0.1")


# Focus Pose
# p.focus()
#
# # Normalize
# info = p.header.normalization_info(
#     p1=("pose_keypoints_2d", "RShoulder"),
#     p2=("pose_keypoints_2d", "LShoulder")
# )
# p.normalize(info)

# #
# # Vectorize
# aggregator = SequenceVectorizer([RelativeAngleVectorizer()])
# vectors = p.to_vectors(aggregator)
# #
# # # Augment Local
# # vectors = p.augment_vectors(vectors)
#
# # Augment Global
# p.augment2d()
#
# # Augment Global ImgAug
# p.augment2d_imgaug(iaa.Sequential([]))
