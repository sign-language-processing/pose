import math
import torch
from pose_format.torch.representation.inner_angle import InnerAngleRepresentation

from pose_format.pose_visualizer import PoseVisualizer

from pose_format import Pose

# import tensorflow as tf
import numpy as np

from pose_format.numpy.pose_body import NumPyPoseBody
# from pose_format.tensorflow.pose_body import TensorflowPoseBody, TF_POSE_RECORD_DESCRIPTION
from pose_format.torch.masked import MaskedTensor

# p1s = MaskedTensor(tensor=torch.tensor([[0, -1, 0], [0, -1, 0]], dtype=torch.float32))
# p2s = MaskedTensor(tensor=torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32))
# p3s = MaskedTensor(tensor=torch.tensor([[-1, 1, 0], [-1, 1, 0]], dtype=torch.float32))
#
# print(p1s.shape)
#
# inner_angle = InnerAngleRepresentation()
# print(inner_angle(p1s, p2s, p3s))
# print(math.pi * 135 / 180)

#
# pose = "1.pose"
# video = "/home/nlp/amit/PhD/SpeakerDetection/dgs-korpus-google/videos/1413925_1b1.mp4"
#
# buffer = open(pose, "rb").read()
#
# p = Pose.read(buffer, NumPyPoseBody)
#
# p = p.tensorflow()
#
#
# with tf.io.TFRecordWriter("test.tfrecord") as writer:
#     for pose in [p]:
#         features = {
#         }
#         features.update(pose.body.as_tfrecord())
#
#         example = tf.train.Example(features=tf.train.Features(feature=features))
#         writer.write(example.SerializeToString())
#
# features = {}
# features.update(TF_POSE_RECORD_DESCRIPTION)
#
# dataset = tf.data.TFRecordDataset(filenames=["test.tfrecord"])
# dataset = dataset.map(lambda serialized: tf.io.parse_single_example(serialized, features))
#
# for datum in dataset.take(1):
#     pose = TensorflowPoseBody.from_tfrecord(datum)
#     print(pose)

#
# p.body.data = p.body.data[:1]
# p.body.confidence = p.body.confidence[:1]
#
# p = p.bbox()
#
#
buffer = open("/home/nlp/amit/PhD/PoseFormat/sample-data/video/sample.pose", "rb").read()
p = Pose.read(buffer, NumPyPoseBody)
ratio = 256 / p.header.dimensions.width
p.body.data *= np.array([ratio, ratio])
p.header.dimensions.width = 256
p.header.dimensions.height *= ratio
p.header.dimensions.height = int(p.header.dimensions.height)

visualizer = PoseVisualizer(p)
frame = next(iter(visualizer.draw(background_color=(255, 255, 255))))
visualizer.save_frame("test.png", frame)
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
