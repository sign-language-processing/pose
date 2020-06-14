from tqdm import tqdm

from pose_format import Pose

from pose_format.numpy import NumPyPoseBody
from pose_format.torch.pose_body import TorchPoseBody

buffer = open("/home/nlp/amit/PhD/meta-scholar/utils/../datasets/SLCrawl/versions/SpreadTheSign/OpenPose/BODY_25/pose_files/655_es.mx_0.pose", "rb").read()

for body in [TorchPoseBody, NumPyPoseBody]:
    p = Pose.read(buffer, body)

    print("original shape", p.body.data.shape)

    for i in tqdm(list(range(20))):
        flat = p.body.flatten()

    print("flat shape", flat.shape)
    print(flat)
    print("\n\n\n")



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
