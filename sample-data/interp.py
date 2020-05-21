from lib.python.pose_format import Pose

buffer = open("/home/nlp/amit/PhD/meta-scholar/utils/../datasets/SLCrawl/versions/SpreadTheSign/OpenPose/BODY_25/pose_files/640_ru.by_0.pose", "rb").read()
p = Pose.read(buffer)
print(p.body.data.shape)
p.interpolate(1, kind="cubic")
