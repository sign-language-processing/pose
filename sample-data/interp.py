from format.python.src.pose import PoseReader

buffer = open("video/sample.pose", "rb").read()
p = PoseReader(buffer).read()
p.focus_pose()

print("Original fps", p.body["fps"])
p.interpolate_fps(24)
print("New fps", p.body["fps"])
p.save("video/sample_interp_24.pose")
p.interpolate_fps(60)
print("New fps", p.body["fps"])
p.save("video/sample_interp_60.pose")