from os import listdir, path

from tqdm import tqdm

from lib.python.pose_format import Pose

directory = "/home/nlp/amit/PhD/meta-scholar/datasets/SLCrawl/versions/SpreadTheSign/OpenPose/BODY_25"

old_dir = path.join(directory, "pose_files_v0")
new_dir = path.join(directory, "pose_files")

files = set(listdir(old_dir))

already_done = listdir(new_dir)
for f in already_done:
    files.remove(f)

for f in tqdm(files):
    buffer = open(path.join(old_dir, f), "rb").read()
    Pose.read(buffer).write(path.join(new_dir, f))
