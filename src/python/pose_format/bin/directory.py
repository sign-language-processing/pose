import argparse
import os

from pose_format.bin.pose_estimation import pose_video
from tqdm import tqdm


def find_missing_pose_files(directory: str):
    all_files = os.listdir(directory)
    mp4_files = [f for f in all_files if f.endswith(".mp4")]
    pose_files = {os.path.splitext(f)[0] for f in all_files if f.endswith(".pose")}
    missing_pose_files = []

    for mp4_file in mp4_files:
        base_name = os.path.splitext(mp4_file)[0]
        if base_name not in pose_files:
            missing_pose_files.append(os.path.join(directory, mp4_file))

    return sorted(missing_pose_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',
                        choices=['mediapipe'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    missing_pose_files = find_missing_pose_files(args.directory)

    for mp4_path in tqdm(missing_pose_files):
        pose_file_name = os.path.splitext(mp4_path)[0] + ".pose"
        pose_video(mp4_path, pose_file_name, 'mediapipe')
