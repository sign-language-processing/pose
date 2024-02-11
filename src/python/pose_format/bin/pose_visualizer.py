#!/usr/bin/env python

import argparse
import os

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


def visualize_pose(pose_path: str, video_path: str):
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    v = PoseVisualizer(pose)

    v.save_video(video_path, v.draw())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')
    parser.add_argument('-o', required=True, type=str, help='path to output video file')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Pose file {args.i} not found")

    visualize_pose(args.i, args.o)
