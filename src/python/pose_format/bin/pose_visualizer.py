#!/usr/bin/env python

import argparse
import os

import numpy as np
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import pose_normalization_info


def visualize_pose(pose_path: str, video_path: str, normalize=False):
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    if normalize:
        pose = pose.normalize(pose_normalization_info(pose.header))

        new_width = 500
        shift = 1.25
        shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
        pose.body.data = (pose.body.data + shift_vec) * new_width
        pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    v = PoseVisualizer(pose)

    v.save_video(video_path, v.draw())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')
    parser.add_argument('-o', required=True, type=str, help='path to output video file')
    parser.add_argument('--normalize', action='store_true', help='Normalize pose before visualization')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Pose file {args.i} not found")

    visualize_pose(args.i, args.o, args.normalize)
