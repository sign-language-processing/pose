#!/usr/bin/env python
import argparse
import os
from pose_format.pose import Pose




def pose_info(input_path: str):
    with open(input_path, "rb") as f:
        pose = Pose.read(f.read())

    print(pose)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input pose file')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Pose file {args.i} not found")

    pose_info(args.i)
