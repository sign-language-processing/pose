#!/usr/bin/env python
import argparse
import os

import cv2
from pose_format.utils.holistic import load_holistic


def load_video_frames(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def pose_video(input_path: str, output_path: str, format: str):
    # Load video frames
    print('Loading video ...')
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = load_video_frames(cap)

    # Perform pose estimation
    print('Estimating pose ...')
    if format == 'mediapipe':
        pose = load_holistic(frames,
                             fps=fps,
                             width=width,
                             height=height,
                             progress=True,
                             additional_holistic_config={'model_complexity': 1})
    else:
        raise NotImplementedError('Pose format not supported')

    # Write
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        pose.write(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format',
                        choices=['mediapipe'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument('-i', required=True, type=str, help='path to input video file')
    parser.add_argument('-o', required=True, type=str, help='path to output pose file')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Video file {args.i} not found")

    pose_video(args.i, args.o, args.format)
