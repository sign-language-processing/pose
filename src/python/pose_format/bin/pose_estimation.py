#!/usr/bin/env python
import argparse
import os
import cv2


def load_video_frames(cap: cv2.VideoCapture):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def pose_video(input_path: str, output_path: str, format: str, use_cpu: bool, additional_config: dict = {'model_complexity': 1}, progress: bool = True):
    
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
        from pose_format.utils.holistic import load_holistic
        pose = load_holistic(frames,
                             fps=fps,
                             width=width,
                             height=height,
                             progress=progress,
                             additional_holistic_config=additional_config)
    elif format == 'openpose':
        from pose_format.utils.openpose import estimate_and_load_openpose
        pose = estimate_and_load_openpose(input_path, 
                            output_path,   
                            fps=fps, 
                            width=width,
                            height=height)
    elif format == 'openpifpaf':
        from pose_format.utils.openpifpaf import estimate_and_load_openpifpaf
        pose = estimate_and_load_openpifpaf(frames,
                            fps=fps, 
                            use_cpu=use_cpu,
                            width=width,
                            height=height)
    elif format == 'mmposewholebody':
        from pose_format.utils.mmposewholebody import load_mmposewholebody
        pose = load_mmposewholebody(input_path,
                            output_path, 
                            use_cpu=use_cpu,
                            fps=fps,
                            width=width,
                            height=height)
    else:
        raise NotImplementedError('Pose format not supported')

    # Build output filename
    video_name = os.path.splitext(os.path.basename(input_path))[0]   # "SOUP" from ".../SOUP.mp4"
    output_pose_file_path = os.path.join(output_path, f"{video_name}.pose")
    os.makedirs(output_path, exist_ok=True)

    # Write
    print(f'Saving to disk at path {output_pose_file_path}')

    with open(output_pose_file_path, "wb") as f:
        pose.write(f)


def parse_additional_config(config: str):
    if not config:
        return {}
    config = config.split(',')

    def parse_value(value):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return value

    return {k: parse_value(v) for k, v in [c.split('=') for c in config]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='path to input video file')
    parser.add_argument('-o', required=True, type=str, help='path to output pose file')
    parser.add_argument('--format',
                        choices=['mediapipe', 'mmposewholebody', 'openpose', 'openpifpaf'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument('--usecpu', 
                        default=False,
                        type=bool, 
                        help='option to use CPU for pose estimation (if supported)')
    parser.add_argument('--additional-config', type=str, help='additional configuration for the pose estimator')

    args = parser.parse_args()

    if not os.path.exists(args.i):
        raise FileNotFoundError(f"Video file {args.i} not found")

    additional_config = parse_additional_config(args.additional_config)
    pose_video(args.i, args.o, args.format, args.usecpu, additional_config)

    # pip install . && video_to_pose -i como.mp4 -o como1.pose --format mediapipe
    # pip install . && video_to_pose -i como.mp4 -o como2.pose --format mediapipe --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
    # pip install . && video_to_pose -i sparen.mp4 -o sparen.pose --format mediapipe --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
