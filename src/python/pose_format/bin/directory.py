import argparse
from pathlib import Path
from pose_format.bin.pose_estimation import pose_video, parse_additional_config
from tqdm import tqdm


_SUPPORTED_VIDEO_FORMATS= [".mp4"] # TODO: add .webm support

def find_missing_pose_files(directory: Path, video_suffix:str=".mp4", recursive:bool=False):
    if recursive:
        vid_files = directory.rglob(f"*{video_suffix}")
    else:
        vid_files = directory.glob(f"*{video_suffix}")

    missing_pose_files = []

    for vid_file in vid_files:
        if not vid_file.with_suffix(".pose").is_file():
            missing_pose_files.append(vid_file)
    return missing_pose_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--format',
                        choices=['mediapipe'],
                        default='mediapipe',
                        type=str,
                        help='type of pose estimation to use')
    parser.add_argument("-d","--directory", type=Path, required=True, help="Directory to search for videos in")
    parser.add_argument("-r", "--recursive", action="store_true", help="Whether to search for videos recursively")
    parser.add_argument("--video_suffix", type=str, 
                        choices=_SUPPORTED_VIDEO_FORMATS,
                        default=".mp4", help="Video extension to search for")
    parser.add_argument('--additional-config', type=str, help='additional configuration for the pose estimator')
    args = parser.parse_args()

    missing_pose_files = find_missing_pose_files(args.directory, video_suffix=args.video_suffix, recursive=args.recursive)
    additional_config = parse_additional_config(args.additional_config)

    for vid_path in tqdm(missing_pose_files):
        pose_video(vid_path, vid_path.with_suffix(".pose"), args.format, additional_config)
