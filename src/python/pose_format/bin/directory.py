import argparse
from pathlib import Path
from pose_format.bin.pose_estimation import pose_video, parse_additional_config
from typing import List
import logging
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import psutil
import os
from functools import partial

# Note: untested other than .mp4. Support for .webm may have issues: https://github.com/sign-language-processing/pose/pull/126
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"]


def find_videos_with_missing_pose_files(
    directory: Path,
    video_suffixes: List[str] = None,
    recursive: bool = False,
    keep_video_suffixes: bool = False,
) -> List[Path]:
    """
    Finds videos with missing .pose files.

    Parameters
    ----------
    directory: Path,
        Directory to search for videos in.
    video_suffixes:  List[str], optional
        Suffixes to look for, e.g. [".mp4", ".webm"]. If None, will use _SUPPORTED_VIDEO_FORMATS
    recursive: bool, optional
        Whether to look for video files recursively, or just the top-level. Defaults to false.
    keep_video_suffixes: bool, optional
        If true, when checking will append .pose suffix (e.g. foo.mp4->foo.mp4.pose, foo.webm->foo.webm.pose),
        If false, will replace it (foo.mp4 becomes foo.pose, and foo.webm ALSO becomes foo.pose).
        Default is false, which can cause name collisions.

    Returns
    -------
    List[Path]
        List of video paths without corresponding .pose files.
    """

    # Prevents the common gotcha with mutable default arg lists:
    # https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
    if video_suffixes is None:
        video_suffixes = SUPPORTED_VIDEO_FORMATS

    glob_method = getattr(directory, "rglob" if recursive else "glob")
    all_files = list(glob_method(f"*"))
    video_files = [path for path in all_files if path.suffix in video_suffixes]
    pose_files = {path for path in all_files if path.suffix == ".pose"}

    videos_with_missing_pose_files = []

    for vid_path in video_files:
        corresponding_pose = get_corresponding_pose_path(video_path=vid_path, keep_video_suffixes=keep_video_suffixes)
        if corresponding_pose not in pose_files:
            videos_with_missing_pose_files.append(vid_path)

    return videos_with_missing_pose_files


def get_corresponding_pose_path(video_path: Path, keep_video_suffixes: bool = False) -> Path:
    """
    Given a video path, and whether to keep the suffix, returns the expected corresponding path with .pose extension.

    Parameters
    ----------
    video_path : Path
        Path to a video file
    keep_video_suffixes : bool, optional
        Whether to keep suffix (e.g. foo.mp4 -> foo.mp4.pose)
        or replace (foo.mp4->foo.pose). Defaults to replace.

    Returns
    -------
    Path
        pathlib Path
    """
    if keep_video_suffixes:
        return video_path.with_name(f"{video_path.name}.pose")
    return video_path.with_suffix(".pose")


def process_video(keep_video_suffixes: bool, pose_format: str, additional_config: dict, vid_path: Path) -> bool:
    cpu_num = psutil.cpu_num() if hasattr(psutil, "cpu_num") else (
        os.sched_getcpu()) if hasattr(os, 'sched_getcpu') else "N/A"
    print(f'Estimating {vid_path} on CPU {cpu_num}')

    try:
        pose_path = get_corresponding_pose_path(video_path=vid_path, keep_video_suffixes=keep_video_suffixes)
        if pose_path.is_file():
            print(f"Skipping {vid_path}, corresponding .pose file already created.")
        else:
            # pose_video function expects string, and passes it unchanged to cv2.VideoCapture(input_path)
            # if you give cv2.VideoCapture(input_path) a Path it crashes on older versions.
            # https://github.com/opencv/opencv/issues/15731
            pose_video(str(vid_path.resolve()), str(pose_path.resolve()), pose_format, additional_config, progress=False)
            return True
            
    except ValueError as e:
        print(f"ValueError on {vid_path}")
        logging.exception(e)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--format",
        choices=["mediapipe"],
        default="mediapipe",
        type=str,
        help="type of pose estimation to use",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Directory to search for videos in",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Whether to search for videos recursively",
    )
    parser.add_argument(
        "--keep-video-suffixes",
        action="store_true",
        help="Whether to drop the video extension (output for foo.mp4 becomes foo.pose, and foo.webm ALSO becomes foo.pose) or append to it (foo.mp4 becomes foo.mp4.pose, foo.webm output is foo.webm.pose). If there are multiple videos with the same basename but different extensions, this will create a .pose file for each. Otherwise only the first video will be posed.",
    )
    parser.add_argument(
        "--video-suffixes",
        type=str,
        choices=SUPPORTED_VIDEO_FORMATS,
        default=SUPPORTED_VIDEO_FORMATS,
        help="Video extensions to search for. Defaults to searching for all supported.",
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=1, 
        help="Number of multiprocessing workers.", 
        required=False
    )
    parser.add_argument(
        "--additional-config",
        type=str,
        help="additional configuration for the pose estimator",
    )
    args = parser.parse_args()

    videos_with_missing_pose_files = find_videos_with_missing_pose_files(
        args.directory,
        video_suffixes=args.video_suffixes,
        recursive=args.recursive,
        keep_video_suffixes=args.keep_video_suffixes,
    )

    print(f"Found {len(videos_with_missing_pose_files)} videos missing pose files.")

    pose_files_that_will_be_created = {get_corresponding_pose_path(vid_path, args.keep_video_suffixes) for vid_path in videos_with_missing_pose_files}

    if len(pose_files_that_will_be_created) < len(videos_with_missing_pose_files):
        continue_input = input(
            f"With current naming strategy (without --keep-video-suffixes), name collisions will result in only {len(pose_files_that_will_be_created)} .pose files being created. Continue? [y/n]"
        )
        if continue_input.lower() != "y":
            print(f"Exiting. To keep video suffixes and avoid collisions, use --keep-video-suffixes")
            exit()

    additional_config = parse_additional_config(args.additional_config)

    pose_with_no_errors_count = 0

    if args.num_workers == 1:
        print('Process sequentially ...')
    else:
        print(f'Multiprocessing with {args.num_workers} workers on {len(os.sched_getaffinity(0))} available CPUs ...')

    func = partial(process_video, args.keep_video_suffixes, args.format, additional_config)
    for success in process_map(func, videos_with_missing_pose_files, max_workers=args.num_workers):
        if success:
            pose_with_no_errors_count += 1

    print(f"Successfully created pose files for {pose_with_no_errors_count}/{len(videos_with_missing_pose_files)} video files")
