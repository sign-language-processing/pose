import os

import cv2
import torch
from pose_format.pose import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import reduce_holistic
from pose_format.utils.openpose import OpenPose_Components
from pose_format.utils.pose_converter import convert_pose

try:
    import folder_paths
except ImportError as e:
    raise ImportError("Please make sure to run this node in ComfyUI Context.")


class PoseLoader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and file_parts[-1].lower() == "pose":
                    files.append(f)

        return {
            "required": {
                "file": (sorted(files),),
                "is_reduce_holistic": ("BOOLEAN", {"default": False, "input": True}),
                "is_convert_to_openpose": ("BOOLEAN", {"default": False}),
                "thickness": ("INT", {"default": 1, "min": 1, "max": 10})
            }
        }

    CATEGORY = "Pose Helper Suite 🕺"

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("file_path", "num_frames", "fps", "width", "height", "frames")

    FUNCTION = "run"

    def run(self, file, is_reduce_holistic, is_convert_to_openpose, thickness):
        pose_file = folder_paths.get_annotated_filepath(file)

        if not os.path.exists(pose_file):
            raise ValueError(f"File {pose_file} does not exist")

        # Load Pose file
        with open(pose_file, "rb") as f:
            pose = Pose.read(f.read())

        if is_reduce_holistic:
            pose = reduce_holistic(pose)

        if is_convert_to_openpose:
            pose = convert_pose(pose, OpenPose_Components)
            pose.header.components[1].colors = [(255, 255, 255)]
            pose.header.components[0].colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
                                                [85, 255, 0], [0, 255, 0], \
                                                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                                                [0, 0, 255], [85, 0, 255], \
                                                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        print("thickness", thickness)
        print(pose.body.fps, pose.header.dimensions.width, pose.header.dimensions.height)

        visualizer = PoseVisualizer(pose, thickness=thickness)
        frames = visualizer.draw(background_color=(0, 0, 0))
        frames_rgb = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames)
        frames_torch = [torch.from_numpy(f).float() / 255 for f in frames_rgb]

        return (pose_file,
                len(pose.body.data),
                pose.body.fps,
                pose.header.dimensions.width,
                pose.header.dimensions.height,
                frames_torch)
