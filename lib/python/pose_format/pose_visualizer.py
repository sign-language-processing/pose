import itertools
from typing import Tuple, Iterator

import cv2
import math
import numpy.ma as ma
import numpy as np
from tqdm import tqdm

from .pose import Pose


class PoseVisualizer:
    def __init__(self, pose: Pose):
        self.pose = pose

    def _draw_frame(self, frame: ma.MaskedArray, frame_confidence: np.ndarray,
                    img=np.ndarray) -> np.ndarray:

        for person, person_confidence in zip(frame, frame_confidence):
            c = person_confidence.tolist()
            idx = 0
            for component in self.pose.header.components:
                # Draw Points
                for i in range(len(component.points)):
                    if c[i + idx] > 0:
                        color = component.colors[i % len(component.colors)] * c[i + idx]
                        cv2.circle(img=img, center=tuple(person[i + idx]), radius=3, color=color, thickness=-1)

                # Draw Limbs
                # TODO
                # for (p1, p2) in limbs[key]:
                #     if p1 in joints and p2 in joints:
                #         point1 = (round(joints[p1]["x"]), round(joints[p1]["y"]))
                #         point2 = (round(joints[p2]["x"]), round(joints[p2]["y"]))
                #
                #         length = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
                #
                #         color = tuple(np.mean([point_color(key, p1), point_color(key, p2)], axis=0))
                #         if "c" in joints[p1]:
                #             color = color_opacity(color, (joints[p1]["c"] + joints[p2]["c"]) / 2)
                #
                #         deg = math.degrees(math.atan2(point1[1] - point2[1], point1[0] - point2[0]))
                #         polygon = cv2.ellipse2Poly((int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)),
                #                                    (int(length / 2), 3),
                #                                    int(deg),
                #                                    0, 360, 1)
                #         cv2.fillConvexPoly(image, polygon, color=color)

                idx += len(component.points)

        return img

    def draw(self, background: Tuple[int, int, int] = (255, 255, 255), max_frames: int = math.inf):
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")
        for frame, confidence in itertools.islice(zip(int_data, self.pose.body.confidence), max_frames):
            background = np.full((self.pose.header.dimensions.height, self.pose.header.dimensions.width, 3), background,
                                 dtype="uint8")
            yield self._draw_frame(frame, confidence, background)

    def draw_on_video(self, background_video: str, max_frames: int = None):
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")

        if max_frames is None:
            max_frames = len(int_data)

        cap = cv2.VideoCapture(background_video)
        for frame, confidence in itertools.islice(zip(int_data, self.pose.body.confidence), max_frames):
            _, background = cap.read()
            background = cv2.resize(background, (self.pose.header.dimensions.width, self.pose.header.dimensions.height))
            yield self._draw_frame(frame, confidence, background)
        cap.release()

    def save_frame(self, f_name: str, frame: np.ndarray):
        cv2.imwrite(f_name, frame)

    def save_video(self, f_name: str, frames: Iterator):
        print("out f", f_name)
        image_size = (self.pose.header.dimensions.width, self.pose.header.dimensions.height)
        out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc(*'MP4V'), self.pose.body.fps, image_size)
        for frame in tqdm(frames):
            out.write(frame)

        out.release()
