from typing import Tuple

import cv2
import numpy.ma as ma
import numpy as np

from .pose import Pose


class PoseVisualizer:
    def __init__(self, pose: Pose):
        self.pose = pose

    def _draw_frame(self, frame: ma.MaskedArray, frame_confidence: np.ndarray,
                    background=(255, 255, 255)) -> np.ndarray:
        img = np.full((self.pose.header.dimensions.width, self.pose.header.dimensions.height, 3), background, dtype="uint8")

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

    def draw(self, f_name: str, background: Tuple[int, int, int] = (255, 255, 255)):
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")
        video = []
        for frame, confidence in zip(int_data, self.pose.body.confidence):
            video.append(self._draw_frame(frame, confidence, background))

        cv2.imwrite(f_name + ".png", video[0])
