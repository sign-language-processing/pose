import itertools
import math
from functools import lru_cache
from typing import Tuple, Iterator

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

from .pose import Pose


class PoseVisualizer:
    def __init__(self, pose: Pose, thickness=None):
        self.pose = pose
        self.thickness = thickness
        self.pose_fps = float(self.pose.body.fps)

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError(
                "Please install OpenCV with: pip install opencv-python"
            )

    def _draw_frame(self, frame: ma.MaskedArray, frame_confidence: np.ndarray, img) -> np.ndarray:
        background_color = img[0][0]  # Estimation of background color for opacity. `mean` is slow

        thickness = self.thickness if self.thickness is not None else round(
            math.sqrt(img.shape[0] * img.shape[1]) / 150)
        radius = round(thickness / 2)

        for person, person_confidence in zip(frame, frame_confidence):
            c = person_confidence.tolist()
            idx = 0
            for component in self.pose.header.components:
                colors = [np.array(c[::-1]) for c in component.colors]

                @lru_cache(maxsize=None)
                def _point_color(p_i: int):
                    opacity = c[p_i + idx]
                    np_color = colors[p_i % len(component.colors)] * opacity + (1 - opacity) * background_color
                    return tuple([int(c) for c in np_color])

                # Draw Points
                for i, point_name in enumerate(component.points):
                    if c[i + idx] > 0:
                        self.cv2.circle(img=img, center=tuple(person[i + idx][:2]), radius=radius,
                                        color=_point_color(i), thickness=-1, lineType=16)

                if self.pose.header.is_bbox:
                    point1 = tuple(person[0 + idx].tolist())
                    point2 = tuple(person[1 + idx].tolist())
                    color = tuple(np.mean([_point_color(0), _point_color(1)], axis=0))

                    self.cv2.rectangle(img=img, pt1=point1, pt2=point2, color=color, thickness=thickness)
                else:
                    int_person = person.astype(np.int32)
                    # Draw Limbs
                    for (p1, p2) in component.limbs:
                        if c[p1 + idx] > 0 and c[p2 + idx] > 0:
                            point1 = tuple(int_person[p1 + idx].tolist()[:2])
                            point2 = tuple(int_person[p2 + idx].tolist()[:2])

                            # length = ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

                            color = tuple(np.mean([_point_color(p1), _point_color(p2)], axis=0))

                            self.cv2.line(img, point1, point2, color, thickness, lineType=self.cv2.LINE_AA)

                            # deg = math.degrees(math.atan2(point1[1] - point2[1], point1[0] - point2[0]))
                            # polygon = cv2.ellipse2Poly(
                            #   (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)),
                            #   (int(length / 2), thickness),
                            #   int(deg),
                            #   0, 360, 1)
                            # cv2.fillConvexPoly(img=img, points=polygon, color=color)

                idx += len(component.points)

        return img

    def draw(self, background_color: Tuple[int, int, int] = (255, 255, 255), max_frames: int = None):
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")
        background = np.full((self.pose.header.dimensions.height, self.pose.header.dimensions.width, 3),
                             fill_value=background_color, dtype="uint8")
        for frame, confidence in itertools.islice(zip(int_data, self.pose.body.confidence), max_frames):
            yield self._draw_frame(frame, confidence, img=background.copy())

    def draw_on_video(self, background_video, max_frames: int = None, blur=False):
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")

        if max_frames is None:
            max_frames = len(int_data)

        def get_frames(video_path):

            cap = self.cv2.VideoCapture(video_path)
            video_fps = cap.get(self.cv2.CAP_PROP_FPS)

            assert math.isclose(video_fps, self.pose_fps, abs_tol=0.5), \
                "Fps of pose and video do not match: %f != %f" % (self.pose_fps, video_fps)

            while True:
                ret, vf = cap.read()
                if not ret:
                    break
                yield vf
            cap.release()

        if isinstance(background_video, str):
            background_video = iter(get_frames(background_video))

        for frame, confidence, background in itertools.islice(
                zip(int_data, self.pose.body.confidence, background_video),
                max_frames):
            background = self.cv2.resize(background,
                                         (self.pose.header.dimensions.width, self.pose.header.dimensions.height))

            if blur:
                background = self.cv2.blur(background, (20, 20))

            yield self._draw_frame(frame, confidence, background)

    def save_frame(self, f_name: str, frame: np.ndarray):
        self.cv2.imwrite(f_name, frame)

    def save_video(self, f_name: str, frames: Iterator, custom_ffmpeg=None):
        try:
            from vidgear.gears import WriteGear
        except ImportError:
            raise ImportError(
                "Please install vidgear with: pip install vidgear"
            )

        # image_size = (self.pose.header.dimensions.width, self.pose.header.dimensions.height)

        output_params = {
            "-vcodec": "libx264",
            "-crf": 0,
            "-preset": "fast",
            "-input_framerate": self.pose.body.fps
        }

        # Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
        out = WriteGear(output_filename=f_name, logging=False, custom_ffmpeg=custom_ffmpeg, **output_params)
        # out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc(*'MP4V'), self.pose.body.fps, image_size)
        for frame in tqdm(frames):
            out.write(frame)

        out.close()
