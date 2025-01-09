import itertools
import logging
import math
from functools import lru_cache
from io import BytesIO
from typing import Iterable, Tuple, Union

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

from .pose import Pose


class PoseVisualizer:
    """
    A class for visualizing Pose objects using OpenCV.

    Parameters
    ----------
    pose : Pose
        The Pose object to visualize.
    thickness : int or None
        Thickness for drawing. If not provided, it is estimated based on image size.
    pose_fps : float
        Frame rate of the Pose data.
    *cv2 : module
        OpenCV Python binding.
    """

    def __init__(self, pose: Pose, thickness=None):
        """Initialize the PoseVisualizer class."""
        self.pose = pose
        self.thickness = thickness
        self.pose_fps = float(self.pose.body.fps)

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("Please install OpenCV with: pip install opencv-python")

    def _draw_frame(self, frame: ma.MaskedArray,
                    frame_confidence: np.ndarray, img,
                    transparency: bool = False) -> np.ndarray:
        """
        Draw frame of pose data of an image.

        Parameters
        ----------
        frame : ma.MaskedArray
            2D array containing the pose data for a frame.
        frame_confidence : np.ndarray
            Confidence values for each point in the frame.
        img : np.ndarray
            Background image where upon pose will be drawn.
        transparency : bool
            transparency decides opacity of background color,

        Returns
        -------
        np.ndarray
            Image with drawn pose data.
        """

        background_color = img[0][0]  # Estimation of background color for opacity. `mean` is slow

        # Estimation of thickness and radius for drawing
        thickness = self.thickness
        if self.thickness is None:
            thickness = round(math.sqrt(img.shape[0] * img.shape[1]) / 150)
        radius = math.ceil(thickness / 2)

        draw_operations = []

        for person, person_confidence in zip(frame, frame_confidence):
            c = person_confidence.tolist()
            idx = 0
            for component in self.pose.header.components:
                colors = [np.array(c[::-1]) for c in component.colors]

                @lru_cache(maxsize=None)
                def _point_color(p_i: int):
                    opacity = c[p_i + idx]
                    np_color = colors[p_i % len(component.colors)] * opacity + (1 - opacity) * background_color[
                                                                                               :3]  # [:3] ignores alpha value if present
                    if transparency:
                        np_color = np.append(np_color, opacity * 255)
                    return tuple([int(c) for c in np_color])

                # Collect Points
                for i, point_name in enumerate(component.points):
                    if c[i + idx] > 0:
                        center = person[i + idx]
                        draw_operations.append({
                            'type': 'circle',
                            'center': center,
                            'radius': radius,
                            'color': _point_color(i),
                            'thickness': -1,
                            'lineType': 16,
                            'z': center[2] if len(center) > 2 else 0
                        })

                if self.pose.header.is_bbox:
                    point1 = person[0 + idx]
                    point2 = person[1 + idx]
                    color = tuple(np.mean([_point_color(0), _point_color(1)], axis=0))

                    draw_operations.append({
                        'type': 'rectangle',
                        'pt1': point1,
                        'pt2': point2,
                        'color': color,
                        'thickness': thickness,
                        'z': (point1[2] + point2[2]) / 2 if len(point1) > 2 else 0
                    })
                else:
                    # Collect Limbs
                    for (p1, p2) in component.limbs:
                        if c[p1 + idx] > 0 and c[p2 + idx] > 0:
                            point1 = person[p1 + idx]
                            point2 = person[p2 + idx]

                            color = tuple(np.mean([_point_color(p1), _point_color(p2)], axis=0))

                            draw_operations.append({
                                'type': 'line',
                                'pt1': point1,
                                'pt2': point2,
                                'color': color,
                                'thickness': thickness,
                                'lineType': self.cv2.LINE_AA,
                                'z': (point1[2] + point2[2]) / 2 if len(point1) > 2 else 0
                            })

                idx += len(component.points)

        draw_operations = sorted(draw_operations, key=lambda op: op['z'], reverse=True)

        def point_to_xy(point: ma.MaskedArray):
            return tuple([round(p) for p in point[:2]])

        # Execute draw operations
        for op in draw_operations:
            if op['type'] == 'circle':
                self.cv2.circle(img=img,
                                center=point_to_xy(op['center']),
                                radius=op['radius'],
                                color=op['color'],
                                thickness=op['thickness'],
                                lineType=op['lineType'])
            elif op['type'] == 'rectangle':
                self.cv2.rectangle(img=img,
                                   pt1=point_to_xy(op['pt1']),
                                   pt2=point_to_xy(op['pt2']),
                                   color=op['color'],
                                   thickness=op['thickness'])
            elif op['type'] == 'line':
                self.cv2.line(img,
                              pt1=point_to_xy(op['pt1']),
                              pt2=point_to_xy(op['pt2']),
                              color=op['color'],
                              thickness=op['thickness'],
                              lineType=op['lineType'])

        return img

    def draw(self, background_color: Tuple[int, int, int] = (255, 255, 255), max_frames: int = None,
             transparency: bool = False):
        """
        draws pose on plain background using the specified color - for a number of frames.

        Parameters
        ----------
        background_color : Tuple[int, int, int], optional
            RGB value for background color, default is white (255, 255, 255).
        max_frames : int, optional
            Maximum number of frames to process, if it is None, it processes all frames.
        transparency : bool
            transparency decides opacity of background color, it is only used in the case of PNG i.e It doesn't affect GIF.
        Yields
        ------
        np.ndarray
            Frames with the pose data drawn on a custom background color.
        """
        # ...
        if transparency:
            background_color += (0,)
        background = np.full(
            (self.pose.header.dimensions.height, self.pose.header.dimensions.width, len(background_color)),
            fill_value=background_color,
            dtype="uint8")
        for frame, confidence in itertools.islice(zip(self.pose.body.data, self.pose.body.confidence), max_frames):
            yield self._draw_frame(frame, confidence, img=background.copy(), transparency=transparency)

    def draw_on_video(self, background_video, max_frames: int = None, blur=False):
        """
        Draw pose on a background video.

        Parameters
        ----------
        background_video : str or iterable
            Path to video file or iterable of video frames.
        max_frames : int, optional
            Maximum number of frames to process. If None, it will be processing all frames.
        blur : bool, optional
            If True, applies a blur effect to the video.

        Yields
        ------
        np.ndarray
            Frames with overlaid pose data.
        """
        int_data = np.array(np.around(self.pose.body.data.data), dtype="int32")

        if max_frames is None:
            max_frames = len(int_data)

        def get_frames(video_path):

            cap = self.cv2.VideoCapture(video_path)
            video_fps = cap.get(self.cv2.CAP_PROP_FPS)

            assert math.isclose(video_fps, self.pose_fps, abs_tol=0.1), \
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
                zip(int_data, self.pose.body.confidence, background_video), max_frames):
            background = self.cv2.resize(background,
                                         (self.pose.header.dimensions.width, self.pose.header.dimensions.height))

            if blur:
                background = self.cv2.blur(background, (20, 20))

            yield self._draw_frame(frame, confidence, background)

    def save_frame(self, f_name: str, frame: np.ndarray):
        """
        Save a single pose frame as im.

        Parameters
        ----------
        f_name : str
            filensmr where the frame will be saved.
        frame : np.ndarray
            Pose frame to be saved

        Returns
        -------
        None
        """
        self.cv2.imwrite(f_name, frame)

    def _save_image(self, f_name: Union[str, None], frames: Iterable[np.ndarray], format: str = "GIF",
                    transparency: bool = False) -> Union[None, bytes]:
        """
        Save pose frames as Image (GIF or PNG).

        Parameters
        ----------
        f_name : Union[str, None]
        	Filename to save Image to. If None, image will be saved to memory and returned as bytes.
        frames : Iterable[np.ndarray]
            Series of pose frames to be included in Image.
        format : str
            format to save takes either GIF or PNG.
        transparency : bool
            transparency decides opacity of background color.

        Returns
        -------
        Union[None, bytes]
        	If f_name is None, returns the image data as bytes. Otherwise, returns None.

        Raises
        ------
        ImportError 
            If Pillow is not installed.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Please install Pillow with: pip install Pillow")

        if transparency:
            cv_code = self.cv2.COLOR_BGR2RGBA
        else:
            cv_code = self.cv2.COLOR_BGR2RGB

        images = [Image.fromarray(self.cv2.cvtColor(frame, cv_code)) for frame in frames]

        def save_to(obj: Union[str, None]):
            images[0].save(obj,
                           format=format,
                           append_images=images[1:],
                           save_all=True,
                           duration=1000 / self.pose.body.fps,
                           loop=0,
                           disposal=2 if transparency else 0)

        if f_name:
            save_to(f_name)
        else:
            with BytesIO() as mem:
                save_to(mem)
                return mem.getvalue()

    def save_gif(self, f_name: Union[str, None], frames: Iterable[np.ndarray]) -> Union[None, bytes]:
        """
        Save pose frames as GIF.

        Parameters
        ----------
        f_name : Union[str, None]
       		Filename to save PNG to. If None, image will be saved to memory and returned as bytes.
        frames : Iterable[np.ndarray]
            Series of pose frames to be included in GIF.

        Returns
        -------
        Union[None, bytes]
        	If f_name is None, returns the PNG image data as bytes. Otherwise, returns None.

        Raises
        ------
        ImportError 
            If Pillow is not installed.
        """
        return self._save_image(f_name, frames, "GIF", False)

    def save_png(self, f_name: Union[str, None], frames: Iterable[np.ndarray],
                 transparency: bool = True) -> Union[None, bytes]:
        """
        Save pose frames as PNG.

        Parameters
        ----------
        f_name : Union[str, None]
        	Filename to save PNG to. If None, image will be saved to memory and returned as bytes.
        frames : Iterable[np.ndarray]
            Series of pose frames to be included in PNG.
        transparency : bool
            transparency decides opacity of background color.

        Returns
        -------
        Union[None, bytes]
        	If f_name is None, returns the PNG image data as bytes. Otherwise, returns None.

        Raises
        ------
        ImportError 
            If Pillow is not installed.
        """
        return self._save_image(f_name, frames, "PNG", transparency)

    def save_video(self, f_name: str, frames: Iterable[np.ndarray], custom_ffmpeg=None):
        """
        Save pose frames as a video.

        Parameters
        ----------
        f_name : str
            Filename to which the generated video is saved to .
        frames : Iterable[np.ndarray]
            Iterable of pose frames include in the video.
        custom_ffmpeg : optional
            Custom ffmpeg parameters for the "video writing".

        Returns
        -------
        None

        Raises
        ------
        ImportError 
            If vidgear is not installed.
        """
        try:
            from vidgear.gears import WriteGear
        except ImportError:
            raise ImportError("Please install vidgear with: pip install vidgear")

        # image_size = (self.pose.header.dimensions.width, self.pose.header.dimensions.height)

        output_params = {
            "-vcodec": "libx264",
            "-preset": "fast",
            "-input_framerate": self.pose.body.fps,
        }

        writer = None  # Define writer with defined parameters and suitable output filename for e.g. `Output.mp4`
        for frame in tqdm(frames):
            if writer is None:  # Create writer on first frame
                if frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0:
                    output_params["-pix_fmt"] = "yuv420p"  # H.264
                else:
                    logging.warning(
                        "Video shape is not divisible by 2. Can not use H.264. Consider resizing to a divisible shape.")
                writer = WriteGear(output=f_name, logging=False, custom_ffmpeg=custom_ffmpeg, **output_params)
            writer.write(frame)

        writer.close()


class FastAndUglyPoseVisualizer(PoseVisualizer):
    """
    This class draws all frames as grayscale, without opacity based on confidence values.
    It is a faster and less detailed "ugly" class for visualizing Pose objects using OpenCV.
    
    * Inherites from `PoseViszaizer`
    """

    def _draw_frame(self, frame: ma.MaskedArray, img, color: int):
        """
        Draw a frame of pose on an image using a one color.

        Parameters
        ----------
        frame : ma.MaskedArray
            2D array containing the pose data for a single frame.
        img : np.ndarray
            The background image on which the pose is to be drawn.
        color : int
            Grayscale color value to use for drawing the pose.

        Returns
        -------
        np.ndarray
            Image with drawn pose data.
        """
        ignored_point = (0, 0)
        # Note: this can be made faster by drawing polylines instead of lines
        thickness = 1
        for person in frame:
            points_2d = [tuple(p) for p in person[:, :2].tolist()]
            idx = 0
            for component in self.pose.header.components:
                for (p1, p2) in component.limbs:
                    point1 = points_2d[p1 + idx]
                    point2 = points_2d[p2 + idx]
                    if point1 != ignored_point and point2 != ignored_point:
                        # Antialiasing is a bit slow, but necessary
                        self.cv2.line(img, point1, point2, color, thickness, lineType=self.cv2.LINE_AA)

                idx += len(component.points)
        return img

    def draw(self, background_color: int = 0, foreground_color: int = 255):
        """
        draws the pose on plain background using a foreground (pose) color.

        Parameters
        ----------
        background_color : int
            Grayscale value for background color.
        foreground_color : int
            Grayscale value for the pose color.

        Yields
        ------
        np.ndarray
            frames with drawn pose
        """
        background = np.full((self.pose.header.dimensions.height, self.pose.header.dimensions.width),
                             fill_value=background_color,
                             dtype="uint8")
        for frame in self.pose.body.data:
            yield self._draw_frame(frame, img=background.copy(), color=foreground_color)
