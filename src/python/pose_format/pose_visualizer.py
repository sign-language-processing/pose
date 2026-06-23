import itertools
import logging
import math
from io import BytesIO
from typing import Iterable, Tuple, Union

import numpy as np
import numpy.ma as ma
from tqdm import tqdm
from simple_video_utils.metadata import video_metadata
from simple_video_utils.frames import read_frames_exact

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

        background_color = np.asarray(img[0][0][:3], dtype=float)  # background color for opacity; `mean` is slow

        thickness = self.thickness
        if self.thickness is None:
            thickness = round(math.sqrt(img.shape[0] * img.shape[1]) / 150)
        radius = math.ceil(thickness / 2)

        # MaskedArray element indexing is very slow, so resolve coordinates to plain ints once per frame
        points = np.asarray(frame)
        xy = np.round(points[..., :2]).astype(int)
        has_z = points.shape[-1] > 2
        z = points[..., 2] if has_z else None

        ops = []  # (z, kind, pt1, pt2, color); kind 0=circle, 1=line, 2=rectangle

        for p, person_confidence in enumerate(frame_confidence):
            conf = np.asarray(person_confidence)
            idx = 0
            for component in self.pose.header.components:
                n = len(component.points)
                comp_conf = conf[idx:idx + n]

                palette = np.array([c[::-1] for c in component.colors], dtype=float)  # RGB -> BGR
                opacity = palette[np.arange(n) % len(palette)] * comp_conf[:, None]
                comp_colors = opacity + (1 - comp_conf[:, None]) * background_color
                colors = [tuple(int(v) for v in col) for col in comp_colors]
                if transparency:
                    colors = [col + (int(o * 255),) for col, o in zip(colors, comp_conf)]

                comp_xy = [tuple(pt) for pt in xy[p, idx:idx + n].tolist()]
                comp_z = z[p, idx:idx + n] if has_z else [0] * n

                if self.pose.header.is_bbox:
                    color = tuple((a + b) / 2 for a, b in zip(colors[0], colors[1]))
                    ops.append(((comp_z[0] + comp_z[1]) / 2, 2, comp_xy[0], comp_xy[1], color))
                else:
                    for i in range(n):
                        if comp_conf[i] > 0:
                            ops.append((comp_z[i], 0, comp_xy[i], None, colors[i]))
                    for (p1, p2) in component.limbs:
                        if comp_conf[p1] > 0 and comp_conf[p2] > 0:
                            color = tuple((a + b) / 2 for a, b in zip(colors[p1], colors[p2]))
                            ops.append(((comp_z[p1] + comp_z[p2]) / 2, 1, comp_xy[p1], comp_xy[p2], color))

                idx += n

        # Painter's algorithm: draw far operations first (larger z = further away)
        ops.sort(key=lambda op: op[0], reverse=True)
        for _, kind, pt1, pt2, color in ops:
            if kind == 0:
                self.cv2.circle(img, pt1, radius, color, thickness=-1, lineType=16)
            elif kind == 1:
                self.cv2.line(img, pt1, pt2, color, thickness=thickness, lineType=self.cv2.LINE_AA)
            else:
                self.cv2.rectangle(img, pt1, pt2, color, thickness=thickness)

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
            # Get video metadata
            metadata = video_metadata(video_path)
            video_fps = metadata.fps

            assert math.isclose(video_fps, self.pose_fps, abs_tol=0.1), \
                "Fps of pose and video do not match: %f != %f" % (self.pose_fps, video_fps)

            # Read frames and convert RGB to BGR (cv2 expects BGR)
            for frame in read_frames_exact(video_path):
                yield self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)

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
