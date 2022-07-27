from typing import BinaryIO, List, Union

import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

from ..pose_body import PoseBody, POINTS_DIMS
from ..pose_header import PoseHeader
from ..utils.reader import BufferReader, ConstStructs


# import numpy as np
# np.seterr(all='raise')

class NumPyPoseBody(PoseBody):
    tensor_reader = 'unpack_numpy'

    def __init__(self, fps: float, data: Union[ma.MaskedArray, np.ndarray], confidence: np.ndarray):
        if isinstance(data, np.ndarray):  # If array is not masked
            mask = confidence == 0  # 0 means no-mask, 1 means with-mask
            stacked_mask = np.stack([mask] * data.shape[-1], axis=3)
            data = ma.masked_array(data, mask=stacked_mask)

        super().__init__(fps, data, confidence)

    @classmethod
    def read_v0_0(cls, header: PoseHeader, reader: BufferReader):
        fps, _frames = reader.unpack(ConstStructs.double_ushort)

        _dims = max([len(c.format) for c in header.components]) - 1
        _points = sum([len(c.points) for c in header.components])

        frames_d = []
        frames_c = []
        for _ in range(_frames):
            _people = reader.unpack(ConstStructs.ushort)
            people_d = []
            people_c = []
            for pid in range(_people):
                reader.advance(ConstStructs.short)  # Skip Person ID
                person_d = []
                person_c = []
                for component in header.components:
                    points = np.array(
                        reader.unpack_numpy(ConstStructs.float, (len(component.points), len(component.format))))
                    dimensions, confidence = np.split(points, [-1], axis=1)
                    boolean_confidence = np.where(confidence > 0, 0, 1)  # To create the mask
                    mask = np.column_stack(tuple([boolean_confidence] * (len(component.format) - 1)))

                    person_d.append(ma.masked_array(dimensions, mask=mask))
                    person_c.append(np.squeeze(confidence, axis=-1))

                if pid == 0:
                    people_d.append(ma.concatenate(person_d))
                    people_c.append(np.concatenate(person_c))

            # In case no person, should all be zeros
            if len(people_d) == 0:
                people_d.append(np.zeros((_points, _dims)))
                people_c.append(np.zeros(_points))

            frames_d.append(ma.stack(people_d))
            frames_c.append(np.stack(people_c))

        return cls(fps, ma.stack(frames_d), ma.stack(frames_c))

    def write(self, version: float, buffer: BinaryIO):
        _frames, _people, _points, _dims = self.data.shape
        _frames = _frames if _frames < 65535 else 0  # TODO change from short to int
        buffer.write(ConstStructs.triple_ushort.pack(self.fps, _frames, _people))

        buffer.write(np.array(self.data.data, dtype=np.float32).tobytes())
        buffer.write(np.array(self.confidence, dtype=np.float32).tobytes())

    @property
    def mask(self):
        return self.data.mask

    def torch(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Please install torch. https://pytorch.org/"
            )

        import torch
        from ..torch.pose_body import TorchPoseBody

        torch_confidence = torch.from_numpy(self.confidence)
        torch_data = torch.from_numpy(self.data.data)
        return TorchPoseBody(self.fps, torch_data, torch_confidence)

    def tensorflow(self):
        import tensorflow
        from ..tensorflow.pose_body import TensorflowPoseBody

        tf_confidence = tensorflow.constant(self.confidence)
        tf_data = tensorflow.constant(self.data.data)
        return TensorflowPoseBody(self.fps, tf_data, tf_confidence)

    def zero_filled(self):
        self.data = ma.array(self.data.filled(0), mask=self.data.mask)
        return self

    def matmul(self, matrix: np.ndarray):
        data = ma.dot(self.data, matrix)
        return NumPyPoseBody(self.fps, data, self.confidence)

    def flip(self, axis=0):
        vec = np.ones(self.data.shape[-1])
        vec[axis] = -1

        data = self.data * vec
        return NumPyPoseBody(self.fps, data, self.confidence)

    def points_perspective(self):
        return ma.transpose(self.data, axes=POINTS_DIMS)

    def get_points(self, indexes: List[int]):
        data = ma.transpose(self.data, axes=POINTS_DIMS)
        new_data = ma.transpose(data[indexes], axes=POINTS_DIMS)

        confidence_reshape = (2, 1, 0)
        confidence = np.transpose(self.confidence, axes=confidence_reshape)
        new_confidence = np.transpose(confidence[indexes], axes=confidence_reshape)

        return NumPyPoseBody(self.fps, new_data, new_confidence)

    def bbox(self, header: PoseHeader):
        data = ma.transpose(self.data, axes=POINTS_DIMS)

        # Split data by components, `ma` doesn't support ".split"
        components = []
        idx = 0
        for component in header.components:
            components.append(data[list(range(idx, idx + len(component.points)))])
            idx += len(component.points)

        boxes = [ma.stack([ma.min(c, axis=0), ma.max(c, axis=0)]) for c in components]
        boxes_cat = ma.concatenate(boxes)
        if type(boxes_cat.mask) == np.bool_:  # Sometimes, it doesn't concatenate the mask...
            boxes_mask = ma.concatenate([b.mask for b in boxes])
            boxes_cat = ma.array(boxes_cat, mask=boxes_mask)

        new_data = ma.transpose(boxes_cat, axes=POINTS_DIMS)

        confidence_mask = np.split(new_data.mask, [-1], axis=3)[0]
        confidence_mask = np.squeeze(confidence_mask, axis=-1)
        confidence = np.where(confidence_mask == True, 0, 1)

        return NumPyPoseBody(self.fps, new_data, confidence)

    def interpolate(self, new_fps: int = None, kind='cubic'):
        if new_fps is None:
            new_fps = self.fps

        _frames = self.data.shape[0]
        if _frames == 1:
            raise ValueError("Can't interpolate single frame")

        _new_frames = round(_frames * new_fps / self.fps)
        steps = np.linspace(0, 1, _frames)
        new_steps = np.linspace(0, 1, _new_frames)

        transposed = self.points_perspective()  # (points, people, frames, dims)
        masked_confidence = ma.array(self.confidence, mask=self.confidence == 0)
        confidence = ma.expand_dims(masked_confidence.transpose(), axis=3)  # (points, people, frames, 1)
        points = ma.concatenate([transposed, confidence], axis=3)

        new_people = []
        for people in points:
            new_frames = []
            for frames in people:
                mask = frames.transpose()[0].mask

                partial_steps = ma.array(steps, mask=mask).compressed()

                if partial_steps.shape[0] == 0:  # No data for this point
                    new_frames.append(np.zeros((_new_frames, frames.shape[1])))
                else:
                    partial_frames = frames.compressed().reshape(partial_steps.shape[0], frames.shape[1])

                    if len(partial_steps) == 1:
                        f = lambda l: partial_frames
                    else:
                        this_kind = kind if len(partial_steps) > 3 \
                            else "quadratic" if len(partial_steps) > 2 and kind == "cubic" \
                            else "linear"  # Can't do something fancy for 2 points
                        f = interp1d(partial_steps, partial_frames, axis=0, kind=this_kind)

                    first_step = partial_steps[0]
                    last_step = partial_steps[-1]
                    if first_step == 0 and last_step == 1:
                        new_frames.append(f(new_steps))
                    else:
                        first_step_where = np.argwhere(new_steps >= first_step)
                        first_step_index = first_step_where[0][0] if len(first_step_where) > 0 else 0

                        last_step_where = np.argwhere(new_steps > last_step)
                        last_step_index = last_step_where[0][0] if len(last_step_where) > 0 else len(new_steps)

                        if first_step_index == last_step_index:
                            new_frames.append(np.zeros((len(new_steps), frames.shape[1])))
                        else:
                            frame_data = f(new_steps[first_step_index:last_step_index])
                            new_frames.append(np.concatenate([
                                np.zeros((first_step_index, frames.shape[1])),
                                np.array(frame_data),
                                np.zeros((len(new_steps) - last_step_index, frames.shape[1]))
                            ]))
            new_people.append(np.stack(new_frames, axis=0))

        new_data = np.stack(new_people, axis=0).transpose([2, 1, 0, 3])
        dimensions, confidence = np.split(new_data, [-1], axis=3)
        confidence = np.squeeze(confidence, axis=3)

        return NumPyPoseBody(fps=new_fps, data=dimensions, confidence=confidence)

    def flatten(self):
        shape = self.data.shape
        data = self.data.data.reshape(-1, shape[-1])  # Not masked data
        confidence = self.confidence.flatten()
        indexes = list(np.ndindex(shape[:-1]))
        flat = np.c_[indexes, confidence, data]
        # Filter data from flat
        flat = flat[confidence != 0]
        # Scale the first axis by fps
        scalar = np.ones(len(shape) + shape[-1])
        scalar[0] = 1 / self.fps
        return flat * scalar
