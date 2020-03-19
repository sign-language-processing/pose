from typing import BinaryIO

from .pose_header import PoseHeader
from .utils.reader import BufferReader, ConstStructs
import numpy as np
import numpy.ma as ma


class PoseBody:
    def __init__(self, fps: int, data: ma.MaskedArray, confidence: np.ndarray):
        self.fps = fps
        self.data = data  # Shape (frames, people, points, dims) - eg (93, 1, 137, 2)
        self.confidence = confidence  # Shape (frames, people, points) - eg (93, 1, 137)

    @staticmethod
    def read(header: PoseHeader, reader: BufferReader):
        fps, _frames = reader.unpack(ConstStructs.double_ushort)

        _dims = max([len(c.format) for c in header.components]) - 1
        _points = sum([len(c.points) for c in header.components])

        if header.version == 0:
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
                        # TODO make sure v0 "people" match person_shape(0)
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

            return PoseBody(fps, ma.stack(frames_d), ma.stack(frames_c))

        elif round(header.version, 3) == 0.1:
            _people = reader.unpack(ConstStructs.ushort)
            _points = sum([len(c.points) for c in header.components])
            _dims = max([len(c.format) for c in header.components]) - 1

            data = reader.unpack_numpy(ConstStructs.float, (_frames, _people, _points, _dims))
            confidence = reader.unpack_numpy(ConstStructs.float, (_frames, _people, _points))

            b_confidence = np.where(confidence > 0, 0, 1) # 0 means no-mask, 1 means with-mask
            stacked_confidence = np.stack([b_confidence, b_confidence], axis=3)
            masked_data = ma.masked_array(data, mask=stacked_confidence)

            return PoseBody(fps, masked_data, confidence)

        raise NotImplementedError("Unknown version - " + str(header.version))

    def write(self, buffer: BinaryIO):
        _frames, _people, _points, _dims = self.data.shape
        buffer.write(ConstStructs.triple_ushort.pack(self.fps, _frames, _people))

        buffer.write(self.data.data.tobytes())
        buffer.write(self.confidence.tobytes())

    def points_perspective(self):
        return ma.transpose(self.data, axes=(2, 1, 0, 3))
