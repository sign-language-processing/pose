from random import sample

import math
from imgaug import Keypoint, KeypointsOnImage
from imgaug.augmenters import Augmenter

from .vectorizer import Vectorizer
from .pose_body import PoseBody
from .pose_header import PoseHeader, PoseHeaderDimensions, PoseNormalizationInfo
from .utils.reader import BufferReader

import numpy as np
import numpy.ma as ma

from .utils.fast_math import distance_batch


class Pose:
    def __init__(self, header: PoseHeader, body: PoseBody):
        self.header = header
        self.body = body

    @staticmethod
    def read(buffer: bytes):
        reader = BufferReader(buffer)
        header = PoseHeader.read(reader)
        body = PoseBody.read(header, reader)

        return Pose(header, body)

    def write(self, f_name: str):
        f = open(f_name, "wb")
        self.header.write(f)
        self.body.write(f)
        f.flush()
        f.close()

    def focus(self):
        """
        Gets the pose to start at (0,0) and have dimensions as big as needed
        """

        mins = ma.min(self.body.data, axis=(0, 1, 2))
        maxs = ma.max(self.body.data, axis=(0, 1, 2))

        if np.count_nonzero(mins) > 0:  # Only translate if there is a number to translate by
            self.body.data = ma.subtract(self.body.data, mins)

        dimensions = (maxs - mins).tolist()
        self.header.dimensions = PoseHeaderDimensions(*dimensions)

    def normalize(self, info: PoseNormalizationInfo, scale_factor: float = 1):
        """
        Normalize the point to a fixed distance between two points
        """
        transposed = self.body.points_perspective()

        p1s = transposed[info.p1]
        p2s = transposed[info.p2]

        if transposed.shape[1] == 0:
            p1s = p1s[0]
            p2s = p2s[0]
        else:
            p1s = ma.concatenate(p1s)
            p2s = ma.concatenate(p2s)

        mean_distance = np.mean(distance_batch(p1s, p2s))

        scale = scale_factor / mean_distance  # scale all points to dist/scale

        if round(scale, 5) != 1:
            self.body.data = ma.multiply(self.body.data, scale)

    def squeeze(self, by: int):
        """
        Fast way of data interpolation
        :param by: take one row every "by" rows
        :return: Pose
        """
        new_data = self.body.data[::by]
        new_confidence = self.body.confidence[::by]

        body = PoseBody(fps=self.body.fps, data=new_data, confidence=new_confidence)
        return Pose(header=self.header, body=body)

    def interpolate(self, new_fps: int, kind='cubic'):
        body = self.body.interpolate(new_fps=new_fps, kind=kind)
        return Pose(header=self.header, body=body)

    def to_vectors(self, vectorizer: Vectorizer) -> np.ndarray:
        transposed = self.body.points_perspective()

        pt1s = []
        pt2s = []

        idx = 0
        for component in self.header.components:
            for (a, b) in component.limbs:
                pt1s.append(a + idx)
                pt2s.append(b + idx)
            idx += len(component.points)

        people_vectors = vectorizer(transposed[pt1s], transposed[pt2s])
        if people_vectors.shape[1] == 1:
            vectors = np.squeeze(people_vectors)
        else:
            vectors = np.concatenate(people_vectors, axis=0)

        return np.transpose(vectors)

    def augment_vectors(self, vectors: np.ndarray, std=0.1, dropout_std=0.1) -> np.ndarray:
        if dropout_std > 0:
            dropout_percent = np.abs(np.random.normal(loc=0, scale=dropout_std, size=1))[0]
            dropout_indexes = sample(range(0, vectors.shape[0]), int(vectors.shape[0] * dropout_percent))
            vectors = np.delete(vectors, dropout_indexes, axis=0)

        multiplier = np.random.normal(loc=0, scale=std, size=vectors.shape[1]) + 1
        return np.multiply(vectors, multiplier)

    def augment2d(self, rotation_std=0.2, shear_std=0.2, scale_std=0.2):
        """
        :param rotation_std: Rotation in radians
        :param shear_std: Shear X in percent
        :param scale_std: Scale X in percent
        :return:
        """
        matrix = np.eye(2)

        # Based on https://en.wikipedia.org/wiki/Shear_matrix
        if shear_std > 0:
            shear_matrix = np.eye(2)
            shear_matrix[0][1] = np.random.normal(loc=0, scale=shear_std, size=1)[0]
            matrix = np.dot(matrix, shear_matrix)

        # Based on https://en.wikipedia.org/wiki/Rotation_matrix
        if rotation_std > 0:
            rotation_angle = np.random.normal(loc=0, scale=rotation_std, size=1)[0]
            rotation_cos = np.cos(rotation_angle)
            rotation_sin = np.sin(rotation_angle)
            rotation_matrix = np.array([[rotation_cos, -rotation_sin], [rotation_sin, rotation_cos]])
            matrix = np.dot(matrix, rotation_matrix)

        # Based on https://en.wikipedia.org/wiki/Scaling_(geometry)
        if scale_std > 0:
            scale_matrix = np.eye(2)
            scale_matrix[0][0] += np.random.normal(loc=0, scale=scale_std, size=1)[0]
            matrix = np.dot(matrix, scale_matrix)

        new_data = ma.dot(self.body.data, matrix.astype(dtype=np.float32))

        body = PoseBody(fps=self.body.fps, data=new_data, confidence=self.body.confidence)
        return Pose(header=self.header, body=body)

    def augment2d_imgaug(self, augmenter: Augmenter):
        _frames, _people, _points, _dims = self.body.data.shape
        np_keypoints = self.body.data.data.reshape((_frames * _people * _points, _dims))

        keypoints = [Keypoint(x=x, y=y) for x, y in np_keypoints]

        kps = KeypointsOnImage(keypoints, shape=(self.header.dimensions.height, self.header.dimensions.width))
        kps_aug = augmenter(keypoints=kps)  # Augment keypoints

        np_keypoints = np.array([[k.x, k.y] for k in kps_aug.keypoints]).reshape((_frames, _people, _points, _dims))

        new_data = ma.array(data=np_keypoints, mask=self.body.data.mask)

        body = PoseBody(fps=self.body.fps, data=new_data, confidence=self.body.confidence)
        return Pose(header=self.header, body=body)
