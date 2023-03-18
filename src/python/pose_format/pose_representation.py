from typing import List

from .pose_header import PoseHeader


class PoseRepresentation:
    def __init__(self, header: PoseHeader, rep_modules1: List = [], rep_modules2: List = [], rep_modules3: List = []):
        self.header = header

        self.input_size = sum([len(c.points) for c in header.components])
        dims = len(header.components[0].format)

        # Modules relying on points
        self.rep_modules1 = rep_modules1
        self.rep_modules1_size = self.input_size * dims

        # Modules relying on limbs
        self.rep_modules2 = rep_modules2
        self.limb_pt1s, self.limb_pt2s = self.get_limbs_points()
        self.rep_modules2_size = len(self.limb_pt1s)

        # Modules relying on triangles
        self.rep_modules3 = rep_modules3
        self.triangle_pt1s, self.triangle_pt2s, self.triangle_pt3s = self.get_triangles_points()
        self.rep_modules3_size = len(self.triangle_pt1s)

        self.output_size = self.calc_output_size()

    def calc_output_size(self):
        return len(self.rep_modules1) * self.rep_modules1_size + \
               len(self.rep_modules2) * self.rep_modules2_size + \
               len(self.rep_modules3) * self.rep_modules3_size

    def get_limbs_points(self):
        pt1s = []
        pt2s = []

        idx = 0
        for component in self.header.components:
            for (a, b) in component.limbs:
                pt1s.append(a + idx)
                pt2s.append(b + idx)
            idx += len(component.points)

        return pt1s, pt2s

    def get_triangles_points(self):
        assert self.limb_pt1s
        assert self.limb_pt2s

        # Limb continuing when limb ended
        chains = [(p1, p2, p4) for p1, p2 in zip(self.limb_pt1s, self.limb_pt2s)
                  for p3, p4 in zip(self.limb_pt1s, self.limb_pt2s) if p2 == p3]
        # # Limbs coming out from the same location
        # branches = [(p2, p1, p4) for p1, p2 in zip(self.limb_pt1s, self.limb_pt2s)
        #             for p3, p4 in zip(self.limb_pt1s, self.limb_pt2s) if p1 == p3 and p2 != p4]
        branches = []

        triangles = chains + branches
        return list(zip(*triangles))

    def group_embeds(self, embeds: List):
        """
        :param embeds: List of tensors size (embed_size, Batch, Len)
        :return: Size (Batch, Len, embed_size)
        """
        raise NotImplementedError('Group embeds is not implemented')

    def get_points(self, tensor, points):
        return tensor[points]

    def permute(self, src, shape: tuple):
        raise NotImplementedError('Group embeds is not implemented')

    def __call__(self, src):
        """
        :param src: Size (Batch, Len, Points, Dims)
        :return: Size (Batch, Len, embed_size)
        """
        points = self.permute(src, (2, 0, 1, 3))  # (Points, Batch, Len, Dims)

        embeds = []  # (embed_size, Batch, Len)

        # Use modules requiring a single point
        if len(self.rep_modules1) > 0:
            embeds += [module(points) for module in self.rep_modules1]

        # Use modules requiring limbs
        if len(self.rep_modules2) > 0:
            pt1s = self.get_points(points, self.limb_pt1s)
            pt2s = self.get_points(points, self.limb_pt2s)
            embeds += [module(p1s=pt1s, p2s=pt2s) for module in self.rep_modules2]

        # Use modules requiring triangles
        if len(self.rep_modules3) > 0:
            pt1s = self.get_points(points, self.triangle_pt1s)
            pt2s = self.get_points(points, self.triangle_pt2s)
            pt3s = self.get_points(points, self.triangle_pt3s)
            embeds += [module(p1s=pt1s, p2s=pt2s, p3s=pt3s) for module in self.rep_modules3]

        return self.group_embeds(embeds)
