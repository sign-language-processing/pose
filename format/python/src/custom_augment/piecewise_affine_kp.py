import imgaug as ia
import imgaug.augmenters as iaa

class PiecewiseAffineKP(iaa.PiecewiseAffine):
    def _augment_keypoints_by_samples(self, kpsois, samples):
        result = []

        for i, kpsoi in enumerate(kpsois):
            transformer = self._get_transformer(
                kpsoi.shape, kpsoi.shape, samples.nb_rows[i],
                samples.nb_cols[i], samples.jitter[i])

            if transformer is None or len(kpsoi.keypoints) == 0:
                result.append(kpsoi)
            else:
                coords = kpsoi.to_xy_array()
                coords_aug = transformer.inverse(coords)
                kpsoi_aug = ia.KeypointsOnImage.from_xy_array(
                    coords_aug,
                    shape=kpsoi.shape
                )

                # replace keypoints that are outside of the image with their
                # old coordinates
                kpsoi_aug.keypoints = [
                    kp if (
                        0 <= kp.x < kpsoi_aug.shape[0]
                        and 0 <= kp.y < kpsoi_aug.shape[1]
                    ) else kp_old
                    for kp, kp_old
                    in zip(kpsoi_aug.keypoints, kpsoi.keypoints)
                ]

                result.append(kpsoi_aug)

        return result