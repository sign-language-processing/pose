class OpticalFlowCalculator:
    def __init__(self, fps, distance):
        self.fps = fps
        self.distance = distance

    def __call__(self, src):
        """Calculate the optical flow norm between frames, normalized by fps."""
        pre_src = src[:-1]
        post_src = src[1:]

        # Calculate distance
        src = self.distance(post_src, pre_src)

        # Normalize distance by fps
        src = src * self.fps

        return src
