class OpticalFlowCalculator:
    """
    Classe used for computing optical flow between frames using distance function
    
    Parameters
    ----------
    fps : float
        frames per second; used to normalize optical flow computation
    distance : callable
        function to compute distance (or optical flow) between two frames (post/pre-src)
    """

    def __init__(self, fps, distance):
        self.fps = fps
        self.distance = distance

    def __call__(self, src):
        """
        Calculate the optical flow norm between frames, normalized by fps
        
        Parameters
        ----------
        src : torch.Tensor
            source tensor representing the frames

        Returns
        -------
        torch.Tensor
            normalized optical flow values between consecutive frames (pre-/post-src)
        """

        pre_src = src[:-1]
        post_src = src[1:]

        # Calculate distance
        src = self.distance(post_src, pre_src)

        # Normalize distance by fps
        src = src * self.fps

        return src
