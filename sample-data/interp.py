import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

from lib.python.pose_format import Pose

buffer = open("test.pose", "rb").read()
p = Pose.read(buffer)

p.interpolate(25)
