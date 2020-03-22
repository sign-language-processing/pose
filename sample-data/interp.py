import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d

# points = np.array([1, 2, 3, 4, 0, 6])
# mask = [0, 0, 0, 0, 1, 0]
# points = ma.array(points, mask=mask)
# print(points)  # [1 2 3 4 -- 6]
#
# lin = np.linspace(0, 1, 6)
#
# # Undesired behavior
# f = interp1d(lin, points, axis=0, kind='cubic')
# print(f(lin))  # [1  2 3 4 -8.8817842e-16 6]
#
# # Desired behavior
# compressed_lin = [0, 0.2, 0.4, 0.6, 1]
# compressed_points = np.array([1,2,3,4,6])
# f = interp1d(compressed_lin, compressed_points, axis=0, kind='cubic')
# print(f(lin)) # [1 2 3 4 5 6]


points = np.array([1, 2, 3, 4, 5, 6])

points = ma.stack([
    ma.array(points, mask=[0, 0, 0, 0, 1, 0]),
    ma.array(points, mask=[0, 0, 1, 0, 0, 0]),
])
print(points)

print(np.reshape(points.compressed(), (2, 5)))
