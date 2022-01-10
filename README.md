# Pose Format

This repository aims to include a complete toolkit for working with poses. 
It includes a new file format with Python and Javascript readers and writers, in hope to make usage simple.

### The File Format
The format supports any type of poses, arbitrary number of people, and arbitrary number of frames (for videos).

The main idea is having a `header` with instructions on how many points exists, where, and how to connect them.

The binary spec can be found in [lib/specs/v0.1.md](lib/specs/v0.1.md).

### Python Usage
```bash
pip install pose-format
```

To load a `.pose` file, use the `PoseReader` class:
```python
from pose_format.pose import Pose

buffer = open("file.pose", "rb").read()
p = Pose.read(buffer)
```
By default, it uses NumPy for the data, but you can also use `torch` and `tensorflow` by writing:
```python
from pose_format.pose import Pose
from pose_format.torch.pose_body import TorchPoseBody
from pose_format.tensorflow.pose_body import TensorflowPoseBody

buffer = open("file.pose", "rb").read()

p = Pose.read(buffer, TorchPoseBody)
p = Pose.read(buffer, TensorflowPoseBody)
```

After creating a pose object that holds numpy data, it can also be converted to torch or tensorflow:
```python
from pose_format.numpy import NumPyPoseBody

# create a pose object that internally stores the data as a numpy array
p = Pose.read(buffer, NumPyPoseBody)

# return stored data as a torch tensor
p.torch()

# return stored data as a tensorflow tensor
p.tensorflow()
```

### Common pose processing operations

Once poses are loaded, the library offers many ways to manipulate `Pose` objects.

#### Data normalization (skeleton scale)
To normalize all of the data to be in the same scale, we can normalize every pose by a constant feature of their body.
For example, for people we can use the average span of their shoulders throughout the video to be a constant width.
```python
p.normalize(p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
))
```

#### Data normalization (keypoint distribution)
Keypoint values can be standardized to have a mean of zero and unit variance:
```python
p.normalize_distribution()
```

The default behaviour is to compute a separate mean and standard deviation for each keypoint and each dimension (usually x and y).
The `axis` argument can be used to customize this. For instance, to compute only two global means and standard deviations for the
x and y dimension:

```python
p.normalize_distribution(axis=(0, 1, 2))
```

#### Data augmentation
```python
p.augment2d(rotation_std=0.2, shear_std=0.2, scale_std=0.2)
```

#### Data interpolation
To change the frame rate of a video, using data interpolation, use the `interpolate_fps` method which gets a new `fps` and a interpolation `kind`.
```python
p.interpolate_fps(24, kind='cubic')
p.interpolate_fps(24, kind='linear')
```

### Loading OpenPose data

To load an OpenPose `directory`, use the `load_openpose_directory` utility:
```python
from pose_format.utils.openpose import load_openpose_directory

directory = "/path/to/openpose/directory"
p = load_openpose_directory(directory, fps=24, width=1000, height=1000)
```

### Testing
Use bazel to run tests
```sh
cd pose_format
bazel test ... --test_output=errors
```

Alternatively, use a different testing framework to run tests, such as pytest. To run an individual
test file:
```sh
pytest pose_format/tensorflow/masked/tensor_test.py
```