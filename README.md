# `pose-format`

This repository helps developers interested in Sign Language Processing (SLP) by providing a complete toolkit for working with poses. 
It includes a file format with Python and Javascript readers and writers, which hopefully makes its usage simple.

### File Format Structure
The file format is designed to accommodate any pose type, an arbitrary number of people, and an indefinite number of frames. 
Therefore it is also very suitable for video data, and not only single frames.

At the core of the file format is `Header` and a `Body`.

* The header for example contains the following information:

    - The total number of pose points. (How many points exist.)
    - The exact positions of these points. (Where do they exist.)
    - The connections between these points. (How are they connected.)

More about the header and the body details and their binary specifics can be found in [docs/specs/v0.1.md](specs_v01.rst#specs_v01).

### Python Usage Guide: 

#### 1. Installation: 

```bash
pip install pose-format
```

#### 2. Estimating Pose from Video:

```bash
video_to_pose --format mediapipe -i example.mp4 -o example.pose

# Or if you have a directory of videos
videos_to_poses --format mediapipe --directory /path/to/videos

# You can also specify additional arguments
video_to_pose --format mediapipe -i example.mp4 -o example.pose \
  --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"

# Recursively search for videos within a directory, and process them 10 at a time
videos_to_poses --format mediapipe -num-workers 10 --recursive --directory /path/to/videos 

```

#### 3. Reading `.pose` Files: 

To load a `.pose` file, use the `Pose` class.

```python
from pose_format import Pose

data_buffer = open("file.pose", "rb").read()
pose = Pose.read(data_buffer)

numpy_data = pose.body.data
confidence_measure  = pose.body.confidence
```

By default, the library uses NumPy (`numpy`) for storing and manipulating pose data. However, integration with PyTorch (`torch`) and TensorFlow (`tensorflow`) is supported, just do the following: 

```python
from pose_format.pose import Pose

data_buffer = open("file.pose", "rb").read()

# Load data as a PyTorch tensor:
from pose_format.torch import TorchPoseBody
pose = Pose.read(buffer, TorchPoseBody)

# Or as a TensorFlow tensor:
from pose_format.tensorflow.pose_body import TensorflowPoseBody
pose = Pose.read(buffer, TensorflowPoseBody)
```

If you initially loaded the data in a NumPy format and want to convert it to PyTorch or TensorFlow format, do the following:

```python
from pose_format.numpy import NumPyPoseBody

# Create a pose object that internally stores data as a NumPy array
pose = Pose.read(buffer, NumPyPoseBody)

# Convert to PyTorch:
pose.torch()

# Convert to TensorFlow:
pose.tensorflow()
```

#### 4. Data Manipulation: 

Once poses are loaded, the library offers many ways to manipulate the created `Pose` objects. 

##### Normalizing Data: 

Maintaining data consistency is very important and data normalization is one method to do this. By normalizing the pose data, all pose information is brought to a consistent scale. This allows every pose to be normalized based on a constant feature of the body.

For instance, you can set the shoulder width to a consistent measurement across all data points. This is useful for comparing poses across different individuals. 

* See this example for manually specifying a standard body feature, such as the shoulder width, for normalization:

```python
pose.normalize(p.header.normalization_info(
    p1=("pose_keypoints_2d", "RShoulder"),
    p2=("pose_keypoints_2d", "LShoulder")
))
```

* If normalization info is not specified, normalize() will automatically base normalization on shoulder joints.

```python
pose.normalize() # same result as above, but attempts to automatically select shoulder points based on format
```

* Keypoint values can be standardized to have a mean of zero and unit variance:

```python

# Normalize all keypoints:
pose.normalize_distribution()
```

The usual way to do this is to compute a separate mean and standard deviation for each keypoint and each dimension (usually x and y). This can be achieved with the `axis` argument of `normalize_distribution`. 

```python

# Normalize each keypoint separately:
pose.normalize_distribution(axis=(0, 1, 2))
```

##### Augmentation: 
Data augmentation is very important for improving the performance of machine learning models. We now provide a simple way to augment pose data.

* Apply 2D data augmentation:

```python

pose.augment2d(rotation_std=0.2, shear_std=0.2, scale_std=0.2)
```

##### Interpolation
If you're dealing with video data and need to adjust its frame rate, use the interpolation functions. 

To change the frame rate of a video, using data interpolation, use the `interpolate_fps` method which gets a new `fps` and a interpolation `kind`.

```python
pose.interpolate_fps(24, kind='cubic')
pose.interpolate_fps(24, kind='linear')
```

#### 5. Visualization
You can visualize the poses stored in the `.pose` files.
Use the `PoseVisualizer` class for visualization tasks, such as generating videos or overlaying pose data on existing videos.

* To save as a video: 
```python
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


with open("example.pose", "rb") as f:
    pose = Pose.read(f.read())

v = PoseVisualizer(pose)

v.save_video("example.mp4", v.draw())
```

* To overlay pose on an existing video: 


```python
# Draws pose on top of video. 
v.save_video("example.mp4", v.draw_on_video("background_video_path.mp4"))
```

* Convert to GIF: 
For those using Google Colab, poses can be converted to GIFs for easy inspection. 

```python
# In a Colab notebook

from IPython.display import Image

v.save_gif("test.gif", v.draw())

display(Image(open('test.gif','rb').read()))
```

#### 6. Integration with External Data Sources:
If you have pose data in OpenPose or MediaPipe Holistic format, you can easily import it. 

##### Loading OpenPose and MediaPipe Holistic Data

* For OpenPose: 

To load an OpenPose `directory`, use the `load_openpose_directory` utility:


```python
from pose_format.utils.openpose import load_openpose_directory

directory = "/path/to/openpose/directory"
pose = load_openpose_directory(directory, fps=24, width=1000, height=1000)
```

* For MediaPipe Holistic: 

Similarly, to load a MediaPipe Holistic `directory`, use the `load_MediaPipe_directory` utility:

```python
from pose_format.utils.holistic import load_MediaPipe_directory

directory = "/path/to/holistic/directory"
pose = load_MediaPipe_directory(directory, fps=24, width=1000, height=1000)
```

### Running Tests:

To ensure the integrity of the toolkit, you can run tests using Bazel:

* Using bazel:

```bash
cd src/python/pose_format
bazel test ... --test_output=errors
```

Alternatively, use a different testing framework to run tests, such as pytest. To run an individual test file.

* Or employ pytest:

```bash
# From src/python directory
pytest .
# or for a single file
pytest pose_format/tensorflow/masked/tensor_test.py
```

### Acknowledging the Work 

If you use our toolkit in your research or projects, please consider citing the work:

```bibtex
@misc{moryossef2021pose-format, 
    title={pose-format: Library for viewing, augmenting, and handling .pose files},
    author={Moryossef, Amit and M\"{u}ller, Mathias and Fahrni, Rebecka},
    howpublished={\url{https://github.com/sign-language-processing/pose}},
    year={2021}
}
```
