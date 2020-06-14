# Pose Format

This repository aims to include a complete toolkit for working with poses. 
It includes a new file format with Python and Javascript readers and writers, in hope to make usage simple.

### The File Format
The format supports any type of poses, arbitrary number of people, and arbitrary number of frames (for videos).

The main idea is having a `header` with instructions on how many points exists, where, and how to connect them.

The binary spec can be found in [format/spec.md](format/spec.md).

### Python Usage

To load a `.pose` file, use the `PoseReader` class:
```python
buffer = open("file.pose", "rb").read()
p = Pose.read(buffer)
```
By default, it uses NumPy for the data, but you can also use `torch` and `tensorflow` by writing:
```python
p = Pose.read(bugger, TorchPoseBody)
```

#### Data Normalization
To normalize all of the data to be in the same scale, we can normalize every pose by a constant feature of their body.
For example, for people we can use the the average span of their shoulders throughout the video to be a constant width.
```python
p.normalize(
    dist_p1=("pose_keypoints_2d", 2),  # RShoulder
    dist_p2=("pose_keypoints_2d", 5),  # LShoulder
)
```

#### Data Augmentation
We use [imgaug](https://github.com/aleju/imgaug) for data augmentation. Simply write an `imgaug:Augmentable`, and pass it to `augment2d`.
For example:
```python
seq = iaa.Sequential([
    iaa.HorizontalFlip(0.5),  # 50% of poses should be flipped left/right
    PiecewiseAffineKP(scale=(0.01, 0.05)),  # Distort keypoints
    iaa.Affine(
        rotate=(-5, 5),  # Rotate up to 10 degrees each way
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # Stretch or squash up to 50% each direction independently
        shear={"x": (-10, 10), "y": (-10, 10)}  # Shear X/Y up to 16 degrees independently
    ),
    iaa.PerspectiveTransform(scale=(0.0, 0.1)) # Change perspective
])

p.augment2d(seq)
```

#### Data Interpolation
To change the frame rate of a video, using data interpolation, use the `interpolate_fps` method which gets a new `fps` and a interpolation `kind`.
```python
p.interpolate_fps(24, kind='cubic')
p.interpolate_fps(24, kind='linear')
```

#### Local install
```bash
pip install -e /home/nlp/amit/PhD/PoseFormat/
```
