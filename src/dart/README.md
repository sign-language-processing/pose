# Pose

[![pub package](https://img.shields.io/pub/v/pose.svg)](https://pub.dev/packages/pose)

This is `dart` implementation of its [python counterpart](https://github.com/sign-language-processing/pose/tree/master/src/python) with limited features

This repository helps developers interested in Sign Language Processing (SLP) by providing a complete toolkit for working with poses.

## File Format Structure

The file format is designed to accommodate any pose type, an arbitrary number of people, and an indefinite number of frames. 
Therefore it is also very suitable for video data, and not only single frames.

At the core of the file format is `Header` and a `Body`.

* The header for example contains the following information:

    - The total number of pose points. (How many points exist.)
    - The exact positions of these points. (Where do they exist.)
    - The connections between these points. (How are they connected.)

## Features

- ✔️ Reading
- ❌ Normalizing
- ❌ Augmentation
- ❌ Interpolation
- ✔️ Visualization (2x slow compared to python and supports only GIF)

## Usage

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose/pose.dart';

void main() async {
  File file = File("pose_file.pose");
  Uint8List fileContent = file.readAsBytesSync();
  Pose pose = Pose.read(fileContent);
  PoseVisualizer p = PoseVisualizer(pose);
  await p.saveGif("demo.gif", p.draw());
}
```

![Demo Gif](https://raw.githubusercontent.com/sign-language-processing/pose/master/src/dart/assets/demo.gif)
