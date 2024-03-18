// ignore_for_file: no_leading_underscores_for_local_identifiers

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart';
import 'package:pose/pose.dart';
import 'package:pose/numdart.dart' as nd;
import 'package:pose/numdart.dart' show MaskedArray;
import 'package:pose/src/pose_header.dart';
import 'package:tuple/tuple.dart';

/// Class responsible for visualizing poses.
class PoseVisualizer {
  final Pose pose;
  int? thickness;
  late double fps;
  Image? background;

  /// Constructs a PoseVisualizer with the given pose and optional thickness.
  PoseVisualizer(this.pose, {this.thickness}) : fps = pose.body.fps;

  /// Draws a single frame of the pose on the given image.
  Image _drawFrame(MaskedArray frame, List frameConfidence, Image img) {
    final Pixel pixelColor = img.getPixel(0, 0);
    final Tuple3<int, int, int> backgroundColor =
        Tuple3<int, int, int>.fromList(
            [pixelColor.r, pixelColor.g, pixelColor.b]);

    thickness ??= (sqrt(img.width * img.height) / 150).round();
    final int radius = (thickness! / 2).round();

    for (int i = 0; i < frame.data.length; i++) {
      final List person = frame.data[i];
      final List personConfidence = frameConfidence[i];

      final List<Tuple2<int, int>> points2D = List<Tuple2<int, int>>.from(
          person.map((p) => Tuple2<int, int>(p[0], p[1])));

      int idx = 0;
      for (PoseHeaderComponent component in pose.header.components) {
        final List<Tuple3<int, int, int>> colors = [
          for (List c in component.colors)
            Tuple3<int, int, int>.fromList(c) // can be reversed
        ];

        Tuple3<int, int, int> _pointColor(int pI) {
          final double opacity = personConfidence[pI + idx];
          final List nColor = colors[pI % component.colors.length]
              .toList()
              .map((e) => (e * opacity).toInt())
              .toList();
          final List newColor = backgroundColor
              .toList()
              .map((e) => (e * (1 - opacity)).toInt())
              .toList();

          final Tuple3<int, int, int> ndColor = Tuple3<int, int, int>.fromList([
            for (int i in Iterable.generate(nColor.length))
              (nColor[i] + newColor[i])
          ]);
          return ndColor;
        }

        // Draw Points
        for (int i = 0; i < component.points.length; i++) {
          if (personConfidence[i + idx] > 0) {
            final Tuple2<int, int> center =
                Tuple2<int, int>.fromList(person[i + idx].take(2).toList());
            final Tuple3<int, int, int> colorTuple = _pointColor(i);

            drawCircle(
              img,
              x: center.item1,
              y: center.item2,
              radius: radius,
              color: ColorFloat16.fromList([
                colorTuple.item1,
                colorTuple.item2,
                colorTuple.item3
              ].map((e) => (e.toDouble())).toList()),
            );
          }
        }

        if (pose.header.isBbox) {
          final Tuple2<int, int> point1 = points2D[0 + idx];
          final Tuple2<int, int> point2 = points2D[1 + idx];

          final Tuple3<int, int, int> temp1 = _pointColor(0);
          final Tuple3<int, int, int> temp2 = _pointColor(1);

          drawRect(img,
              x1: point1.item1,
              y1: point1.item2,
              x2: point2.item1,
              y2: point2.item2,
              color: ColorFloat16.fromList(nd.mean([
                [temp1.item1, temp1.item2, temp1.item3],
                [temp2.item1, temp2.item2, temp2.item3]
              ], axis: 0)),
              thickness: thickness!);
        } else {
          // Draw Limbs
          for (var limb in component.limbs) {
            if (personConfidence[limb.x + idx] > 0 &&
                personConfidence[limb.y + idx] > 0) {
              final Tuple2<int, int> point1 = points2D[limb.x + idx];
              final Tuple2<int, int> point2 = points2D[limb.y + idx];

              final Tuple3<int, int, int> temp1 = _pointColor(limb.x);
              final Tuple3<int, int, int> temp2 = _pointColor(limb.y);

              drawLine(img,
                  x1: point1.item1,
                  y1: point1.item2,
                  x2: point2.item1,
                  y2: point2.item2,
                  color: ColorFloat16.fromList(nd.mean([
                    [temp1.item1, temp1.item2, temp1.item3],
                    [temp2.item1, temp2.item2, temp2.item3]
                  ], axis: 0)),
                  thickness: thickness!);
            }
          }
        }

        idx += component.points.length;
      }
    }

    return img;
  }

  /// Generates frames for the pose visualization.
  Stream<Image> draw(
      {List<double> backgroundColor = const [0, 0, 0], int? maxFrames}) async* {
    final List intFrames = MaskedArray(pose.body.data, []).round();

    final background = Image(
      width: pose.header.dimensions.width,
      height: pose.header.dimensions.height,
      backgroundColor: ColorFloat16.fromList(backgroundColor),
    );

    for (int i = 0;
        i < min(intFrames.length, maxFrames ?? intFrames.length);
        i++) {
      yield _drawFrame(MaskedArray(intFrames[i], []), pose.body.confidence[i],
          background.clone());
    }
  }

  // Generate GIF from frames
  Future<Uint8List> generateGif(Stream<Image> frames, {double fps = 24}) async {
    final int frameDuration = (100 / fps).round();
    final GifEncoder encoder = GifEncoder(delay: 0, repeat: 0);

    await for (Image frame in frames) {
      encoder.addFrame(frame, duration: frameDuration);
    }

    final Uint8List? image = encoder.finish();
    if (image != null) {
      return image;
    }

    throw Exception('Failed to encode GIF.');
  }

  /// Saves the visualization as a GIF.
  Future<File> saveGif(String fileName, Stream<Image> frames,
      {double fps = 24}) async {
    Uint8List image = await generateGif(frames, fps: fps);
    return await File(fileName).writeAsBytes(image);
  }
}

class FastAndUglyPoseVisualizer extends PoseVisualizer {
  FastAndUglyPoseVisualizer(Pose pose, {int? thickness})
      : super(pose, thickness: thickness);

  Image _uglyDrawFrame(MaskedArray frame, Image img, int color) {
    final Tuple2<int, int> ignoredPoint = Tuple2<int, int>.fromList([0, 0]);

    //  Note: this can be made faster by drawing polylines instead of lines
    final thickness = 1;

    for (int i = 0; i < frame.data.length; i++) {
      final List person = frame.data[i];

      final List<Tuple2<int, int>> points2D = List<Tuple2<int, int>>.from(
          person.map((p) => Tuple2<int, int>(p[0], p[1])));

      int idx = 0;
      for (PoseHeaderComponent component in pose.header.components) {
        for (var limb in component.limbs) {
          final Tuple2<int, int> point1 = points2D[limb.x + idx];
          final Tuple2<int, int> point2 = points2D[limb.y + idx];

          if (point1 != ignoredPoint && point2 != ignoredPoint) {
            // Antialiasing is a bit slow, but necessary
            drawLine(
              img,
              x1: point1.item1,
              y1: point1.item2,
              x2: point2.item1,
              y2: point2.item2,
              antialias: true,
              color: ColorFloat16.fromList([color.toDouble()]),
              thickness: thickness,
            );
          }
        }
        idx += component.points.length;
      }
    }
    return img;
  }

  Stream<Image> uglyDraw(
      {int backgroundColor = 0, int foregroundColor = 255}) async* {
    final List intFrames = MaskedArray(pose.body.data, []).round();

    final background = Image(
      width: pose.header.dimensions.width,
      height: pose.header.dimensions.height,
      backgroundColor: ColorFloat16.fromList([backgroundColor.toDouble()]),
    );

    for (int i = 0; i < intFrames.length; i++) {
      yield _uglyDrawFrame(
        MaskedArray(intFrames[i], []),
        background.clone(),
        foregroundColor,
      );
    }
  }
}
