import 'dart:io';
import 'dart:typed_data';
import 'package:pose/pose.dart';
import 'package:pose/src/pose_header.dart';
import 'package:test/test.dart';

void main() {
  Pose getPose(String filePath) {
    File file = File(filePath);
    Uint8List fileContent = file.readAsBytesSync();
    return Pose.read(fileContent);
  }

  group('Pose Tests', () {
    test("Mediapipe", () {
      Pose pose = getPose("test/data/mediapipe.pose");

      List confidence = pose.body.confidence;
      expect((confidence.length, confidence[0].length, confidence[0][0].length),
          equals((170, 1, 178)));

      List data = pose.body.data;
      expect((
        data.length,
        data[0].length,
        data[0][0].length,
        data[0][0][0].length
      ), equals((170, 1, 178, 3)));

      PoseHeaderDimensions dimensions = pose.header.dimensions;
      expect((dimensions.depth, dimensions.width, dimensions.height),
          equals((640, 1250, 1250)));

      expect(pose.body.fps, equals(24.0));
      expect(pose.header.version, equals(0.10000000149011612));
    });
    test("Mediapipe long", () {
      Pose pose = getPose("test/data/mediapipe_long.pose");

      List confidence = pose.body.confidence;
      expect((confidence.length, confidence[0].length, confidence[0][0].length),
          equals((278, 1, 178)));

      List data = pose.body.data;
      expect((
        data.length,
        data[0].length,
        data[0][0].length,
        data[0][0][0].length
      ), equals((278, 1, 178, 3)));

      PoseHeaderDimensions dimensions = pose.header.dimensions;
      expect((dimensions.depth, dimensions.width, dimensions.height),
          equals((640, 1250, 1250)));

      expect(pose.body.fps, equals(24.0));
      expect(pose.header.version, equals(0.10000000149011612));
    });
    test("Mediapipe hand normalized", () {
      Pose pose = getPose("test/data/mediapipe_hand_normalized.pose");

      List confidence = pose.body.confidence;
      expect((confidence.length, confidence[0].length, confidence[0][0].length),
          equals((170, 1, 21)));

      List data = pose.body.data;
      expect((
        data.length,
        data[0].length,
        data[0][0].length,
        data[0][0][0].length
      ), equals((170, 1, 21, 3)));

      PoseHeaderDimensions dimensions = pose.header.dimensions;
      expect((dimensions.depth, dimensions.width, dimensions.height),
          equals((1, 223, 229)));

      expect(pose.body.fps, equals(24.0));
      expect(pose.header.version, equals(0.10000000149011612));
    });
    test("Mediapipe long hand normalized", () {
      Pose pose = getPose("test/data/mediapipe_long_hand_normalized.pose");

      List confidence = pose.body.confidence;
      expect((confidence.length, confidence[0].length, confidence[0][0].length),
          equals((278, 1, 21)));

      List data = pose.body.data;
      expect((
        data.length,
        data[0].length,
        data[0][0].length,
        data[0][0][0].length
      ), equals((278, 1, 21, 3)));

      PoseHeaderDimensions dimensions = pose.header.dimensions;
      expect((dimensions.depth, dimensions.width, dimensions.height),
          equals((1, 221, 229)));

      expect(pose.body.fps, equals(24.0));
      expect(pose.header.version, equals(0.10000000149011612));
    });
  });
}
