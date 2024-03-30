import 'dart:io';
import 'dart:typed_data';
import 'package:pose/pose.dart';

void main() async {
  Stopwatch stopwatch = Stopwatch()..start();

  File file = File("pose_file.pose");
  Uint8List fileContent = file.readAsBytesSync();
  print("File Read");

  Pose pose = Pose.read(fileContent);
  print("File Loaded");

  PoseVisualizer p = PoseVisualizer(pose, thickness: 2);
  print("File Visualized");

  File giffile = await p.saveGif("demo.gif", p.draw());
  print("File Saved ${giffile.path}");

  print('Time taken : ${stopwatch.elapsed}');
}
