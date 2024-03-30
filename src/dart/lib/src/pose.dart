import 'dart:typed_data';
import 'package:pose/src/pose_header.dart';
import 'package:pose/src/pose_body.dart';
import 'package:pose/reader.dart';

/// Class representing a pose.
///
/// This class contains a pose header and pose body.
class Pose {
  final PoseHeader header;
  final PoseBody body;

  /// Constructor for Pose.
  ///
  /// Takes [header] and [body] as parameters.
  Pose(this.header, this.body);

  /// Reads a Pose from the buffer.
  ///
  /// Takes [buffer] as a parameter.
  /// Returns a Pose instance.
  static Pose read(Uint8List buffer) {
    final BufferReader reader = BufferReader(buffer);
    final PoseHeader header = PoseHeader.read(reader);
    final PoseBody body = PoseBody.read(header, reader, {});

    return Pose(header, body);
  }
}
