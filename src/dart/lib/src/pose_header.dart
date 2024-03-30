import 'dart:math';
import 'package:pose/reader.dart';
import 'package:pose/numdart.dart';

/// Class representing a component of a pose header.
///
/// This class contains information about the points, limbs, colors, and format of a pose header component.
class PoseHeaderComponent {
  final String name;
  final List<String> points;
  final List<Point<int>> limbs;
  final List<List<dynamic>> colors;
  final String format;
  late List<int?> relativeLimbs;

  /// Constructor for PoseHeaderComponent.
  ///
  /// Takes [name], [points], [limbs], [colors], and [format] as parameters.
  PoseHeaderComponent(
      this.name, this.points, this.limbs, this.colors, this.format) {
    relativeLimbs = getRelativeLimbs();
  }

  /// Reads a PoseHeaderComponent from the reader based on the specified version.
  ///
  /// Takes [version] and [reader] as parameters.
  /// Returns a PoseHeaderComponent instance.
  static PoseHeaderComponent read(double version, BufferReader reader) {
    final String name = reader.unpackStr();
    final String pointFormat = reader.unpackStr();
    final int pointsCount = reader.unpack(ConstStructs.ushort);
    final int limbsCount = reader.unpack(ConstStructs.ushort);
    final int colorsCount = reader.unpack(ConstStructs.ushort);
    final List<String> points =
        List.generate(pointsCount, (_) => reader.unpackStr());
    final List<Point<int>> limbs = List.generate(
        limbsCount,
        (_) => Point<int>(reader.unpack(ConstStructs.ushort),
            reader.unpack(ConstStructs.ushort)));
    final List<List<dynamic>> colors = List.generate(
      colorsCount,
      (_) => [
        reader.unpack(ConstStructs.ushort),
        reader.unpack(ConstStructs.ushort),
        reader.unpack(ConstStructs.ushort)
      ],
    );

    return PoseHeaderComponent(name, points, limbs, colors, pointFormat);
  }

  /// Calculates the relative limbs for the component.
  ///
  /// Returns a list of relative limbs.
  List<int?> getRelativeLimbs() {
    final Map<int, int> limbsMap = {};
    for (int i = 0; i < limbs.length; i++) {
      limbsMap[limbs[i].y] = i;
    }
    return limbs.map((limb) => limbsMap[limb.x]).toList();
  }
}

/// Class representing dimensions of a pose header.
///
/// This class contains information about the width, height, and depth of a pose header.
class PoseHeaderDimensions {
  final int width;
  final int height;
  final int depth;

  /// Constructor for PoseHeaderDimensions.
  ///
  /// Takes [width], [height], and [depth] as parameters.
  PoseHeaderDimensions(this.width, this.height, this.depth);

  /// Reads PoseHeaderDimensions from the reader based on the specified version.
  ///
  /// Takes [version] and [reader] as parameters.
  /// Returns a PoseHeaderDimensions instance.
  static PoseHeaderDimensions read(double version, BufferReader reader) {
    final int width = reader.unpack(ConstStructs.ushort);
    final int height = reader.unpack(ConstStructs.ushort);
    final int depth = reader.unpack(ConstStructs.ushort);

    return PoseHeaderDimensions(width, height, depth);
  }
}

/// Class representing a pose header.
///
/// This class contains information about the version, dimensions, components, and bounding box status of a pose header.
class PoseHeader {
  final double version;
  final PoseHeaderDimensions dimensions;
  final List<PoseHeaderComponent> components;
  final bool isBbox;

  /// Constructor for PoseHeader.
  ///
  /// Takes [version], [dimensions], [components], and [isBbox] as parameters.
  PoseHeader(this.version, this.dimensions, this.components,
      {this.isBbox = false});

  /// Reads a PoseHeader from the reader.
  ///
  /// Takes [reader] as a parameter.
  /// Returns a PoseHeader instance.
  static PoseHeader read(BufferReader reader) {
    final double version = reader.unpack(ConstStructs.float);
    final PoseHeaderDimensions dimensions =
        PoseHeaderDimensions.read(version, reader);
    final int componentsCount = reader.unpack(ConstStructs.ushort);
    final List<PoseHeaderComponent> components = List.generate(
        componentsCount, (_) => PoseHeaderComponent.read(version, reader));

    return PoseHeader(version, dimensions, components);
  }
}
