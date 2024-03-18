// ignore_for_file: non_constant_identifier_names

import 'dart:math' as math;
import 'dart:typed_data';

/// Represents a structure with a specified format and size.
class Struct {
  final String format;
  final int size;

  Struct(this.format, this.size);
}

/// Contains predefined Struct objects for commonly used data types.
class ConstStructs {
  static final Struct float = Struct("<f", 4);
  static final Struct short = Struct("<h", 2);
  static final Struct ushort = Struct("<H", 2);
  static final Struct double_ushort = Struct("<HH", 4);
  static final Struct triple_ushort = Struct("<HHH", 6);
}

/// Converts bytes to a floating-point number.
double bytesToFloat(List<int> bytesData) {
  int intValue = 0;
  for (int i = 0; i < bytesData.length; i++) {
    intValue += bytesData[i] << (i * 8);
  }
  final int sign = (intValue & (1 << (8 * bytesData.length - 1))) != 0 ? -1 : 1;
  final int exponent = ((intValue >> 23) & 0xFF) - 127;
  final int mantissa = (intValue & 0x7FFFFF) | 0x800000;
  final num result = sign * mantissa * math.pow(2, exponent - 23);
  return result.toDouble();
}

/// Converts bytes to an integer.
int bytesToInt(List<int> bytesData,
    {bool signed = false, Endian byteOrder = Endian.little}) {
  final ByteData byteData = ByteData.sublistView(Uint8List.fromList(bytesData));
  if (signed) {
    switch (bytesData.length) {
      case 1:
        return byteData.getInt8(0);
      case 2:
        return byteData.getInt16(0, byteOrder);
      case 4:
        return byteData.getInt32(0, byteOrder);
      case 8:
        return byteData.getInt64(0, byteOrder);
      default:
        throw ArgumentError('Invalid byte length for signed integer');
    }
  } else {
    switch (bytesData.length) {
      case 1:
        return byteData.getUint8(0);
      case 2:
        return byteData.getUint16(0, byteOrder);
      case 4:
        return byteData.getUint32(0, byteOrder);
      case 8:
        return byteData.getUint64(0, byteOrder);
      default:
        throw ArgumentError('Invalid byte length for unsigned integer');
    }
  }
}

/// Calculates the product of a sequence of integers.
int prod(List<int> seq) {
  int result = 1;
  for (int num in seq) {
    result *= num;
  }
  return result;
}

/// Represents a function that converts bytes to a numeric value.
typedef NumConversionFunction = num Function(List<int>);

/// Constructs an n-dimensional array from a buffer based on the given shape and format.
List<dynamic> ndarray(List<int> shape, Struct s, List<int> buffer, int offset) {
  NumConversionFunction func;
  if (s.format == "<H") {
    func = bytesToInt;
  } else if (s.format == "<f") {
    func = bytesToFloat;
  } else {
    throw ArgumentError("Format should be <H or <f");
  }

  final List<dynamic> matrix = [];

  if (shape.length == 2) {
    for (int i = 0; i < shape[0]; i++) {
      List<dynamic> row = [];
      for (int j = 0; j < shape[1]; j++) {
        row.add(func(buffer.sublist(offset, offset + s.size)));
        offset += s.size;
      }
      matrix.add(row);
    }
  } else if (shape.length == 3) {
    for (int i = 0; i < shape[0]; i++) {
      List<dynamic> innerMatrix = [];
      for (int j = 0; j < shape[1]; j++) {
        List<dynamic> row = [];
        for (int k = 0; k < shape[2]; k++) {
          row.add(func(buffer.sublist(offset, offset + s.size)));
          offset += s.size;
        }
        innerMatrix.add(row);
      }
      matrix.add(innerMatrix);
    }
  } else if (shape.length == 4) {
    for (int i = 0; i < shape[0]; i++) {
      List<dynamic> innerMatrix1 = [];
      for (int j = 0; j < shape[1]; j++) {
        List<dynamic> innerMatrix2 = [];
        for (int k = 0; k < shape[2]; k++) {
          List<dynamic> innerMatrix3 = [];
          for (int l = 0; l < shape[3]; l++) {
            innerMatrix3.add(func(buffer.sublist(offset, offset + s.size)));
            offset += s.size;
          }
          innerMatrix2.add(innerMatrix3);
        }
        innerMatrix1.add(innerMatrix2);
      }
      matrix.add(innerMatrix1);
    }
  } else {
    throw ArgumentError("Shape length must be 2, 3, or 4.");
  }

  return matrix;
}

/// Computes the mean along a specified axis.
List<double> mean(List<List<num>> values, {int? axis}) {
  if (values.isEmpty) {
    return [double.nan]; // Return NaN for empty lists
  }

  if (axis == null) {
    final List<num> flattenedValues = values.expand((list) => list).toList();
    final num total = flattenedValues.reduce((a, b) => a + b);
    return [total / flattenedValues.length];
  } else if (axis == 0) {
    final List<num> columnSums = List<num>.filled(values[0].length, 0);
    for (List<num> row in values) {
      for (int i = 0; i < row.length; i++) {
        columnSums[i] += row[i];
      }
    }
    return columnSums.map((sum) => sum / values.length).toList();
  } else if (axis == 1) {
    return values
        .map((row) => row.reduce((a, b) => a + b) / row.length)
        .toList();
  } else {
    throw ArgumentError("Axis must be null, 0, or 1.");
  }
}

/// Represents a masked array with data and mask.
class MaskedArray {
  final List data;
  final List<List<int>> mask;

  MaskedArray(this.data, this.mask);

  /// Rounds the data values in the masked array.
  MaskedArray rint() {
    final List<List<dynamic>> roundedData = [];
    for (int i = 0; i < data.length; i++) {
      List<dynamic> row = [];
      for (int j = 0; j < data[i].length; j++) {
        if (mask[i][j] != 0) {
          row.add(_round(data[i][j]));
        } else {
          row.add(data[i][j]);
        }
      }
      roundedData.add(row);
    }
    return MaskedArray(roundedData, mask);
  }

  List round() {
    return _roundList(data);
  }

  dynamic _round(dynamic elem) {
    if (elem is List) {
      return _roundList(elem);
    } else {
      return (elem).round();
    }
  }

  List _roundList(List<dynamic> elem) {
    List<dynamic> roundedList = [];
    for (int i = 0; i < elem.length; i++) {
      roundedList.add(_round(elem[i]));
    }
    return roundedList;
  }
}
