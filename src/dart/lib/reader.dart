import 'dart:convert';
import 'dart:typed_data';
import 'package:pose/numdart.dart' show Struct, ConstStructs;
import 'package:pose/numdart.dart' as nd;

/// Class for reading data from a byte buffer.
class BufferReader {
  final Uint8List buffer;
  int readOffset;

  /// Constructs a BufferReader with the given byte buffer.
  BufferReader(this.buffer) : readOffset = 0;

  /// Returns the number of bytes left to read from the buffer.
  int bytesLeft() {
    return buffer.length - readOffset;
  }

  /// Reads a fixed-size chunk of bytes from the buffer.
  Uint8List unpackF(int size) {
    final Uint8List data = buffer.sublist(readOffset, readOffset + size);
    advance(Struct("", size));
    return data;
  }

  /// Reads numeric data from the buffer and constructs an n-dimensional array.
  List<dynamic> unpackNum(Struct s, List<int> shape) {
    final List<dynamic> arr = nd.ndarray(shape, s, buffer, readOffset);
    final int arrayBufferSize = nd.prod(shape);
    advance(s, arrayBufferSize);
    return arr;
  }

  /// Unpacks a single value from the buffer based on the given format.
  dynamic unpack(Struct s) {
    final Uint8List data = buffer.sublist(readOffset, readOffset + s.size);
    advance(s);

    final List<num> result = [];

    if (s.format == "<f") {
      result.add(nd.bytesToFloat(data));
    } else if (s.format == "<h") {
      result.add(nd.bytesToInt(data, signed: true));
    } else if (["<H", "<HH", "<HHH"].contains(s.format)) {
      for (int i = 0; i < data.length; i += 2) {
        result.add(nd.bytesToInt(data.sublist(i, i + 2)));
      }
    } else {
      throw ArgumentError("Invalid format.");
    }

    if (result.length == 1) {
      return result[0];
    }
    return result;
  }

  /// Advances the read offset by the size of the given format multiplied by [times].
  void advance(Struct s, [int times = 1]) {
    readOffset += s.size * times;
  }

  /// Unpacks a string from the buffer.
  String unpackStr() {
    final int length = unpack(ConstStructs.ushort);
    final Uint8List bytes_ = unpackF(length);
    return utf8.decode(bytes_);
  }
}
