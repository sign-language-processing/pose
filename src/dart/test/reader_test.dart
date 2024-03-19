import 'dart:convert';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:pose/reader.dart';
import 'package:pose/numdart.dart';

void main() {
  group('BufferReader Tests', () {
    test('Advance Test', () {
      BufferReader reader = BufferReader(Uint8List(0));
      reader.advance(ConstStructs.float, 10);
      expect(reader.readOffset, equals(40));
    });

    test('Unpack float Test', () {
      double expected = 5.5;
      Uint8List buffer = Uint8List.fromList(
          [0x00, 0x00, 0xb0, 0x40]); // Equivalent to struct.pack("<f", 5.5)

      BufferReader reader = BufferReader(buffer);
      double unpacked = reader.unpack(ConstStructs.float);
      expect(unpacked, equals(expected));
    });

    test('Unpack string Test', () {
      String expected = "hello";
      List<int> stringBytes = utf8.encode(expected);
      int length = stringBytes.length;
      Uint8List buffer = Uint8List.fromList([
        length & 0xFF, // Low byte of the length
        (length >> 8) & 0xFF, // High byte of the length
        ...stringBytes // String data bytes
      ]); // Equivalent to struct.pack("<H%ds" % len(s), len(s), bytes(s, 'utf8'))
      BufferReader reader = BufferReader(buffer);
      String unpacked = reader.unpackStr();
      expect(unpacked, equals(expected));
    });

    test('Unpack num Test', () {
      List<List<double>> expected = [
        [1.0, 2.5],
        [3.5, 4.5]
      ];

      List<List<double>> modifiedExpected = [
        [0.9, 2.5],
        [3.5, 4.5]
      ]; // Expected result after modification

      Uint8List buffer = Uint8List.fromList([
        0x00,
        0x00,
        0x80,
        0x3f,
        0x00,
        0x00,
        0x20,
        0x40,
        0x00,
        0x00,
        0x60,
        0x40,
        0x00,
        0x00,
        0x90,
        0x40
      ]); // Equivalent to struct.pack("<ffff", 1., 2.5, 3.5, 4.5)

      BufferReader reader = BufferReader(buffer);
      List<dynamic> unpacked = reader.unpackNum(ConstStructs.float, [2, 2]);
      expect(unpacked, equals(expected));

      unpacked[0][0] -= 0.1; // Modify the first element of the array
      expect(unpacked, equals(modifiedExpected));
    });
  });
}

// print Uint8List in string format
String formatBytes(Uint8List bytes) {
  StringBuffer buffer = StringBuffer('b\'');
  for (int byte in bytes) {
    if (byte >= 32 && byte <= 126) {
      buffer.write(String.fromCharCode(byte));
    } else {
      buffer.write('\\x${byte.toRadixString(16).padLeft(2, '0')}');
    }
  }
  buffer.write('\'');
  return buffer.toString();
}
