import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ImageClassifier {
  late Interpreter _interpreter;

  // Cargar el modelo
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilenet.tflite');
      print("Modelo cargado correctamente.");
    } catch (e) {
      print("Error al cargar el modelo: $e");
    }
  }

  // Proceso para clasificar la imagen
  Future<void> classifyImageFolder(String folderPath, List<String> tags) async {
    final directory = Directory(folderPath);
    final imageFiles = directory.listSync().where((file) => file.path.endsWith('.jpg') || file.path.endsWith('.png'));

    for (var file in imageFiles) {
      File originalImage = File(file.path);
      File tempImage = await _resizeImage(originalImage);

      List<double> probabilities = await _classifyImage(tempImage);

      // Encontrar el tag con la mayor probabilidad
      Map<String, double> tagProbabilities = _mapTagsToProbabilities(probabilities, tags);
      var bestTag = tagProbabilities.entries.reduce((a, b) => a.value > b.value ? a : b).key;

      // Mover la imagen original a la carpeta del tag más probable
      String destinationFolder = '$folderPath/$bestTag';
      Directory(destinationFolder).createSync();
      originalImage.renameSync('$destinationFolder/${file.uri.pathSegments.last}');

      // Eliminar la imagen temporal
      tempImage.deleteSync();
    }
  }

  // Redimensionar la imagen a 224x224 para la clasificación
  Future<File> _resizeImage(File image) async {
    final imageBytes = await image.readAsBytes();
    final decodedImg = img.decodeImage(imageBytes);

    if (decodedImg == null) {
      throw Exception("Error decoding image.");
    }

    // Si la imagen es PNG y tiene transparencia, rellenar con blanco
    img.Image processedImg;
    if (decodedImg.numChannels == 4) {
      processedImg = img.Image(width: decodedImg.width, height: decodedImg.height);
      img.fill(processedImg, color: img.ColorRgb8(255, 255, 255));
      img.compositeImage(processedImg, decodedImg);
    } else {
      processedImg = decodedImg;
    }

    // Redimensionar la imagen
    final resizedImg = img.copyResize(processedImg, width: 224, height: 224);

    // Guardar la imagen redimensionada como JPG
    final resizedBytes = img.encodeJpg(resizedImg);
    final tempImage = File('${image.path}_resized.jpg');
    await tempImage.writeAsBytes(resizedBytes);

    return tempImage;
  }

  // Clasificar la imagen utilizando el modelo
  Future<List<double>> _classifyImage(File image) async {
    var input = await _preprocessImage(image.path);

    var output = List.filled(1001, 0.0).reshape([1, 1001]);
    _interpreter.run(input, output);

    return output[0];
  }

  // Preprocesar la imagen redimensionada
  Future<List<List<List<List<double>>>>> _preprocessImage(String imagePath) async {
    final inputImg = await File(imagePath).readAsBytes();
    final decodedImg = img.decodeImage(inputImg);

    if (decodedImg == null) {
      throw Exception("Error decoding image.");
    }

    final resizedBytes = decodedImg.getBytes();

    List<List<List<List<double>>>> input = List.generate(
      1,
          (batch) => List.generate(
        224,
            (i) => List.generate(
          224,
              (j) => List.generate(3, (c) {
            int offset = (i * 224 + j) * 3;
            return resizedBytes[offset + c] / 255.0;
          }),
        ),
      ),
    );

    return input;
  }

  // Mapear probabilidades a los tags
  Map<String, double> _mapTagsToProbabilities(List<double> probabilities, List<String> tags) {
    Map<String, double> tagProbabilities = {};

    for (int i = 0; i < tags.length; i++) {
      if (i < probabilities.length) {
        tagProbabilities[tags[i]] = probabilities[i];
      }
    }

    return tagProbabilities;
  }
}