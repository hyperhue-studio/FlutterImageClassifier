import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/services.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Solicitar permisos antes de usar almacenamiento
  await checkAndRequestPermissions();

  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String? _folderPath;
  bool _isLoading = false;
  String _statusMessage = '';
  ImageClassifier _classifier = ImageClassifier();

  Future<void> _selectFolder() async {
    // Solicitar permiso para acceder a almacenamiento y después, seleccionar la carpeta
    if (await Permission.storage.request().isGranted) {
      String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
      if (selectedDirectory != null) {
        setState(() {
          _folderPath = selectedDirectory;
          _isLoading = true;
          _statusMessage = 'Cargando el modelo y las etiquetas...';
        });

        // Cargar el modelo
        bool modelLoaded = await _classifier.loadModel();
        if (modelLoaded) {
          setState(() {
            _statusMessage = 'Modelo y etiquetas cargados correctamente.';
          });

          // Clasificar imágenes en la carpeta seleccionada
          setState(() {
            _statusMessage = 'Clasificando imágenes...';
          });
          await _classifier.classifyImageFolder(_folderPath!);

          setState(() {
            _isLoading = false;
            _statusMessage = 'Clasificación completada y archivos movidos.';
          });

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Clasificación completada y archivos movidos.')),
          );
        } else {
          setState(() {
            _isLoading = false;
            _statusMessage = 'Error al cargar el modelo o las etiquetas.';
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error al cargar el modelo o las etiquetas.')),
          );
        }
      }
    } else {
      // Si no se concedió el permiso, muestra un mensaje o maneja el error
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Permiso para acceder al almacenamiento denegado')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Clasificador de Imágenes'),
        ),
        body: Center(
          child: _isLoading
              ? Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 20),
              Text(_statusMessage),
            ],
          )
              : Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: _selectFolder,
                child: Text('Seleccionar Carpeta'),
              ),
              SizedBox(height: 20),
              _folderPath != null
                  ? Text('Carpeta seleccionada: $_folderPath')
                  : Text('No se ha seleccionado ninguna carpeta'),
              SizedBox(height: 20),
              Text(_statusMessage),
            ],
          ),
        ),
      ),
    );
  }
}

Future<void> checkAndRequestPermissions() async {
  // Verificar si ya tiene permisos de almacenamiento
  if (await Permission.storage.isGranted) {
    return;
  }

  // Solicitar permisos de almacenamiento
  Map<Permission, PermissionStatus> statuses = await [
    Permission.storage,
    if (await Permission.manageExternalStorage.isDenied) Permission.manageExternalStorage, // Android 11+
  ].request();

  // Verificar el estado del permiso
  if (statuses[Permission.storage] != PermissionStatus.granted ||
      (statuses[Permission.manageExternalStorage] != PermissionStatus.granted && statuses.containsKey(Permission.manageExternalStorage))) {
    print("Permisos de almacenamiento denegados.");
  } else {
    print("Permisos de almacenamiento concedidos.");
  }
}

class ImageClassifier {
  late Interpreter _interpreter;
  late List<String> _labels;

  // Cargar el modelo y las etiquetas
  Future<bool> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilenet.tflite');
      _labels = (await rootBundle.loadString('assets/labels.txt')).split('\n');      print("Modelo y etiquetas cargados correctamente.");
      return true;
    } catch (e) {
      print("Error al cargar el modelo o las etiquetas: $e");
      return false;
    }
  }

  // Proceso para clasificar la imagen
  Future<void> classifyImageFolder(String folderPath) async {
    final directory = Directory(folderPath);
    final imageFiles = directory
        .listSync()
        .where((file) => file.path.endsWith('.jpg') || file.path.endsWith('.png'));

    for (var file in imageFiles) {
      File originalImage = File(file.path);
      File tempImage = await _resizeImage(originalImage);

      List<double> probabilities = await _classifyImage(tempImage);

      // Encontrar el tag con la mayor probabilidad
      Map<String, double> tagProbabilities = _mapTagsToProbabilities(probabilities);
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
  Map<String, double> _mapTagsToProbabilities(List<double> probabilities) {
    Map<String, double> tagProbabilities = {};

    for (int i = 0; i < _labels.length; i++) {
      if (i < probabilities.length) {
        tagProbabilities[_labels[i]] = probabilities[i];
      }
    }

    return tagProbabilities;
  }
}
