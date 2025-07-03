import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'home_page.dart';

void main() {
  runApp(const WeatherApp());
}

class WeatherApp extends StatelessWidget {
  const WeatherApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deteksi Cuaca AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
        fontFamily: 'Roboto',
      ),
      debugShowCheckedModeBanner: false,
      home: const HomePage(), // Ubah ke HomePage sebagai initial route
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  File? _imageFile;
  String? _predictionLabel;
  bool _isLoading = false;
  String? _debugInfo;

  final ImagePicker _picker = ImagePicker();
  Interpreter? _interpreter;
  List<String> _labels = [];
  final int _inputSize = 224;
  bool _modelLoaded = false;

  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  // Peta terjemahan kondisi cuaca
  final Map<String, String> _weatherTranslations = {
    'dew': 'Embun',
    'fog': 'Kabut',
    'smog': 'Kabut Asap',
    'frost': 'Embun Beku',
    'glaze': 'Hujan Es Tipis',
    'hail': 'Hujan Es',
    'lightning': 'Petir',
    'rainbow': 'Pelangi',
    'rain': 'Hujan',
    'rime': 'Embun Beku Putih',
    'sandstorm': 'Badai Pasir',
    'snow': 'Salju',
    'cloudy': 'Berawan',
    'sunny': 'Cerah',
    'hot': 'Panas',
    'windy': 'Berangin',
    'storm': 'Badai',
    'drizzle': 'Gerimis',
    'mist': 'Kabut Tipis',
    'clear': 'Cerah',
    'overcast': 'Mendung',
  };

  @override
  void initState() {
    super.initState();
    _initializeModel();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.elasticOut),
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _initializeModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/model_unquant.tflite',
        options: InterpreterOptions()..threads = 4,
      );

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      if (inputTensor.shape.length != 4 ||
          inputTensor.shape[1] != _inputSize ||
          inputTensor.shape[2] != _inputSize) {
        throw Exception("Bentuk input model tidak sesuai");
      }

      final labelData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelData.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();

      if (_labels.isEmpty) throw Exception("Label kosong");
      if (outputTensor.shape.last != _labels.length) {
        throw Exception("Jumlah label tidak sesuai dengan output model");
      }

      setState(() {
        _modelLoaded = true;
        _debugInfo = 'Model siap (${_labels.length} label)';
      });
    } catch (e) {
      setState(() {
        _debugInfo = 'Error: ${e.toString().split(':').first}';
        _predictionLabel = "Gagal memuat model";
      });
    }
  }

  Future<void> _getImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(source: source);
      if (pickedFile != null) {
        setState(() {
          _imageFile = File(pickedFile.path);
          _predictionLabel = null;
        });
      }
    } catch (e) {
      setState(() {
        _predictionLabel = "Gagal mengambil gambar";
      });
    }
  }

  String _cleanAndTranslatePredictionLabel(String rawLabel) {
    // Bersihkan label dan cari terjemahan yang sesuai
    String cleaned = rawLabel.replaceAll(RegExp(r'[^a-zA-Z]'), '').toLowerCase().trim();
    
    // Cari terjemahan yang tepat
    if (_weatherTranslations.containsKey(cleaned)) {
      return _weatherTranslations[cleaned]!;
    }
    
    // Cari partial match untuk kondisi gabungan
    for (final key in _weatherTranslations.keys) {
      if (cleaned.contains(key)) {
        return _weatherTranslations[key]!;
      }
    }
    
    // Jika tidak ditemukan, kembalikan dalam format yang rapi
    if (cleaned.isNotEmpty) {
      return cleaned[0].toUpperCase() + cleaned.substring(1);
    }
    
    return rawLabel;
  }

  Future<void> detectWeather() async {
    if (!_modelLoaded || _imageFile == null) {
      setState(() {
        _predictionLabel = "Model atau gambar belum siap";
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _predictionLabel = null;
    });

    try {
      final imageBytes = await _imageFile!.readAsBytes();
      final originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) throw Exception("Gagal memproses gambar");

      final processedImage = _preprocessImage(originalImage);
      final results = await _runInference(processedImage);
      final maxIndex = results.indexOf(results.reduce((a, b) => a > b ? a : b));

      setState(() {
        _predictionLabel = _cleanAndTranslatePredictionLabel(_labels[maxIndex]);
        _isLoading = false;
      });

      _animationController.reset();
      _animationController.forward();
    } catch (e) {
      debugPrint("Error Deteksi: $e");
      setState(() {
        _predictionLabel = "Gagal mendeteksi: ${e.toString().split(':').first}";
        _isLoading = false;
      });
    }
  }

  Float32List _preprocessImage(img.Image image) {
    final resizedImage = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
    );
    final inputBuffer = Float32List(1 * _inputSize * _inputSize * 3);
    var bufferIndex = 0;

    for (var y = 0; y < _inputSize; y++) {
      for (var x = 0; x < _inputSize; x++) {
        final pixel = resizedImage.getPixel(x, y);
        inputBuffer[bufferIndex++] = pixel.r / 255.0;
        inputBuffer[bufferIndex++] = pixel.g / 255.0;
        inputBuffer[bufferIndex++] = pixel.b / 255.0;
      }
    }
    return inputBuffer;
  }

  Future<Float32List> _runInference(Float32List input) async {
    final output = Float32List(_labels.length);
    try {
      _interpreter!.run(input.buffer, output.buffer);
    } catch (e) {
      debugPrint("Error Inference: $e");
      throw Exception("Gagal menjalankan model");
    }
    return output;
  }

  Widget _getWeatherIcon(String? condition) {
    if (condition == null) {
      return Icon(Icons.help_outline, size: 100, color: Colors.grey);
    }

    // Normalisasi kondisi untuk pencocokan
    final lowerCondition = condition.toLowerCase();

    // Peta kondisi ke ikon Material
    final iconMap = {
      'embun': Icons.water_drop,
      'kabut': Icons.cloud,
      'kabut asap': Icons.smoke_free,
      'embun beku': Icons.ac_unit,
      'hujan es tipis': Icons.ac_unit,
      'hujan es': Icons.ac_unit,
      'petir': Icons.flash_on,
      'pelangi': Icons.filter_vintage,
      'hujan': Icons.beach_access,
      'badai pasir': Icons.storm,
      'salju': Icons.ac_unit,
      'berawan': Icons.cloud,
      'cerah': Icons.wb_sunny,
      'panas': Icons.wb_sunny,
      'berangin': Icons.air,
      'badai': Icons.thunderstorm,
      'gerimis': Icons.grain,
      'kabut tipis': Icons.cloud,
      'mendung': Icons.cloud_queue,
    };

    // Cari ikon yang sesuai
    IconData icon = Icons.wb_sunny; // default
    Color iconColor = Colors.orange;

    for (final key in iconMap.keys) {
      if (lowerCondition.contains(key.toLowerCase())) {
        icon = iconMap[key]!;
        break;
      }
    }

    // Sesuaikan warna berdasarkan kondisi cuaca
    if (lowerCondition.contains('hujan') || lowerCondition.contains('gerimis')) {
      iconColor = Colors.blue;
    } else if (lowerCondition.contains('es') || lowerCondition.contains('salju') || lowerCondition.contains('beku')) {
      iconColor = Colors.lightBlue;
    } else if (lowerCondition.contains('cerah') || lowerCondition.contains('panas')) {
      iconColor = Colors.orange;
    } else if (lowerCondition.contains('badai') || lowerCondition.contains('petir')) {
      iconColor = Colors.deepPurple;
    } else if (lowerCondition.contains('kabut') || lowerCondition.contains('mendung')) {
      iconColor = Colors.grey;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: iconColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: iconColor.withOpacity(0.3),
          width: 2,
        ),
      ),
      child: Icon(
        icon,
        size: 100,
        color: iconColor,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pushReplacement(
            MaterialPageRoute(builder: (context) => const HomePage()),
          ),
        ),
        title: Text(
          'Deteksi Cuaca',
          style: TextStyle(color: Colors.white), // Style moved inside Text widget
        ),
        backgroundColor: Colors.blue[800],
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.blue[400]!,
              Colors.blue[100]!,
              Colors.white,
            ],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                const SizedBox(height: 20),
                
                // Header
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.blue[800]!,
                        Colors.blue[600]!,
                      ],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.blue.withOpacity(0.4),
                        spreadRadius: 2,
                        blurRadius: 15,
                        offset: const Offset(0, 6),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Icon(
                          Icons.wb_sunny,
                          size: 50,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        "Deteksi Cuaca AI",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                          letterSpacing: 0.5,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        "Unggah gambar untuk mendeteksi kondisi cuaca",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.white.withOpacity(0.9),
                          height: 1.3,
                        ),
                      ),
                    ],
                  ),
                ),
                
                const SizedBox(height: 32),
                
                // Tombol Kamera & Galeri
                Row(
                  children: [
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: [Colors.blue[700]!, Colors.blue[800]!],
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.blue.withOpacity(0.3),
                              spreadRadius: 1,
                              blurRadius: 8,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ElevatedButton.icon(
                          onPressed: () => _getImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt, size: 24, color: Colors.white),
                          label: const Text(
                            "Kamera",
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                            ),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            padding: const EdgeInsets.symmetric(vertical: 18),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          gradient: LinearGradient(
                            colors: [Colors.blue[700]!, Colors.blue[800]!],
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.blue.withOpacity(0.3),
                              spreadRadius: 1,
                              blurRadius: 8,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: ElevatedButton.icon(
                          onPressed: () => _getImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library, size: 24, color: Colors.white),
                          label: const Text(
                            "Galeri",
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                            ),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            padding: const EdgeInsets.symmetric(vertical: 18),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                
                const SizedBox(height: 32),
                
                // Preview Gambar
                Container(
                  width: double.infinity,
                  height: 280,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: Colors.blue.withOpacity(0.3),
                      width: 3,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.3),
                        spreadRadius: 2,
                        blurRadius: 20,
                        offset: const Offset(0, 10),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(21),
                    child: _imageFile != null
                        ? Image.file(
                            _imageFile!,
                            fit: BoxFit.cover,
                          )
                        : Container(
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [Colors.grey[50]!, Colors.grey[100]!],
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                              ),
                            ),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(20),
                                  decoration: BoxDecoration(
                                    color: Colors.blue.withOpacity(0.1),
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: Icon(
                                    Icons.add_photo_alternate_outlined,
                                    size: 80,
                                    color: Colors.blue[400],
                                  ),
                                ),
                                const SizedBox(height: 16),
                                Text(
                                  "Pilih gambar untuk dianalisis",
                                  style: TextStyle(
                                    color: Colors.grey[600],
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ],
                            ),
                          ),
                  ),
                ),
                
                const SizedBox(height: 32),
                
                // Tombol Deteksi
                Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    gradient: LinearGradient(
                      colors: [Colors.blue[700]!, Colors.lightBlue[600]!],
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.blue.withOpacity(0.4),
                        spreadRadius: 1,
                        blurRadius: 12,
                        offset: const Offset(0, 6),
                      ),
                    ],
                  ),
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : detectWeather,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      foregroundColor: Colors.white,
                      shadowColor: Colors.transparent,
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),
                    child: _isLoading
                        ? Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const SizedBox(
                                width: 24,
                                height: 24,
                                child: CircularProgressIndicator(
                                  color: Colors.white,
                                  strokeWidth: 3,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Text(
                                "Menganalisis...",
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white.withOpacity(0.9),
                                ),
                              ),
                            ],
                          )
                        : const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.psychology, size: 26, color: Colors.white),
                              SizedBox(width: 10),
                              Text(
                                "Deteksi Cuaca",
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                  ),
                ),
                
                const SizedBox(height: 32),
                
                // Hasil Deteksi
                if (_predictionLabel != null)
                  AnimatedBuilder(
                    animation: _animationController,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _scaleAnimation.value,
                        child: Opacity(
                          opacity: _fadeAnimation.value,
                          child: Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(28),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.white,
                                  Colors.lightBlue[50]!,
                                ],
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                              ),
                              borderRadius: BorderRadius.circular(24),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.blue.withOpacity(0.15),
                                  spreadRadius: 3,
                                  blurRadius: 20,
                                  offset: const Offset(0, 10),
                                ),
                              ],
                            ),
                            child: Column(
                              children: [
                                Text(
                                  "Hasil Deteksi",
                                  style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.grey[700],
                                  ),
                                ),
                                const SizedBox(height: 24),
                                Column(
                                  children: [
                                    _getWeatherIcon(_predictionLabel),
                                    const SizedBox(height: 20),
                                    Text(
                                      _predictionLabel!.toUpperCase(),
                                      style: const TextStyle(
                                        fontSize: 26,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.blue,
                                        letterSpacing: 1.2,
                                      ),
                                    ),
                                    const SizedBox(height: 12),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 16,
                                        vertical: 8,
                                      ),
                                      decoration: BoxDecoration(
                                        color: Colors.lightBlue.withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(16),
                                        border: Border.all(
                                          color: Colors.lightBlue.withOpacity(0.3),
                                          width: 1.5,
                                        ),
                                      ),
                                      child: Row(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          Icon(
                                            Icons.verified,
                                            size: 16,
                                            color: Colors.blue[600],
                                          ),
                                          const SizedBox(width: 8),
                                          Text(
                                            "Terdeteksi",
                                            style: TextStyle(
                                              fontSize: 14,
                                              fontWeight: FontWeight.w600,
                                              color: Colors.blue[700],
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                
                // Informasi Debug
                if (_debugInfo != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 24),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.grey[50],
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.grey[200]!),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            Icons.info_outline,
                            size: 18,
                            color: Colors.grey[600],
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              _debugInfo!,
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 13,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                
                const SizedBox(height: 32),
              ],
            ),
          ),
        ),
      ),
    );
  }
}