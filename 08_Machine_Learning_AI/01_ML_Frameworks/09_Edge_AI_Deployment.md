# Edge AI Deployment

*Duration: 2-3 weeks*

Edge AI deployment involves running machine learning models on resource-constrained devices like mobile phones, IoT devices, microcontrollers, and embedded systems. This approach brings AI inference closer to data sources, reducing latency, improving privacy, and enabling offline operation.

## Why Edge AI?

### Key Benefits
- **Low Latency**: No network round-trip delays
- **Privacy**: Data stays on device
- **Offline Operation**: Works without internet connectivity
- **Reduced Bandwidth**: No need to send raw data to cloud
- **Cost Efficiency**: Lower cloud computing costs
- **Real-time Processing**: Critical for autonomous systems

### Edge Deployment Challenges
- **Limited Resources**: CPU, memory, storage, power constraints
- **Model Size**: Need compressed models
- **Inference Speed**: Real-time requirements
- **Accuracy Trade-offs**: Smaller models may be less accurate
- **Hardware Diversity**: Different architectures and capabilities

## TensorFlow Lite

TensorFlow Lite is a lightweight solution for mobile and embedded devices, designed to enable on-device machine learning inference with low latency and small binary size.

### Model Conversion

Converting TensorFlow models to TensorFlow Lite format involves optimization and quantization to reduce model size and improve inference speed.

#### Basic Conversion
```python
import tensorflow as tf

# Method 1: Convert from SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model('path/to/saved_model')
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Method 2: Convert from Keras model
model = tf.keras.models.load_model('my_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Method 3: Convert from concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
```

#### Advanced Conversion with Optimizations
```python
import tensorflow as tf
import numpy as np

def create_representative_dataset():
    """Create representative dataset for quantization calibration"""
    # Load your validation/calibration data
    for _ in range(100):  # Use 100 samples for calibration
        # Replace with your actual data loading logic
        data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [data]

# Load your trained model
model = tf.keras.models.load_model('trained_model.h5')

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Post-training quantization options
# 1. Dynamic range quantization (weights only)
converter.target_spec.supported_types = [tf.float16]

# 2. Full integer quantization
converter.representative_dataset = create_representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
try:
    tflite_quantized_model = converter.convert()
    print("Model converted successfully!")
    
    # Save the optimized model
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_quantized_model)
        
    # Check model size reduction
    original_size = len(tflite_model)
    quantized_size = len(tflite_quantized_model)
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size) * 100:.1f}%")
    
except Exception as e:
    print(f"Conversion failed: {e}")
```

### Interpreter API

The TensorFlow Lite Interpreter provides a high-level API for running inference on converted models.

#### Python Interpreter Usage
```python
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

class TFLiteInference:
    def __init__(self, model_path, num_threads=1):
        """Initialize TensorFlow Lite interpreter"""
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Input type: {self.input_details[0]['dtype']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
        
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model input"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Normalize to [-1, 1] or [0, 1] depending on your model
        input_data = (input_data / 127.5) - 1.0  # For MobileNet
        
        return input_data
    
    def run_inference(self, input_data):
        """Run inference on input data"""
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            input_data
        )
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output_data, inference_time
    
    def predict_image(self, image_path, class_labels=None):
        """Complete image classification pipeline"""
        # Preprocess
        input_data = self.preprocess_image(image_path)
        
        # Run inference
        predictions, inference_time = self.run_inference(input_data)
        
        # Post-process results
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'all_predictions': predictions[0].tolist()
        }
        
        if class_labels:
            result['class_name'] = class_labels[predicted_class]
            
        return result

# Usage example
if __name__ == "__main__":
    # Load class labels (ImageNet example)
    class_labels = []
    with open('imagenet_labels.txt', 'r') as f:
        class_labels = [line.strip() for line in f.readlines()]
    
    # Initialize model
    model = TFLiteInference('quantized_model.tflite', num_threads=4)
    
    # Run inference
    result = model.predict_image('test_image.jpg', class_labels)
    
    print(f"Predicted: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Inference time: {result['inference_time_ms']:.1f}ms")
```

### C++ and Java Interfaces

#### C++ Interface for High Performance
```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>
#include <chrono>

class TFLiteClassifier {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
public:
    bool LoadModel(const std::string& model_path) {
        // Load model
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            return false;
        }
        
        // Build interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        
        if (!interpreter) {
            std::cerr << "Failed to create interpreter" << std::endl;
            return false;
        }
        
        // Allocate tensors
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors" << std::endl;
            return false;
        }
        
        // Set number of threads
        interpreter->SetNumThreads(4);
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Input tensor count: " << interpreter->inputs().size() << std::endl;
        std::cout << "Output tensor count: " << interpreter->outputs().size() << std::endl;
        
        return true;
    }
    
    std::vector<float> Classify(const cv::Mat& image) {
        // Get input tensor
        int input_tensor_idx = interpreter->inputs()[0];
        TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_idx);
        
        // Preprocess image
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(224, 224));
        resized_image.convertTo(resized_image, CV_32F, 1.0/127.5, -1.0);
        
        // Copy data to input tensor
        float* input_data = interpreter->typed_input_tensor<float>(0);
        memcpy(input_data, resized_image.data, 
               resized_image.total() * resized_image.elemSize());
        
        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return {};
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
        
        // Get output
        float* output_data = interpreter->typed_output_tensor<float>(0);
        int output_size = interpreter->output_tensor(0)->bytes / sizeof(float);
        
        return std::vector<float>(output_data, output_data + output_size);
    }
};

// Usage example
int main() {
    TFLiteClassifier classifier;
    
    if (!classifier.LoadModel("model.tflite")) {
        return -1;
    }
    
    cv::Mat image = cv::imread("test_image.jpg");
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }
    
    auto predictions = classifier.Classify(image);
    
    // Find max prediction
    auto max_iter = std::max_element(predictions.begin(), predictions.end());
    int predicted_class = std::distance(predictions.begin(), max_iter);
    float confidence = *max_iter;
    
    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Confidence: " << confidence << std::endl;
    
    return 0;
}
```

#### Android Java Interface
```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import android.graphics.Bitmap;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;

public class TFLiteImageClassifier {
    private Interpreter tflite;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private ByteBuffer outputBuffer;
    
    public TFLiteImageClassifier(MappedByteBuffer modelBuffer) {
        // Initialize interpreter
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        
        // Use GPU delegate if available
        /*
        GpuDelegate delegate = new GpuDelegate();
        options.addDelegate(delegate);
        */
        
        tflite = new Interpreter(modelBuffer, options);
        
        // Setup image processor
        imageProcessor = new ImageProcessor.Builder()
            .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(new Rot90Op()) // if needed
            .build();
            
        // Initialize input image buffer
        inputImageBuffer = new TensorImage(DataType.FLOAT32);
        
        // Allocate output buffer
        int[] outputShape = tflite.getOutputTensor(0).shape();
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }
        outputBuffer = ByteBuffer.allocateDirect(outputSize * 4); // 4 bytes per float
        outputBuffer.order(ByteOrder.nativeOrder());
    }
    
    public ClassificationResult classify(Bitmap bitmap) {
        // Preprocess image
        inputImageBuffer = imageProcessor.process(inputImageBuffer.load(bitmap));
        
        // Run inference
        long startTime = System.currentTimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputBuffer);
        long inferenceTime = System.currentTimeMillis() - startTime;
        
        // Parse output
        outputBuffer.rewind();
        float[] probabilities = new float[outputBuffer.remaining() / 4];
        outputBuffer.asFloatBuffer().get(probabilities);
        
        // Find top prediction
        int maxIndex = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        return new ClassificationResult(maxIndex, maxProb, inferenceTime, probabilities);
    }
    
    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
    
    public static class ClassificationResult {
        public final int classIndex;
        public final float confidence;
        public final long inferenceTimeMs;
        public final float[] allProbabilities;
        
        public ClassificationResult(int classIndex, float confidence, 
                                  long inferenceTimeMs, float[] allProbabilities) {
            this.classIndex = classIndex;
            this.confidence = confidence;
            this.inferenceTimeMs = inferenceTimeMs;
            this.allProbabilities = allProbabilities;
        }
    }
}
```

### Microcontroller Deployment

TensorFlow Lite Micro enables running ML models on microcontrollers with as little as a few KB of memory.

#### Arduino Example
```cpp
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model data (converted to C array)
#include "model_data.h"

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  constexpr int kTensorArenaSize = 60 * 1024; // Adjust based on your model
  uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(9600);
  
  // Initialize TensorFlow Lite
  tflite::InitializeTarget();
  
  // Load model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }
  
  // Set up resolver
  static tflite::AllOpsResolver resolver;
  
  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  
  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("TensorFlow Lite Micro initialized!");
  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Read sensor data (example: accelerometer)
  float accel_x = readAccelX();
  float accel_y = readAccelY();
  float accel_z = readAccelZ();
  
  // Prepare input data
  input->data.f[0] = accel_x;
  input->data.f[1] = accel_y;
  input->data.f[2] = accel_z;
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }
  
  // Get result
  float gesture_probability = output->data.f[0];
  
  if (gesture_probability > 0.8) {
    Serial.println("Gesture detected!");
  }
  
  delay(100);
}

// Dummy sensor reading functions
float readAccelX() { return analogRead(A0) / 1023.0 * 2.0 - 1.0; }
float readAccelY() { return analogRead(A1) / 1023.0 * 2.0 - 1.0; }
float readAccelZ() { return analogRead(A2) / 1023.0 * 2.0 - 1.0; }
```

### Delegates for Acceleration

TensorFlow Lite supports hardware acceleration through delegates.

#### GPU Delegate Example
```python
import tensorflow as tf

# For mobile devices
try:
    # Create GPU delegate
    gpu_delegate = tf.lite.experimental.load_delegate('libdelegate.so')
    
    interpreter = tf.lite.Interpreter(
        model_path="model.tflite",
        experimental_delegates=[gpu_delegate]
    )
    
    print("GPU acceleration enabled")
except:
    # Fallback to CPU
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    print("Using CPU inference")

# Alternative: Use built-in GPU delegate (Android/iOS)
try:
    interpreter = tf.lite.Interpreter(
        model_path="model.tflite",
        experimental_delegates=[tf.lite.experimental.load_delegate('gpu')]
    )
except:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
```

## PyTorch Mobile

PyTorch Mobile enables deploying PyTorch models on iOS and Android devices with optimized performance and reduced memory footprint.

### Model Optimization

Before deploying to mobile, PyTorch models need to be optimized and converted to the mobile-friendly format.

#### Converting Models to TorchScript
```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile

class MobileNetV3Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(960, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Load your trained model
model = MobileNetV3Classifier(num_classes=10)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Convert to TorchScript
example_input = torch.randn(1, 3, 224, 224)

# Method 1: Tracing (recommended for most cases)
traced_model = torch.jit.trace(model, example_input)

# Method 2: Scripting (for models with control flow)
scripted_model = torch.jit.script(model)

# Optimize for mobile
mobile_model = optimize_for_mobile(traced_model)

# Save mobile-optimized model
mobile_model.save('mobile_model.pt')

print("Model converted and optimized for mobile deployment")
```

#### Advanced Optimization Techniques
```python
import torch
import torch.quantization as quantization
from torch.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def optimize_model_for_mobile(model, example_input, use_quantization=True):
    """Comprehensive mobile optimization pipeline"""
    
    # Step 1: Convert to TorchScript
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    
    # Step 2: Apply graph optimizations
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Step 3: Quantization (optional but recommended)
    if use_quantization:
        # Post-training quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        # Convert quantized model to TorchScript
        quantized_model.eval()
        traced_quantized = torch.jit.trace(quantized_model, example_input)
        optimized_model = torch.jit.optimize_for_inference(traced_quantized)
    
    # Step 4: Mobile-specific optimizations
    mobile_model = optimize_for_mobile(optimized_model)
    
    return mobile_model

# Usage example
model = MobileNetV3Classifier(num_classes=10)
model.load_state_dict(torch.load('trained_model.pth'))
example_input = torch.randn(1, 3, 224, 224)

# Original model size
original_size = len(torch.jit.trace(model, example_input).save_to_buffer())

# Optimized model
optimized_model = optimize_model_for_mobile(model, example_input)
optimized_model.save('optimized_mobile_model.pt')

# Optimized model size
optimized_size = len(optimized_model.save_to_buffer())

print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
print(f"Optimized model size: {optimized_size / 1024 / 1024:.2f} MB")
print(f"Size reduction: {((original_size - optimized_size) / original_size) * 100:.1f}%")
```

### Mobile Interpreter

PyTorch Mobile provides lightweight interpreters for running models on mobile devices.

#### Python Mobile Interpreter
```python
import torch
from torchvision import transforms
from PIL import Image
import time
import numpy as np

class PyTorchMobileInference:
    def __init__(self, model_path):
        """Initialize PyTorch Mobile interpreter"""
        self.model = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        
        # Define preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model loaded: {model_path}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor
    
    def run_inference(self, input_tensor):
        """Run inference with timing"""
        with torch.no_grad():
            start_time = time.time()
            output = self.model(input_tensor)
            inference_time = (time.time() - start_time) * 1000
            
        return output, inference_time
    
    def predict_image(self, image_path, class_labels=None):
        """Complete image classification pipeline"""
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference
        output, inference_time = self.run_inference(input_tensor)
        
        # Post-process
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'top5_predictions': []
        }
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        for i in range(5):
            class_info = {
                'class_id': top5_indices[i].item(),
                'probability': top5_prob[i].item()
            }
            if class_labels:
                class_info['class_name'] = class_labels[top5_indices[i].item()]
            result['top5_predictions'].append(class_info)
        
        return result

    def benchmark_model(self, input_shape=(1, 3, 224, 224), num_runs=100):
        """Benchmark model performance"""
        dummy_input = torch.randn(input_shape)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            times.append((time.time() - start_time) * 1000)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }

# Usage example
if __name__ == "__main__":
    # Load model
    inference = PyTorchMobileInference('optimized_mobile_model.pt')
    
    # Run prediction
    result = inference.predict_image('test_image.jpg')
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Inference time: {result['inference_time_ms']:.1f}ms")
    
    # Benchmark performance
    benchmark = inference.benchmark_model()
    print(f"Average inference time: {benchmark['mean_time_ms']:.1f}ms")
    print(f"FPS: {benchmark['fps']:.1f}")
```

### iOS and Android Deployment

#### iOS Deployment (Swift)
```swift
import UIKit
import Vision
import CoreML

class PyTorchMobileClassifier {
    private var model: VNCoreMLModel?
    
    init(modelName: String) {
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "pt") else {
            fatalError("Model file not found")
        }
        
        do {
            // Load PyTorch Mobile model
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
        } catch {
            print("Error loading model: \(error)")
        }
    }
    
    func classifyImage(_ image: UIImage, completion: @escaping (String?, Float) -> Void) {
        guard let model = model else {
            completion(nil, 0.0)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNClassificationObservation],
                  let topResult = results.first else {
                completion(nil, 0.0)
                return
            }
            
            DispatchQueue.main.async {
                completion(topResult.identifier, topResult.confidence)
            }
        }
        
        guard let ciImage = CIImage(image: image) else {
            completion(nil, 0.0)
            return
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        
        do {
            try handler.perform([request])
        } catch {
            print("Error performing classification: \(error)")
            completion(nil, 0.0)
        }
    }
}

// Usage in ViewController
class ViewController: UIViewController {
    private let classifier = PyTorchMobileClassifier(modelName: "mobile_model")
    
    @IBAction func classifyButtonTapped(_ sender: UIButton) {
        guard let image = imageView.image else { return }
        
        classifier.classifyImage(image) { [weak self] label, confidence in
            self?.resultLabel.text = "\(label ?? "Unknown") (\(Int(confidence * 100))%)"
        }
    }
}
```

#### Android Deployment (Kotlin)
```kotlin
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

class PyTorchMobileClassifier(private val context: Context) {
    private var module: Module? = null
    private val classes = arrayOf("class1", "class2", "class3") // Your class names
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            // Copy model from assets to internal storage
            val modelFile = File(context.filesDir, "mobile_model.pt")
            if (!modelFile.exists()) {
                copyAssetToFile("mobile_model.pt", modelFile)
            }
            
            module = Module.load(modelFile.absolutePath)
            Log.d("PyTorchMobile", "Model loaded successfully")
        } catch (e: IOException) {
            Log.e("PyTorchMobile", "Error loading model", e)
        }
    }
    
    private fun copyAssetToFile(assetName: String, outFile: File) {
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(outFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
    }
    
    fun classifyImage(bitmap: Bitmap): ClassificationResult? {
        val module = this.module ?: return null
        
        try {
            // Preprocess image
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                floatArrayOf(0.485f, 0.456f, 0.406f), // ImageNet mean
                floatArrayOf(0.229f, 0.224f, 0.225f)  // ImageNet std
            )
            
            // Run inference
            val startTime = System.currentTimeMillis()
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val inferenceTime = System.currentTimeMillis() - startTime
            
            // Get output scores
            val scores = outputTensor.dataAsFloatArray
            
            // Find max score and index
            var maxScore = -Float.MAX_VALUE
            var maxScoreIdx = -1
            for (i in scores.indices) {
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }
            
            // Apply softmax to get probabilities
            val exp = scores.map { kotlin.math.exp(it.toDouble()).toFloat() }
            val sum = exp.sum()
            val probabilities = exp.map { it / sum }
            
            return ClassificationResult(
                classIndex = maxScoreIdx,
                className = if (maxScoreIdx < classes.size) classes[maxScoreIdx] else "Unknown",
                confidence = probabilities[maxScoreIdx],
                inferenceTimeMs = inferenceTime,
                allProbabilities = probabilities.toFloatArray()
            )
            
        } catch (e: Exception) {
            Log.e("PyTorchMobile", "Error during inference", e)
            return null
        }
    }
    
    data class ClassificationResult(
        val classIndex: Int,
        val className: String,
        val confidence: Float,
        val inferenceTimeMs: Long,
        val allProbabilities: FloatArray
    )
}

// Usage in Activity/Fragment
class MainActivity : AppCompatActivity() {
    private lateinit var classifier: PyTorchMobileClassifier
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        classifier = PyTorchMobileClassifier(this)
    }
    
    private fun classifyImage(bitmap: Bitmap) {
        val result = classifier.classifyImage(bitmap)
        result?.let {
            resultTextView.text = "${it.className} (${(it.confidence * 100).toInt()}%)"
            timeTextView.text = "Inference: ${it.inferenceTimeMs}ms"
        }
    }
}
```

### Memory Management

Effective memory management is crucial for mobile deployment to prevent crashes and ensure smooth performance.

#### Memory Optimization Strategies
```python
import torch
import gc
from typing import Optional

class MemoryEfficientInference:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model: Optional[torch.jit.ScriptModule] = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model with memory optimization"""
        try:
            # Load model to CPU first to save GPU memory
            self.model = torch.jit.load(model_path, map_location='cpu')
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(self.device)
                # Clear cache after loading
                torch.cuda.empty_cache()
            
            self.model.eval()
            print(f"Model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def run_inference_with_memory_management(self, input_tensor: torch.Tensor):
        """Run inference with careful memory management"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Move input to device
            input_tensor = input_tensor.to(self.device)
            
            # Run inference in no_grad context to save memory
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Move output back to CPU to free device memory
            if self.device == 'cuda':
                output = output.cpu()
                torch.cuda.empty_cache()
            
            return output
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Handle OOM gracefully
                print("Out of memory error, clearing cache and retrying...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Retry with CPU
                input_tensor = input_tensor.cpu()
                with torch.no_grad():
                    output = self.model.cpu()(input_tensor)
                return output
            else:
                raise e
    
    def batch_inference_memory_safe(self, inputs: list, batch_size: int = 1):
        """Process multiple inputs with memory-safe batching"""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Stack inputs into batch
            batch_tensor = torch.stack(batch)
            
            # Run inference
            batch_output = self.run_inference_with_memory_management(batch_tensor)
            
            # Append results
            for j in range(batch_output.size(0)):
                results.append(batch_output[j])
            
            # Clear intermediate tensors
            del batch_tensor, batch_output
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def get_memory_usage(self):
        """Monitor memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2     # MB
            return {
                'gpu_allocated_mb': allocated,
                'gpu_cached_mb': cached,
                'gpu_free_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2 - cached
            }
        else:
            import psutil
            return {
                'cpu_memory_percent': psutil.virtual_memory().percent,
                'available_gb': psutil.virtual_memory().available / 1024**3
            }

# Usage example with memory monitoring
def memory_efficient_classification():
    classifier = MemoryEfficientInference('mobile_model.pt', device='cuda')
    
    # Monitor memory before inference
    print("Memory before inference:", classifier.get_memory_usage())
    
    # Create dummy batch
    batch = [torch.randn(3, 224, 224) for _ in range(10)]
    
    # Process batch safely
    results = classifier.batch_inference_memory_safe(batch, batch_size=2)
    
    # Monitor memory after inference
    print("Memory after inference:", classifier.get_memory_usage())
    
    return results
```

## ONNX Runtime for Edge

ONNX Runtime provides a high-performance inference engine optimized for edge deployment with minimal footprint and maximum performance across different hardware platforms.

### Minimal Build

ONNX Runtime supports custom builds that include only the operators and execution providers needed for your specific models, significantly reducing binary size.

#### Building Custom ONNX Runtime
```bash
# Clone ONNX Runtime repository
git clone --recursive https://github.com/Microsoft/onnxruntime
cd onnxruntime

# Build minimal version with only CPU execution provider
./build.sh --config MinSizeRel --build_shared_lib --minimal_build extended \
           --disable_rtti --disable_exceptions --enable_reduced_operator_type_support

# Build with specific operators only
./build.sh --config MinSizeRel --minimal_build extended \
           --include_ops_by_config my_required_ops.config

# Example my_required_ops.config content:
# ai.onnx;11;Add,Conv,Relu,GlobalAveragePool,Gemm
# ai.onnx;13;Sigmoid
```

#### Python Minimal ONNX Runtime Usage
```python
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple
import time

class ONNXEdgeInference:
    def __init__(self, model_path: str, providers: List[str] = None):
        """Initialize ONNX Runtime for edge inference"""
        
        # Default to CPU provider for edge devices
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Configure session options for edge deployment
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Set thread pool size for edge devices
        sess_options.intra_op_num_threads = 1  # Conservative for edge
        sess_options.inter_op_num_threads = 1
        
        # Disable profiling to save memory
        sess_options.enable_profiling = False
        
        try:
            self.session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            # Get model metadata
            self.input_details = self.session.get_inputs()
            self.output_details = self.session.get_outputs()
            
            print(f"ONNX model loaded: {model_path}")
            print(f"Execution providers: {self.session.get_providers()}")
            self.print_model_info()
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise
    
    def print_model_info(self):
        """Print model input/output information"""
        print("\nModel Information:")
        print("Inputs:")
        for input_detail in self.input_details:
            print(f"  {input_detail.name}: {input_detail.shape} ({input_detail.type})")
        
        print("Outputs:")
        for output_detail in self.output_details:
            print(f"  {output_detail.name}: {output_detail.shape} ({output_detail.type})")
    
    def preprocess_input(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess input data for the model"""
        input_name = self.input_details[0].name
        
        # Ensure correct data type
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        # Ensure correct shape (add batch dimension if needed)
        expected_shape = self.input_details[0].shape
        if len(data.shape) == len(expected_shape) - 1:
            data = np.expand_dims(data, axis=0)
        
        return {input_name: data}
    
    def run_inference(self, input_data: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Run inference and return results with timing"""
        start_time = time.time()
        
        try:
            outputs = self.session.run(None, input_data)
            inference_time = (time.time() - start_time) * 1000
            return outputs, inference_time
            
        except Exception as e:
            print(f"Inference error: {e}")
            raise
    
    def benchmark_performance(self, input_shape: Tuple[int, ...], num_runs: int = 100):
        """Benchmark model performance"""
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_dict = self.preprocess_input(dummy_input)
        
        # Warmup runs
        for _ in range(10):
            self.session.run(None, input_dict)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.session.run(None, input_dict)
            times.append((time.time() - start_time) * 1000)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000 / np.mean(times)
        }

# Example usage for image classification
class ONNXImageClassifier(ONNXEdgeInference):
    def __init__(self, model_path: str, providers: List[str] = None):
        super().__init__(model_path, providers)
        
    def classify_image(self, image: np.ndarray, class_labels: List[str] = None):
        """Classify an image and return results"""
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Normalize if needed (assuming ImageNet preprocessing)
        if image.max() > 1.0:
            image = image / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        input_dict = self.preprocess_input(image)
        
        # Run inference
        outputs, inference_time = self.run_inference(input_dict)
        
        # Post-process results
        logits = outputs[0][0]  # Remove batch dimension
        probabilities = self.softmax(logits)
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        results = {
            'inference_time_ms': inference_time,
            'predictions': []
        }
        
        for i, idx in enumerate(top_indices):
            prediction = {
                'rank': i + 1,
                'class_id': int(idx),
                'probability': float(probabilities[idx])
            }
            
            if class_labels and idx < len(class_labels):
                prediction['class_name'] = class_labels[idx]
            
            results['predictions'].append(prediction)
        
        return results
    
    @staticmethod
    def softmax(x):
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

# Usage example
if __name__ == "__main__":
    # Initialize classifier
    classifier = ONNXImageClassifier('mobilenet_v2.onnx')
    
    # Load and preprocess image
    import cv2
    image = cv2.imread('test_image.jpg')
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    
    # Run classification
    results = classifier.classify_image(image)
    
    print(f"Inference time: {results['inference_time_ms']:.1f}ms")
    for pred in results['predictions'][:3]:
        print(f"Rank {pred['rank']}: Class {pred['class_id']} "
              f"({pred['probability']:.3f})")
    
    # Benchmark performance
    benchmark = classifier.benchmark_performance((1, 3, 224, 224))
    print(f"Average inference time: {benchmark['mean_time_ms']:.1f}ms")
    print(f"FPS: {benchmark['fps']:.1f}")
```

### Execution Provider Selection

ONNX Runtime supports multiple execution providers for different hardware accelerators. Choosing the right provider is crucial for edge performance.

#### Available Execution Providers
```python
import onnxruntime as ort

# Check available providers
print("Available providers:", ort.get_available_providers())

# Provider selection strategy for edge devices
def select_best_provider():
    """Select the best available execution provider"""
    available_providers = ort.get_available_providers()
    
    # Priority order for edge devices
    provider_priority = [
        'TensorrtExecutionProvider',  # NVIDIA edge devices
        'OpenVINOExecutionProvider',  # Intel edge devices
        'CoreMLExecutionProvider',    # Apple devices
        'QNNExecutionProvider',       # Qualcomm devices
        'CPUExecutionProvider'        # Fallback
    ]
    
    for provider in provider_priority:
        if provider in available_providers:
            print(f"Selected provider: {provider}")
            return [provider]
    
    return ['CPUExecutionProvider']

# Configure providers with specific options
def create_optimized_session(model_path: str):
    """Create optimized inference session for edge"""
    
    # Get best provider
    providers = select_best_provider()
    
    # Configure provider-specific options
    provider_options = {}
    
    if 'TensorrtExecutionProvider' in providers:
        provider_options = {
            'trt_max_workspace_size': 2147483648,  # 2GB
            'trt_fp16_enable': True,  # Use FP16 for speed
            'trt_int8_enable': False,  # Disable INT8 for compatibility
        }
    elif 'OpenVINOExecutionProvider' in providers:
        provider_options = {
            'device_type': 'CPU',
            'precision': 'FP16',
            'num_of_threads': 4,
        }
    elif 'CoreMLExecutionProvider' in providers:
        provider_options = {
            'useCPUOnly': False,
            'coreml_flags': 1,  # Use CPU and GPU
        }
    
    # Session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create session
    if provider_options:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[(providers[0], provider_options)]
        )
    else:
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
    
    return session
```

### Memory Planning

Efficient memory planning is critical for edge devices with limited RAM.

#### Memory Optimization Techniques
```python
import onnxruntime as ort
import numpy as np
import psutil
import gc

class MemoryOptimizedONNX:
    def __init__(self, model_path: str, max_memory_mb: int = 512):
        self.max_memory_mb = max_memory_mb
        self.session = None
        self.input_buffer = None
        self.output_buffer = None
        
        self.load_model_with_memory_limit(model_path)
    
    def load_model_with_memory_limit(self, model_path: str):
        """Load model with memory constraints"""
        
        # Configure session for memory optimization
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        sess_options.enable_cpu_mem_arena = False  # Disable memory arena for tight control
        
        # Limit memory usage
        sess_options.session_config_entries = {
            'session.memory_limit_mb': str(self.max_memory_mb)
        }
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options)
            
            # Pre-allocate input/output buffers
            self.preallocate_buffers()
            
            print(f"Model loaded with {self.max_memory_mb}MB memory limit")
            
        except Exception as e:
            print(f"Error loading model with memory limit: {e}")
            raise
    
    def preallocate_buffers(self):
        """Pre-allocate input and output buffers to avoid runtime allocation"""
        
        # Allocate input buffer
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic shapes
        input_shape = [1 if dim is None or isinstance(dim, str) else dim for dim in input_shape]
        self.input_buffer = np.zeros(input_shape, dtype=np.float32)
        
        # Allocate output buffer
        output_shape = self.session.get_outputs()[0].shape
        output_shape = [1 if dim is None or isinstance(dim, str) else dim for dim in output_shape]
        self.output_buffer = np.zeros(output_shape, dtype=np.float32)
        
        print(f"Pre-allocated buffers: input {input_shape}, output {output_shape}")
    
    def run_inference_memory_safe(self, input_data: np.ndarray):
        """Run inference with memory safety checks"""
        
        # Check available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        if available_memory < self.max_memory_mb:
            print(f"Warning: Low memory ({available_memory:.0f}MB available)")
            gc.collect()  # Force garbage collection
        
        # Reuse pre-allocated buffer
        np.copyto(self.input_buffer, input_data)
        
        # Prepare input dictionary
        input_name = self.session.get_inputs()[0].name
        input_dict = {input_name: self.input_buffer}
        
        try:
            # Run inference
            outputs = self.session.run(None, input_dict)
            
            # Copy to pre-allocated output buffer
            np.copyto(self.output_buffer, outputs[0])
            
            return self.output_buffer.copy()  # Return copy to avoid reference issues
            
        except Exception as e:
            print(f"Inference failed: {e}")
            # Clear buffers and retry
            gc.collect()
            raise
    
    def get_memory_usage(self):
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

# Example usage
def memory_constrained_inference():
    """Example of running inference on memory-constrained device"""
    
    # Initialize with 256MB memory limit
    model = MemoryOptimizedONNX('model.onnx', max_memory_mb=256)
    
    # Monitor memory before inference
    before_memory = model.get_memory_usage()
    print(f"Memory before: {before_memory['rss_mb']:.1f}MB")
    
    # Create input data
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Run inference
    result = model.run_inference_memory_safe(input_data)
    
    # Monitor memory after inference
    after_memory = model.get_memory_usage()
    print(f"Memory after: {after_memory['rss_mb']:.1f}MB")
    print(f"Memory increase: {after_memory['rss_mb'] - before_memory['rss_mb']:.1f}MB")
    
    return result
```

### Threading Models

ONNX Runtime supports different threading models optimized for various edge hardware configurations.

#### Threading Configuration
```python
import onnxruntime as ort
import threading
import time
from concurrent.futures import ThreadPoolExecutor

class ThreadOptimizedONNX:
    def __init__(self, model_path: str, thread_config: dict = None):
        """Initialize with optimized threading configuration"""
        
        if thread_config is None:
            thread_config = self.detect_optimal_threading()
        
        # Configure session options for threading
        sess_options = ort.SessionOptions()
        
        # Intra-op threading (within operations)
        sess_options.intra_op_num_threads = thread_config['intra_op_threads']
        
        # Inter-op threading (between operations)
        sess_options.inter_op_num_threads = thread_config['inter_op_threads']
        
        # Execution mode
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL if thread_config['parallel_execution'] else ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Graph optimization
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=sess_options)
        self.thread_config = thread_config
        
        print(f"Threading configuration: {thread_config}")
    
    def detect_optimal_threading(self):
        """Detect optimal threading configuration for current hardware"""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # Conservative settings for edge devices
        if cpu_count <= 2:
            return {
                'intra_op_threads': 1,
                'inter_op_threads': 1,
                'parallel_execution': False
            }
        elif cpu_count <= 4:
            return {
                'intra_op_threads': 2,
                'inter_op_threads': 1,
                'parallel_execution': True
            }
        else:
            return {
                'intra_op_threads': min(4, cpu_count // 2),
                'inter_op_threads': 2,
                'parallel_execution': True
            }
    
    def benchmark_threading_configurations(self, input_shape: tuple, configs: list):
        """Benchmark different threading configurations"""
        
        results = {}
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        input_dict = {input_name: dummy_input}
        
        for config_name, config in configs:
            print(f"Testing configuration: {config_name}")
            
            # Create new session with this configuration
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = config['intra_op_threads']
            sess_options.inter_op_num_threads = config['inter_op_threads']
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL if config['parallel_execution'] else ort.ExecutionMode.ORT_SEQUENTIAL
            
            temp_session = ort.InferenceSession(self.session._model_path, sess_options=sess_options)
            
            # Warmup
            for _ in range(5):
                temp_session.run(None, input_dict)
            
            # Benchmark
            times = []
            for _ in range(20):
                start_time = time.time()
                temp_session.run(None, input_dict)
                times.append((time.time() - start_time) * 1000)
            
            results[config_name] = {
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'config': config
            }
        
        return results
    
    def concurrent_inference(self, inputs: list, max_workers: int = None):
        """Run inference on multiple inputs concurrently"""
        
        if max_workers is None:
            max_workers = min(len(inputs), self.thread_config['inter_op_threads'])
        
        def single_inference(input_data):
            input_name = self.session.get_inputs()[0].name
            input_dict = {input_name: input_data}
            return self.session.run(None, input_dict)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_inference, inp) for inp in inputs]
            results = [future.result() for future in futures]
        
        return results

# Example usage and benchmarking
def optimize_threading_for_edge():
    """Find optimal threading configuration for edge device"""
    
    model_path = 'model.onnx'
    
    # Test different configurations
    configs = [
        ('single_thread', {'intra_op_threads': 1, 'inter_op_threads': 1, 'parallel_execution': False}),
        ('dual_thread', {'intra_op_threads': 2, 'inter_op_threads': 1, 'parallel_execution': False}),
        ('parallel_small', {'intra_op_threads': 2, 'inter_op_threads': 2, 'parallel_execution': True}),
        ('parallel_medium', {'intra_op_threads': 4, 'inter_op_threads': 2, 'parallel_execution': True}),
    ]
    
    # Create test instance
    test_model = ThreadOptimizedONNX(model_path)
    
    # Benchmark configurations
    results = test_model.benchmark_threading_configurations((1, 3, 224, 224), configs)
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['mean_time_ms'])
    
    print("\nBenchmark Results:")
    for config_name, result in results.items():
        print(f"{config_name}: {result['mean_time_ms']:.1f}ms ± {result['std_time_ms']:.1f}ms")
    
    print(f"\nBest configuration: {best_config[0]} ({best_config[1]['mean_time_ms']:.1f}ms)")
    
## Study Materials and Resources

### Essential Reading
- **Primary:** "Efficient Deep Learning" by Vivienne Sze et al. - Comprehensive guide to edge AI optimization
- **TensorFlow Lite Guide:** [Official TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- **PyTorch Mobile:** [PyTorch Mobile Documentation](https://pytorch.org/mobile/home/)
- **ONNX Runtime:** [Edge Deployment Guide](https://onnxruntime.ai/docs/tutorials/mobile/)
- **Research Papers:**
  - "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
  - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
  - "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"

### Video Tutorials and Courses
- **Google I/O Sessions:** TensorFlow Lite and Mobile AI
- **PyTorch Developer Conference:** Mobile deployment talks
- **Edge AI Coursera Specialization:** Stanford/DeepLearning.ai
- **YouTube Channels:**
  - TensorFlow (official mobile AI tutorials)
  - PyTorch (mobile deployment series)
  - Two Minute Papers (latest edge AI research)

### Practical Labs and Tutorials
- **TensorFlow Lite Codelabs:** Hands-on mobile AI development
- **PyTorch Mobile Tutorials:** iOS and Android deployment
- **Edge TPU Tutorials:** Google Coral development board
- **NVIDIA Jetson Tutorials:** GPU-accelerated edge AI
- **Qualcomm AI Engine:** Mobile SoC optimization

### Development Tools and Frameworks

#### Model Optimization Tools
```bash
# TensorFlow Model Optimization Toolkit
pip install tensorflow-model-optimization

# PyTorch Quantization
pip install torch torchvision

# ONNX Tools
pip install onnx onnxruntime onnx-simplifier

# Netron - Neural network visualizer
pip install netron

# Model compression tools
pip install neural-compressor  # Intel
pip install brevitas          # Quantization-aware training
```

#### Profiling and Benchmarking Tools
```bash
# TensorFlow Lite Benchmark Tool
# Download from TensorFlow releases

# PyTorch Mobile Benchmark
# Built into PyTorch Mobile SDK

# ONNX Runtime Performance Tools
pip install onnxruntime-tools

# Hardware-specific tools
# Intel VTune Profiler
# ARM Performance Studio
# Qualcomm Snapdragon Profiler
```

#### Mobile Development Setup
```bash
# Android Development
# Android Studio + NDK
# Flutter with TensorFlow Lite plugin

# iOS Development
# Xcode + Core ML tools
# Flutter with TensorFlow Lite iOS

# Cross-platform
# React Native with ML libraries
# Xamarin with ML.NET
```

### Hardware Platforms for Testing

#### Mobile Devices
- **High-end:** iPhone 14 Pro, Samsung Galaxy S23, Google Pixel 7 Pro
- **Mid-range:** iPhone SE, Samsung Galaxy A54, Google Pixel 6a
- **Low-end:** Budget Android devices with 3-4GB RAM

#### Edge Computing Boards
- **NVIDIA Jetson Series:** Nano, Xavier NX, AGX Orin
- **Google Coral:** Dev Board, USB Accelerator, Mini PCIe
- **Raspberry Pi:** 4B with AI HAT, CM4 modules
- **Intel NUC:** With Movidius neural compute sticks

#### Microcontrollers
- **Arduino:** Nano 33 BLE Sense, Portenta H7
- **ESP32:** AI Thinker modules, M5Stack
- **STM32:** Nucleo boards with AI packages
- **Nordic:** nRF52840 with Edge Impulse

### Practice Exercises and Challenges

#### Beginner Level
1. **Model Conversion Challenge**
   - Convert pre-trained ResNet-18 to TensorFlow Lite
   - Apply INT8 quantization
   - Compare accuracy and model size
   - Target: <5MB model, <2% accuracy drop

2. **Mobile App Integration**
   - Create simple Android app with image classification
   - Implement camera capture and preprocessing
   - Display results with confidence scores
   - Target: <100ms inference time

3. **Memory Optimization**
   - Deploy MobileNetV2 on device with 1GB RAM
   - Implement tiled inference for large images
   - Monitor memory usage throughout inference
   - Target: <200MB peak memory usage

#### Intermediate Level
4. **Multi-model Pipeline**
   - Implement object detection + classification pipeline
   - Optimize scheduling between models
   - Handle different input resolutions efficiently
   - Target: 15 FPS on mid-range mobile device

5. **Power-aware Inference**
   - Implement adaptive inference based on battery level
   - Create power profiles for different performance modes
   - Measure actual power consumption
   - Target: 50% battery life improvement in low-power mode

6. **Hardware Acceleration**
   - Deploy same model on CPU, GPU, and specialized accelerators
   - Compare performance and energy efficiency
   - Implement fallback mechanisms
   - Document optimal configuration for each platform

#### Advanced Level
7. **Custom Quantization Scheme**
   - Implement mixed-precision quantization
   - Design custom calibration dataset
   - Validate accuracy across different domains
   - Target: INT4 average precision with <3% accuracy loss

8. **Edge AI Framework**
   - Build deployment framework supporting multiple backends
   - Implement automatic optimization based on hardware detection
   - Create unified API for different model formats
   - Include monitoring and telemetry

9. **Real-time Video Processing**
   - Deploy video object tracking on edge device
   - Implement temporal optimization techniques
   - Handle varying lighting and occlusion
   - Target: 30 FPS on embedded GPU

### Assessment and Certification

#### Knowledge Assessment Questions

**Conceptual Questions:**
1. Explain the trade-offs between different quantization methods (INT8, FP16, binary)
2. When would you choose TensorFlow Lite vs PyTorch Mobile vs ONNX Runtime?
3. How do memory bandwidth limitations affect edge AI performance?
4. What are the key considerations for power-efficient AI inference?
5. Describe the challenges of deploying AI on microcontrollers vs mobile devices

**Technical Implementation:**
6. Implement gradient-based quantization calibration
7. Design memory pooling system for inference engines
8. Create adaptive batching algorithm for varying loads
9. Implement custom pruning strategy for specific architecture
10. Build power profiling framework for edge devices

#### Practical Assessment
- **Portfolio Project:** Complete end-to-end edge AI application
- **Performance Benchmarking:** Comprehensive analysis across multiple devices
- **Optimization Report:** Document optimization process and results
- **Code Review:** Peer review of edge deployment implementation

### Industry Connections and Next Steps

#### Professional Development
- **Edge AI Engineer Certification** (NVIDIA, Intel, Qualcomm)
- **Mobile AI Developer Specialization** (Google, Apple)
- **Embedded AI Professional** (Edge Impulse, ARM)

#### Research and Advanced Topics
- **Neural Architecture Search** for edge devices
- **Federated Learning** on edge networks
- **Edge-Cloud Hybrid Intelligence**
- **Neuromorphic Computing** for ultra-low power AI
- **AI Accelerator Design** and custom silicon

#### Career Paths
- **Mobile AI Developer:** Focus on smartphone and tablet applications
- **Embedded AI Engineer:** IoT and microcontroller deployment
- **Edge Infrastructure Architect:** Design edge computing platforms
- **AI Hardware Specialist:** Optimize AI for specific hardware
- **Performance Engineer:** Focus on optimization and benchmarking

### Community and Continued Learning

#### Online Communities
- **Edge AI Forums:** Reddit r/MachineLearning, Stack Overflow
- **Mobile AI Groups:** LinkedIn professional groups
- **Framework Communities:** TensorFlow/PyTorch Discord servers
- **Hardware Communities:** NVIDIA Developer forums, ARM Community

#### Conferences and Events
- **Mobile World Congress:** Latest mobile AI hardware
- **Edge Computing World:** Industry applications and case studies
- **NeurIPS/ICML:** Latest research in efficient AI
- **Embedded Vision Summit:** Computer vision on edge devices

#### Open Source Contributions
- Contribute to TensorFlow Lite, PyTorch Mobile, ONNX Runtime
- Develop optimization tools and benchmarking suites
- Create educational content and tutorials
- Build example applications and demos

This comprehensive study plan provides both theoretical knowledge and practical skills needed for successful edge AI deployment across various platforms and use cases.

## Next Steps

### Advanced Topics to Explore
- [Federated Learning for Edge Devices](../Advanced_Topics/01_Federated_Learning.md)
- [Neural Architecture Search for Mobile](../Advanced_Topics/02_Mobile_NAS.md)
- [Hardware-Software Co-design](../Advanced_Topics/03_HW_SW_Codesign.md)
- [Edge-Cloud Hybrid Systems](../Advanced_Topics/04_Hybrid_Systems.md)

### Related Learning Paths
- [Model Optimization and Compression](02_Model_Optimization.md)
- [Hardware Acceleration](03_Hardware_Acceleration.md)
- [Production ML Systems](../Production_ML/01_MLOps_Fundamentals.md)
- [Computer Vision Applications](../Applications/01_Computer_Vision.md)

## Edge-Specific Optimizations

Edge deployment requires specialized optimization techniques to balance accuracy, performance, and resource constraints. These optimizations are crucial for real-world edge AI applications.

### Binary and Ternary Networks

Binary and ternary quantization represent extreme quantization techniques that drastically reduce model size and computation requirements.

#### Binary Neural Networks (BNNs)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BinaryActivation(torch.autograd.Function):
    """Binary activation function with straight-through estimator"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Straight-through estimator: pass gradient through if |input| <= 1
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

class BinaryLinear(nn.Module):
    """Binary linear layer"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Binary weight scaling factor
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        
    def forward(self, input):
        # Binarize weights
        binary_weight = torch.sign(self.weight)
        
        # Scale binary weights
        scaled_weight = binary_weight * self.weight_scale
        
        # Binary activation
        binary_input = BinaryActivation.apply(input)
        
        # Convolution with binary weights and activations
        output = F.linear(binary_input, scaled_weight, self.bias)
        
        return output

class BinaryConv2d(nn.Module):
    """Binary convolutional layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        
        # Scaling factors
        self.weight_scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        
    def forward(self, input):
        # Binarize weights: sign function
        binary_weight = torch.sign(self.weight)
        
        # Scale binary weights
        scaled_weight = binary_weight * self.weight_scale
        
        # Binary activation
        binary_input = BinaryActivation.apply(input)
        
        # Convolution
        output = F.conv2d(binary_input, scaled_weight, self.bias, self.stride, self.padding)
        
        return output

class BinaryMobileNet(nn.Module):
    """Binary version of MobileNet for edge deployment"""
    
    def __init__(self, num_classes=10):
        super(BinaryMobileNet, self).__init__()
        
        # First layer remains full precision
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Binary depthwise separable blocks
        self.features = nn.Sequential(
            self._make_binary_block(32, 64, 1),
            self._make_binary_block(64, 128, 2),
            self._make_binary_block(128, 128, 1),
            self._make_binary_block(128, 256, 2),
            self._make_binary_block(256, 256, 1),
            self._make_binary_block(256, 512, 2),
            # Multiple 512 channel blocks
            *[self._make_binary_block(512, 512, 1) for _ in range(5)],
            self._make_binary_block(512, 1024, 2),
            self._make_binary_block(1024, 1024, 1),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Final classifier (keep full precision for better accuracy)
        self.classifier = nn.Linear(1024, num_classes)
        
    def _make_binary_block(self, in_channels, out_channels, stride):
        """Create a binary depthwise separable block"""
        return nn.Sequential(
            # Depthwise convolution (binary)
            BinaryConv2d(in_channels, in_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution (binary)
            BinaryConv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # First conv (full precision)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Binary features
        x = self.features(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# Training binary networks
def train_binary_network():
    """Training procedure for binary networks"""
    
    model = BinaryMobileNet(num_classes=10)
    
    # Use larger learning rate for binary networks
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEnLoss()
    
    # Training loop with weight clipping
    model.train()
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            
            # Clip weights to [-1, 1] range
            for param in model.parameters():
                if param.grad is not None:
                    param.data.clamp_(-1, 1)
            
            optimizer.step()
        
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Convert to deployment format
def convert_binary_model_for_deployment(model):
    """Convert trained binary model for efficient deployment"""
    
    model.eval()
    
    # Extract binary weights
    binary_weights = {}
    scaling_factors = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (BinaryConv2d, BinaryLinear)):
            # Binarize weights
            binary_weights[name] = torch.sign(module.weight.data)
            scaling_factors[name] = module.weight_scale.data
    
    # Pack binary weights efficiently (32 weights per int32)
    packed_weights = {}
    for name, weight in binary_weights.items():
        # Flatten and pack
        flat_weight = weight.flatten()
        # Convert {-1, 1} to {0, 1}
        binary_bits = (flat_weight + 1) // 2
        
        # Pack into int32
        packed = []
        for i in range(0, len(binary_bits), 32):
            chunk = binary_bits[i:i+32]
            packed_value = 0
            for j, bit in enumerate(chunk):
                if bit == 1:
                    packed_value |= (1 << j)
            packed.append(packed_value)
        
        packed_weights[name] = packed
    
    return {
        'packed_weights': packed_weights,
        'scaling_factors': scaling_factors,
        'model_structure': model.state_dict()
    }
```

#### Ternary Neural Networks (TNNs)
```python
class TernaryActivation(torch.autograd.Function):
    """Ternary activation function {-1, 0, 1}"""
    
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        
        # Ternary quantization
        output = input.clone()
        output[input > threshold] = 1
        output[input < -threshold] = -1
        output[torch.abs(input) <= threshold] = 0
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # Straight-through estimator
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 1] = 0
        
        return grad_input, None

class TernaryConv2d(nn.Module):
    """Ternary convolutional layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TernaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Learnable threshold for ternary quantization
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # Scaling factors for positive and negative weights
        self.alpha_p = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        self.alpha_n = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
    
    def forward(self, input):
        # Ternary weight quantization
        weight_mean = torch.mean(torch.abs(self.weight))
        threshold = self.threshold * weight_mean
        
        ternary_weight = torch.zeros_like(self.weight)
        ternary_weight[self.weight > threshold] = 1
        ternary_weight[self.weight < -threshold] = -1
        
        # Apply different scaling for positive and negative weights
        scaled_weight = ternary_weight.clone()
        scaled_weight[ternary_weight == 1] *= self.alpha_p[ternary_weight == 1]
        scaled_weight[ternary_weight == -1] *= self.alpha_n[ternary_weight == -1]
        
        # Ternary activation
        ternary_input = TernaryActivation.apply(input, threshold)
        
        # Convolution
        output = F.conv2d(ternary_input, scaled_weight, None, self.stride, self.padding)
        
        return output
```

### Sparse Computation

Sparsity exploits the fact that many neural network weights and activations are zero or near-zero, allowing for computational savings.

#### Structured Sparsity Implementation
```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class SparseConv2d(nn.Module):
    """Sparse convolutional layer with structured pruning"""
    
    def __init__(self, in_channels, out_channels, kernel_size, sparsity_ratio=0.5):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.sparsity_ratio = sparsity_ratio
        
        # Apply structured pruning during initialization
        self.apply_structured_pruning()
    
    def apply_structured_pruning(self):
        """Apply channel-wise structured pruning"""
        # Prune entire channels based on L2 norm
        prune.ln_structured(
            self.conv, 
            name='weight', 
            amount=self.sparsity_ratio, 
            n=2, 
            dim=0  # Prune output channels
        )
        
        # Remove pruning re-parametrization to make it permanent
        prune.remove(self.conv, 'weight')
    
    def forward(self, x):
        return self.conv(x)

class SparseLinear(nn.Module):
    """Sparse linear layer with unstructured pruning"""
    
    def __init__(self, in_features, out_features, sparsity_ratio=0.8):
        super(SparseLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sparsity_ratio = sparsity_ratio
        
        self.apply_unstructured_pruning()
    
    def apply_unstructured_pruning(self):
        """Apply magnitude-based unstructured pruning"""
        prune.l1_unstructured(
            self.linear,
            name='weight',
            amount=self.sparsity_ratio
        )
        
        # Make pruning permanent
        prune.remove(self.linear, 'weight')
    
    def forward(self, x):
        return self.linear(x)

# Sparse model with block-wise sparsity
class BlockSparseLinear(nn.Module):
    """Block-sparse linear layer for hardware efficiency"""
    
    def __init__(self, in_features, out_features, block_size=4, sparsity_ratio=0.75):
        super(BlockSparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.sparsity_ratio = sparsity_ratio
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Create block sparse mask
        self.register_buffer('mask', self.create_block_sparse_mask())
    
    def create_block_sparse_mask(self):
        """Create block-sparse mask"""
        mask = torch.ones(self.out_features, self.in_features)
        
        # Calculate number of blocks
        blocks_per_row = self.in_features // self.block_size
        blocks_per_col = self.out_features // self.block_size
        
        # Randomly select blocks to keep
        total_blocks = blocks_per_row * blocks_per_col
        blocks_to_keep = int(total_blocks * (1 - self.sparsity_ratio))
        
        # Create block indices
        block_indices = torch.randperm(total_blocks)[:blocks_to_keep]
        
        # Set mask to zero for pruned blocks
        mask.fill_(0)
        for idx in block_indices:
            row_block = idx // blocks_per_row
            col_block = idx % blocks_per_row
            
            row_start = row_block * self.block_size
            row_end = min(row_start + self.block_size, self.out_features)
            col_start = col_block * self.block_size
            col_end = min(col_start + self.block_size, self.in_features)
            
            mask[row_start:row_end, col_start:col_end] = 1
        
        return mask
    
    def forward(self, x):
        # Apply sparse mask
        sparse_weight = self.weight * self.mask
        return F.linear(x, sparse_weight, self.bias)

# Training with gradual pruning
class GradualPruningTrainer:
    """Trainer with gradual magnitude pruning"""
    
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.9, 
                 pruning_frequency=100, start_epoch=10, end_epoch=90):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        
    def get_current_sparsity(self, epoch):
        """Calculate current sparsity level"""
        if epoch < self.start_epoch:
            return self.initial_sparsity
        elif epoch > self.end_epoch:
            return self.final_sparsity
        else:
            # Linear increase in sparsity
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
    
    def prune_model(self, epoch):
        """Apply pruning to model"""
        if epoch >= self.start_epoch and epoch % self.pruning_frequency == 0:
            current_sparsity = self.get_current_sparsity(epoch)
            
            # Apply global magnitude pruning
            parameters_to_prune = []
            for module in self.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=current_sparsity
            )
            
            print(f"Epoch {epoch}: Applied {current_sparsity:.2f} sparsity")
    
    def finalize_pruning(self):
        """Remove pruning reparametrization"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass  # No pruning applied to this module

# Sparse inference optimization
def optimize_sparse_inference(model):
    """Optimize sparse model for inference"""
    
    # Convert to sparse tensor format for efficient computation
    optimized_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            
            # Check if weight is sparse enough to benefit from sparse format
            sparsity = (weight == 0).float().mean().item()
            
            if sparsity > 0.5:  # More than 50% sparse
                # Convert to COO sparse format
                sparse_weight = weight.to_sparse()
                
                # Store sparse representation
                optimized_modules[name] = {
                    'indices': sparse_weight.indices(),
                    'values': sparse_weight.values(),
                    'shape': sparse_weight.shape,
                    'sparsity': sparsity
                }
                
                print(f"Module {name}: {sparsity:.2f} sparsity, "
                      f"memory reduction: {sparsity:.1%}")
    
    return optimized_modules
```

### Memory Bandwidth Optimization

Memory bandwidth is often the bottleneck in edge devices. These techniques optimize memory access patterns.

#### Memory Access Pattern Optimization
```python
import torch
import torch.nn as nn
import numpy as np

class MemoryEfficientConv2d(nn.Module):
    """Memory-efficient convolution with optimized access patterns"""
    
    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(MemoryEfficientConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        
        # Use depthwise separable convolution to reduce memory bandwidth
        if groups == in_channels and in_channels == out_channels:
            # Depthwise convolution
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                groups=groups, padding=kernel_size//2)
        else:
            # Regular convolution with memory optimization
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                padding=kernel_size//2)
    
    def forward(self, x):
        # Optimize memory layout for better cache efficiency
        if x.is_contiguous():
            return self.conv(x)
        else:
            # Make input contiguous to improve memory access
            return self.conv(x.contiguous())

class TiledConvolution(nn.Module):
    """Tiled convolution for reduced memory footprint"""
    
    def __init__(self, in_channels, out_channels, kernel_size, tile_size=64):
        super(TiledConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding=kernel_size//2)
        self.tile_size = tile_size
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Process input in tiles to reduce peak memory usage
        if H > self.tile_size or W > self.tile_size:
            return self.tiled_forward(x)
        else:
            return self.conv(x)
    
    def tiled_forward(self, x):
        B, C, H, W = x.shape
        
        # Calculate tile dimensions
        tile_h = min(self.tile_size, H)
        tile_w = min(self.tile_size, W)
        
        # Prepare output tensor
        with torch.no_grad():
            dummy_output = self.conv(x[:1, :, :tile_h, :tile_w])
            output_shape = (B, dummy_output.shape[1], H, W)
        
        output = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
        
        # Process tiles
        for h_start in range(0, H, tile_h):
            h_end = min(h_start + tile_h, H)
            
            for w_start in range(0, W, tile_w):
                w_end = min(w_start + tile_w, W)
                
                # Extract tile with padding consideration
                tile_input = x[:, :, h_start:h_end, w_start:w_end]
                
                # Process tile
                tile_output = self.conv(tile_input)
                
                # Place result in output
                output[:, :, h_start:h_end, w_start:w_end] = tile_output
        
        return output

# In-place operations to reduce memory allocation
class InPlaceReLU(nn.Module):
    """In-place ReLU to save memory"""
    
    def __init__(self):
        super(InPlaceReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x)

class MemoryOptimizedBlock(nn.Module):
    """Memory-optimized residual block"""
    
    def __init__(self, channels):
        super(MemoryOptimizedBlock, self).__init__()
        
        # Use grouped convolutions to reduce memory bandwidth
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels//4)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = InPlaceReLU()
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels//4)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Use additive residual instead of concatenation
        self.relu2 = InPlaceReLU()
    
    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual (in-place when possible)
        out += residual
        out = self.relu2(out)
        
        return out

# Memory pooling and reuse
class MemoryPool:
    """Memory pool for tensor reuse"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.pool = {}
    
    def get_tensor(self, shape, dtype=torch.float32):
        """Get tensor from pool or create new one"""
        key = (shape, dtype)
        
        if key in self.pool and len(self.pool[key]) > 0:
            return self.pool[key].pop()
        else:
            return torch.empty(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor):
        """Return tensor to pool for reuse"""
        key = (tuple(tensor.shape), tensor.dtype)
        
        if key not in self.pool:
            self.pool[key] = []
        
        # Clear tensor data and return to pool
        tensor.zero_()
        self.pool[key].append(tensor)
    
    def clear_pool(self):
        """Clear all tensors from pool"""
        self.pool.clear()

# Memory-efficient model wrapper
class MemoryEfficientModel(nn.Module):
    """Wrapper for memory-efficient inference"""
    
    def __init__(self, model):
        super(MemoryEfficientModel, self).__init__()
        self.model = model
        self.memory_pool = MemoryPool()
    
    def forward(self, x):
        # Enable memory optimization
        with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False):
            # Use gradient checkpointing if available
            if hasattr(torch.utils.checkpoint, 'checkpoint'):
                return torch.utils.checkpoint.checkpoint(self.model, x)
            else:
                return self.model(x)
    
    def inference_with_memory_reuse(self, x):
        """Inference with explicit memory management"""
        
        # Pre-allocate intermediate tensors
        intermediate_shapes = self.estimate_intermediate_shapes(x)
        intermediate_tensors = [
            self.memory_pool.get_tensor(shape) 
            for shape in intermediate_shapes
        ]
        
        try:
            # Run inference with pre-allocated tensors
            output = self.model(x)
            return output
            
        finally:
            # Return tensors to pool
            for tensor in intermediate_tensors:
                self.memory_pool.return_tensor(tensor)
    
    def estimate_intermediate_shapes(self, x):
        """Estimate shapes of intermediate tensors for pre-allocation"""
        shapes = []
        
        # This is a simplified example - in practice, you'd need to
        # trace through your specific model architecture
        current_shape = list(x.shape)
        
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                # Estimate output shape after convolution
                # This is simplified - actual calculation depends on stride, padding, etc.
                current_shape[1] = module.out_channels
                shapes.append(tuple(current_shape))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                current_shape[2] = module.output_size[0] if hasattr(module, 'output_size') else 1
                current_shape[3] = module.output_size[1] if hasattr(module, 'output_size') else 1
                shapes.append(tuple(current_shape))
        
        return shapes
```

### Power Efficiency Techniques

Power efficiency is critical for battery-powered edge devices and embedded systems.

#### Dynamic Voltage and Frequency Scaling (DVFS) Simulation
```python
import time
import threading
from typing import Dict, List, Tuple

class PowerManager:
    """Simulates power management for edge AI inference"""
    
    def __init__(self):
        self.power_states = {
            'high_performance': {'freq_ghz': 2.5, 'voltage': 1.2, 'power_watts': 15.0},
            'balanced': {'freq_ghz': 1.8, 'voltage': 1.0, 'power_watts': 8.0},
            'power_saver': {'freq_ghz': 1.0, 'voltage': 0.8, 'power_watts': 3.0},
            'ultra_low': {'freq_ghz': 0.5, 'voltage': 0.6, 'power_watts': 1.0}
        }
        self.current_state = 'balanced'
        self.performance_history = []
        self.power_consumption = 0.0
        self.battery_level = 100.0  # Percentage
    
    def set_power_state(self, state: str):
        """Set CPU power state"""
        if state in self.power_states:
            self.current_state = state
            print(f"Power state changed to: {state}")
            print(f"Frequency: {self.power_states[state]['freq_ghz']}GHz, "
                  f"Power: {self.power_states[state]['power_watts']}W")
        else:
            raise ValueError(f"Invalid power state: {state}")
    
    def estimate_inference_time(self, base_time_ms: float) -> float:
        """Estimate inference time based on current power state"""
        base_freq = self.power_states['high_performance']['freq_ghz']
        current_freq = self.power_states[self.current_state]['freq_ghz']
        
        # Simple linear scaling (in reality, it's more complex)
        time_scaling = base_freq / current_freq
        return base_time_ms * time_scaling
    
    def update_battery(self, inference_time_ms: float):
        """Update battery level based on power consumption"""
        power_watts = self.power_states[self.current_state]['power_watts']
        energy_consumed = power_watts * (inference_time_ms / 1000) / 3600  # Wh
        
        # Assume 50Wh battery capacity
        battery_capacity_wh = 50.0
        battery_drain = (energy_consumed / battery_capacity_wh) * 100
        
        self.battery_level = max(0, self.battery_level - battery_drain)
        self.power_consumption += energy_consumed
    
    def adaptive_power_management(self, workload_intensity: float, 
                                battery_threshold: float = 20.0):
        """Adaptive power management based on workload and battery"""
        
        if self.battery_level < battery_threshold:
            # Low battery - prioritize power saving
            if workload_intensity < 0.3:
                self.set_power_state('ultra_low')
            else:
                self.set_power_state('power_saver')
        elif workload_intensity > 0.8:
            # High workload - prioritize performance
            self.set_power_state('high_performance')
        elif workload_intensity > 0.5:
            # Medium workload - balanced approach
            self.set_power_state('balanced')
        else:
            # Low workload - save power
            self.set_power_state('power_saver')

class PowerEfficientInference:
    """Power-efficient AI inference engine"""
    
    def __init__(self, model, power_manager: PowerManager):
        self.model = model
        self.power_manager = power_manager
        self.inference_history = []
        self.adaptive_threshold = 50.0  # ms
    
    def run_inference_with_power_management(self, input_data, 
                                          target_latency_ms: float = None):
        """Run inference with dynamic power management"""
        
        # Estimate required performance level
        if target_latency_ms:
            required_performance = self.estimate_required_performance(target_latency_ms)
            self.power_manager.adaptive_power_management(required_performance)
        
        # Run inference
        start_time = time.time()
        
        # Simulate model inference (replace with actual model call)
        with torch.no_grad():
            output = self.model(input_data)
        
        actual_time_ms = (time.time() - start_time) * 1000
        
        # Update power consumption
        self.power_manager.update_battery(actual_time_ms)
        
        # Record performance
        self.inference_history.append({
            'timestamp': time.time(),
            'inference_time_ms': actual_time_ms,
            'power_state': self.power_manager.current_state,
            'battery_level': self.power_manager.battery_level
        })
        
        return output, actual_time_ms
    
    def estimate_required_performance(self, target_latency_ms: float) -> float:
        """Estimate required performance level to meet latency target"""
        
        if not self.inference_history:
            return 0.5  # Default to balanced
        
        # Use recent performance to estimate requirements
        recent_times = [h['inference_time_ms'] for h in self.inference_history[-10:]]
        avg_time = sum(recent_times) / len(recent_times)
        
        if avg_time > target_latency_ms * 1.2:
            return 1.0  # Need high performance
        elif avg_time > target_latency_ms:
            return 0.8  # Need above balanced
        elif avg_time < target_latency_ms * 0.5:
            return 0.2  # Can use low power
        else:
            return 0.5  # Balanced is fine
    
    def batch_inference_with_power_scaling(self, batch_inputs: List, 
                                         max_batch_size: int = 8):
        """Process batch with dynamic batching based on power state"""
        
        # Adjust batch size based on power state
        power_state = self.power_manager.current_state
        if power_state == 'ultra_low':
            effective_batch_size = 1
        elif power_state == 'power_saver':
            effective_batch_size = max(1, max_batch_size // 4)
        elif power_state == 'balanced':
            effective_batch_size = max_batch_size // 2
        else:  # high_performance
            effective_batch_size = max_batch_size
        
        results = []
        total_time = 0
        
        for i in range(0, len(batch_inputs), effective_batch_size):
            batch = batch_inputs[i:i + effective_batch_size]
            
            # Stack batch
            if len(batch) == 1:
                batch_tensor = batch[0].unsqueeze(0)
            else:
                batch_tensor = torch.stack(batch)
            
            # Run inference
            output, inference_time = self.run_inference_with_power_management(
                batch_tensor
            )
            
            # Unpack results
            for j in range(len(batch)):
                results.append(output[j] if len(batch) > 1 else output[0])
            
            total_time += inference_time
        
        return results, total_time
    
    def get_power_efficiency_report(self):
        """Generate power efficiency report"""
        if not self.inference_history:
            return "No inference history available"
        
        total_inferences = len(self.inference_history)
        total_time_ms = sum(h['inference_time_ms'] for h in self.inference_history)
        total_energy_consumed = self.power_manager.power_consumption
        
        # Calculate metrics
        avg_inference_time = total_time_ms / total_inferences
        avg_power_per_inference = total_energy_consumed / total_inferences * 3600 * 1000  # mWh
        
        # Power state distribution
        state_counts = {}
        for h in self.inference_history:
            state = h['power_state']
            state_counts[state] = state_counts.get(state, 0) + 1
        
        report = f"""
Power Efficiency Report:
========================
Total Inferences: {total_inferences}
Average Inference Time: {avg_inference_time:.1f}ms
Total Energy Consumed: {total_energy_consumed:.3f}Wh
Average Energy per Inference: {avg_power_per_inference:.2f}mWh
Current Battery Level: {self.power_manager.battery_level:.1f}%

Power State Distribution:
"""
        for state, count in state_counts.items():
            percentage = (count / total_inferences) * 100
            report += f"  {state}: {count} ({percentage:.1f}%)\n"
        
        return report

# Example usage
def power_efficient_edge_deployment():
    """Example of power-efficient edge AI deployment"""
    
    # Initialize components
    power_manager = PowerManager()
    
    # Create a simple model (replace with your actual model)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 10)
    )
    model.eval()
    
    inference_engine = PowerEfficientInference(model, power_manager)
    
    # Simulate inference workload
    print("Starting power-efficient inference simulation...")
    
    for i in range(50):
        # Create dummy input
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Vary target latency to simulate different scenarios
        if i < 10:
            target_latency = 100.0  # Relaxed timing
        elif i < 30:
            target_latency = 50.0   # Moderate timing
        else:
            target_latency = 20.0   # Tight timing
        
        # Run inference
        output, actual_time = inference_engine.run_inference_with_power_management(
            input_tensor, target_latency
        )
        
        print(f"Inference {i+1}: {actual_time:.1f}ms, "
              f"Power: {power_manager.current_state}, "
              f"Battery: {power_manager.battery_level:.1f}%")
        
        # Simulate some delay between inferences
        time.sleep(0.1)
    
    # Print final report
    print("\n" + inference_engine.get_power_efficiency_report())

if __name__ == "__main__":
    power_efficient_edge_deployment()
```

This comprehensive edge optimization section covers:

1. **Binary/Ternary Networks**: Extreme quantization techniques that reduce both model size and computation
2. **Sparse Computation**: Structured and unstructured sparsity with pruning strategies
3. **Memory Bandwidth Optimization**: Techniques to optimize memory access patterns and reduce peak memory usage
4. **Power Efficiency**: Dynamic power management and adaptive inference strategies

Each technique includes practical implementation examples and can be combined for maximum efficiency in edge deployment scenarios.


## Learning Objectives

By the end of this section, you should be able to:

### Technical Proficiency
- **Convert and optimize models** for edge deployment using TensorFlow Lite, PyTorch Mobile, and ONNX Runtime
- **Implement quantization techniques** including INT8, FP16, binary, and ternary networks
- **Apply model compression** through pruning, sparsity, and knowledge distillation
- **Deploy models** on mobile devices (iOS/Android), microcontrollers, and embedded systems
- **Optimize memory usage** and implement efficient memory management strategies
- **Configure hardware acceleration** using GPU delegates, specialized chips, and edge TPUs

### Practical Skills
- **Benchmark and profile** edge AI models for latency, throughput, and energy consumption
- **Implement power management** strategies for battery-powered devices
- **Debug edge deployment issues** including memory constraints and performance bottlenecks
- **Design inference pipelines** that balance accuracy, speed, and resource usage
- **Integrate edge AI** into real applications with proper error handling and fallback mechanisms

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Convert a trained PyTorch/TensorFlow model to mobile-optimized format  
□ Implement INT8 quantization with accuracy validation  
□ Deploy a model to an Android/iOS app with proper preprocessing  
□ Optimize model inference for sub-100ms latency on edge devices  
□ Implement memory-efficient inference for devices with <1GB RAM  
□ Apply structured pruning to reduce model size by 50%+ while maintaining accuracy  
□ Configure different hardware acceleration providers (CPU, GPU, NPU)  
□ Benchmark model performance across different power states  
□ Debug common edge deployment issues (OOM, slow inference, accuracy drop)  
□ Design a complete edge AI pipeline from data input to result output  

### Hands-on Projects

**Project 1: Mobile Image Classifier**
```python
# TODO: Complete this mobile deployment pipeline

class MobileImageClassifier:
    def __init__(self, model_path, platform='android'):
        # Your implementation here
        pass
    
    def preprocess_image(self, image_path):
        # Implement mobile-optimized preprocessing
        pass
    
    def run_inference(self, input_data):
        # Implement efficient inference with timing
        pass
    
    def postprocess_results(self, model_output):
        # Convert model output to user-friendly format
        pass

# Requirements:
# - Support both TensorFlow Lite and PyTorch Mobile
# - Implement quantized inference (INT8)
# - Optimize for <50ms inference time
# - Handle edge cases (low memory, battery saving mode)
# - Include performance benchmarking
```

**Project 2: IoT Sensor Data Classifier**
```python
# TODO: Implement edge AI for IoT sensor classification

class IoTEdgeClassifier:
    def __init__(self, model_path, sensor_config):
        # Initialize for microcontroller deployment
        pass
    
    def collect_sensor_data(self):
        # Simulate sensor data collection (accelerometer, gyro, etc.)
        pass
    
    def run_real_time_inference(self):
        # Continuous inference loop with power management
        pass
    
    def handle_edge_cases(self):
        # Low battery, connectivity loss, sensor failures
        pass

# Requirements:
# - Implement binary/ternary quantization
# - Memory usage < 512KB
# - Power consumption optimization
# - Real-time inference (10Hz+)
# - Robust error handling
```

**Project 3: Edge AI Performance Optimizer**
```python
# TODO: Build a comprehensive optimization tool

class EdgeOptimizer:
    def __init__(self, model, target_device):
        # Auto-optimization for specific edge devices
        pass
    
    def analyze_model_complexity(self):
        # Analyze FLOPS, parameters, memory requirements
        pass
    
    def suggest_optimizations(self):
        # Recommend quantization, pruning, architecture changes
        pass
    
    def apply_optimizations(self, optimization_config):
        # Apply selected optimizations automatically
        pass
    
    def validate_optimized_model(self):
        # Comprehensive validation: accuracy, speed, memory
        pass

# Requirements:
# - Support multiple frameworks (TF, PyTorch, ONNX)
# - Automated optimization recommendation
# - Accuracy vs. performance trade-off analysis
# - Integration with hardware profiling tools
# - Export optimized models for different platforms
```
