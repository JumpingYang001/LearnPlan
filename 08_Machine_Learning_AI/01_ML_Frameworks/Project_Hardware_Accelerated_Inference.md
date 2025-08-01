# Project: Hardware-Accelerated Inference

*Duration: 2-3 weeks | Difficulty: Advanced*

## Project Overview

This comprehensive project focuses on optimizing machine learning models for efficient deployment on various hardware platforms, from edge devices to high-performance servers. You'll learn to implement model compression techniques, optimize for different hardware accelerators, and benchmark performance across multiple targets.

## Learning Objectives

By completing this project, you will:
- **Master model optimization techniques** including quantization, pruning, and knowledge distillation
- **Implement edge deployment pipelines** for resource-constrained devices
- **Utilize hardware accelerators** (GPU, TPU, Neural Processing Units)
- **Benchmark and profile** model performance across different platforms
- **Understand trade-offs** between model accuracy, size, and inference speed
- **Deploy models** using TensorRT, ONNX Runtime, and TensorFlow Lite
- **Optimize memory usage** and reduce computational overhead

## Project Architecture

```
Hardware-Accelerated Inference Pipeline
├── Model Preparation
│   ├── Base Model Training/Loading
│   ├── Model Analysis & Profiling
│   └── Baseline Benchmarking
├── Optimization Techniques
│   ├── Quantization (INT8, FP16, Dynamic)
│   ├── Pruning (Structured/Unstructured)
│   ├── Knowledge Distillation
│   └── Graph Optimization
├── Hardware-Specific Optimization
│   ├── GPU (CUDA, TensorRT)
│   ├── CPU (OpenVINO, ONNX)
│   ├── TPU (TensorFlow Lite)
│   └── Mobile/Edge (ARM, NPU)
├── Deployment & Benchmarking
│   ├── Performance Metrics Collection
│   ├── Memory Usage Analysis
│   └── Cross-Platform Comparison
└── Production Integration
    ├── API Development
    ├── Containerization
    └── Monitoring & Logging
```

## Hardware Targets

This project covers optimization for multiple hardware platforms:

| Hardware Type | Examples | Use Cases | Key Optimizations |
|---------------|----------|-----------|-------------------|
| **CPU** | Intel x86, ARM Cortex | General compute, edge | SIMD, vectorization, threading |
| **GPU** | NVIDIA RTX, AMD Radeon | High-throughput inference | CUDA, cuDNN, mixed precision |
| **TPU** | Google TPU, Edge TPU | ML-specific workloads | XLA compilation, quantization |
| **Mobile** | Apple A-series, Snapdragon | Mobile apps | Model compression, NPU acceleration |
| **Edge** | Raspberry Pi, Jetson Nano | IoT, embedded systems | Ultra-low power, minimal memory |

## Part 1: Model Preparation and Analysis

### 1.1 Base Model Setup

First, let's establish a baseline model and analysis framework:

```python
import tensorflow as tf
import torch
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

class ModelAnalyzer:
    """Comprehensive model analysis and benchmarking toolkit"""
    
    def __init__(self, model_path: str, framework: str = 'tensorflow'):
        self.model_path = model_path
        self.framework = framework
        self.model = self._load_model()
        self.baseline_metrics = {}
        
    def _load_model(self):
        """Load model based on framework"""
        if self.framework == 'tensorflow':
            return tf.keras.models.load_model(self.model_path)
        elif self.framework == 'pytorch':
            return torch.load(self.model_path)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def analyze_model_structure(self) -> Dict:
        """Analyze model architecture and parameters"""
        if self.framework == 'tensorflow':
            total_params = self.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) 
                                  for w in self.model.trainable_weights])
            
            layer_info = []
            for layer in self.model.layers:
                layer_info.append({
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'output_shape': str(layer.output_shape),
                    'params': layer.count_params()
                })
                
        elif self.framework == 'pytorch':
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() 
                                 if p.requires_grad)
            
            layer_info = []
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    params = sum(p.numel() for p in module.parameters())
                    layer_info.append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': params
                    })
        
        analysis = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming FP32
            'layers': layer_info
        }
        
        return analysis
    
    def benchmark_inference(self, input_shape: Tuple, num_iterations: int = 100) -> Dict:
        """Benchmark inference performance"""
        # Generate random input data
        if self.framework == 'tensorflow':
            dummy_input = tf.random.normal(input_shape)
            
            # Warmup
            for _ in range(10):
                _ = self.model(dummy_input)
            
            # Actual benchmarking
            start_time = time.time()
            for _ in range(num_iterations):
                predictions = self.model(dummy_input)
            end_time = time.time()
            
        elif self.framework == 'pytorch':
            dummy_input = torch.randn(input_shape)
            self.model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # Actual benchmarking
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    predictions = self.model(dummy_input)
            end_time = time.time()
        
        total_time = end_time - start_time
        avg_inference_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time,
            'iterations': num_iterations
        }
    
    def measure_memory_usage(self, input_shape: Tuple) -> Dict:
        """Measure memory usage during inference"""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.framework == 'tensorflow':
            dummy_input = tf.random.normal(input_shape)
            predictions = self.model(dummy_input)
        elif self.framework == 'pytorch':
            dummy_input = torch.randn(input_shape)
            with torch.no_grad():
                predictions = self.model(dummy_input)
        
        # Peak memory after inference
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - baseline_memory
        }

# Example usage
def setup_baseline_model():
    """Create and analyze a baseline model"""
    
    # Create a sample CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Save the model
    model.save('baseline_model')
    
    # Analyze the model
    analyzer = ModelAnalyzer('baseline_model', 'tensorflow')
    
    # Get structural analysis
    structure = analyzer.analyze_model_structure()
    print("Model Structure Analysis:")
    print(f"Total Parameters: {structure['total_parameters']:,}")
    print(f"Model Size: {structure['model_size_mb']:.2f} MB")
    
    # Benchmark performance
    input_shape = (1, 224, 224, 3)
    performance = analyzer.benchmark_inference(input_shape)
    print(f"\nBaseline Performance:")
    print(f"Average Inference Time: {performance['avg_inference_time_ms']:.2f} ms")
    print(f"Throughput: {performance['throughput_fps']:.2f} FPS")
    
    # Measure memory usage
    memory = analyzer.measure_memory_usage(input_shape)
    print(f"\nMemory Usage:")
    print(f"Peak Memory: {memory['peak_memory_mb']:.2f} MB")
    
    return model, analyzer

if __name__ == "__main__":
    baseline_model, analyzer = setup_baseline_model()
```

### 1.2 Model Profiling and Bottleneck Analysis

```python
import tensorflow as tf
from tensorflow.python.profiler import profiler_v2 as profiler
import torch
import torch.profiler

class ModelProfiler:
    """Advanced model profiling for identifying bottlenecks"""
    
    def __init__(self, model, framework='tensorflow'):
        self.model = model
        self.framework = framework
    
    def profile_tensorflow_model(self, input_data, logdir='./logs'):
        """Profile TensorFlow model using TensorBoard profiler"""
        
        # Start profiling
        profiler.start(logdir)
        
        # Run inference with profiling
        with tf.profiler.experimental.Trace('inference'):
            for _ in range(100):
                predictions = self.model(input_data)
        
        # Stop profiling
        profiler.stop()
        
        print(f"Profiling data saved to {logdir}")
        print("View with: tensorboard --logdir=./logs")
        
        return predictions
    
    def profile_pytorch_model(self, input_data):
        """Profile PyTorch model using torch.profiler"""
        
        self.model.eval()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/pytorch'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            for step in range(10):
                with torch.no_grad():
                    predictions = self.model(input_data)
                prof.step()
        
        # Print summary
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        return predictions
    
    def analyze_layer_performance(self, input_data):
        """Analyze per-layer performance"""
        
        if self.framework == 'tensorflow':
            # Custom timing for TensorFlow layers
            layer_times = {}
            
            # Create intermediate models for each layer
            for i, layer in enumerate(self.model.layers):
                intermediate_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                
                # Time this layer
                start_time = time.time()
                for _ in range(50):
                    _ = intermediate_model(input_data)
                end_time = time.time()
                
                layer_times[f"layer_{i}_{layer.name}"] = (end_time - start_time) / 50 * 1000
            
            return layer_times
        
        elif self.framework == 'pytorch':
            # PyTorch layer timing using hooks
            layer_times = {}
            
            def timing_hook(name):
                def hook(module, input, output):
                    if not hasattr(module, 'start_time'):
                        module.start_time = time.time()
                    else:
                        layer_times[name] = (time.time() - module.start_time) * 1000
                return hook
            
            # Register hooks
            handles = []
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    handle = module.register_forward_hook(timing_hook(name))
                    handles.append(handle)
            
            # Run inference
            with torch.no_grad():
                _ = self.model(input_data)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            return layer_times

# Usage example
def analyze_model_performance():
    """Comprehensive performance analysis"""
    
    # Load model
    model = tf.keras.models.load_model('baseline_model')
    profiler = ModelProfiler(model, 'tensorflow')
    
    # Create input data
    input_data = tf.random.normal((1, 224, 224, 3))
    
    # Profile the model
    profiler.profile_tensorflow_model(input_data)
    
    # Analyze layer performance
    layer_times = profiler.analyze_layer_performance(input_data)
    
    print("\nPer-Layer Performance Analysis:")
    sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
    for layer_name, time_ms in sorted_layers[:10]:
        print(f"{layer_name}: {time_ms:.2f} ms")
    
    return layer_times

if __name__ == "__main__":
    layer_performance = analyze_model_performance()
```

## Part 2: Model Optimization Techniques

### 2.1 Quantization Implementation

Quantization reduces model size and improves inference speed by using lower precision data types.

#### Post-Training Quantization (PTQ)

```python
import tensorflow as tf
import numpy as np
from typing import Generator

class QuantizationOptimizer:
    """Comprehensive quantization implementation"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
    
    def dynamic_range_quantization(self) -> bytes:
        """
        Dynamic range quantization - fastest and simplest method
        Weights: FP32 -> INT8, Activations: FP32
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Set supported ops for compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        tflite_model = converter.convert()
        
        # Save the quantized model
        with open('quantized_dynamic.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"Dynamic range quantized model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        return tflite_model
    
    def full_integer_quantization(self, representative_dataset: Generator) -> bytes:
        """
        Full integer quantization - maximum optimization
        Weights + Activations: FP32 -> INT8
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for calibration
        converter.representative_dataset = representative_dataset
        
        # Ensure only INT8 operations
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open('quantized_int8.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"INT8 quantized model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        return tflite_model
    
    def float16_quantization(self) -> bytes:
        """
        Float16 quantization - good balance between size and accuracy
        Weights: FP32 -> FP16
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open('quantized_fp16.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"FP16 quantized model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        return tflite_model
    
    def quantization_aware_training(self, train_dataset, validation_dataset, epochs=5):
        """
        Quantization Aware Training (QAT) - best accuracy retention
        """
        import tensorflow_model_optimization as tfmot
        
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(self.model)
        
        # Compile the quantization aware model
        q_aware_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train the quantization aware model
        q_aware_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3),
                tf.keras.callbacks.ReduceLROnPlateau(patience=2)
            ]
        )
        
        # Convert to quantized TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open('quantized_qat.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"QAT quantized model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        return tflite_model, q_aware_model

def create_representative_dataset(input_shape, num_samples=100):
    """Create representative dataset for calibration"""
    def representative_data_gen():
        for _ in range(num_samples):
            # Generate representative data (replace with actual data)
            data = np.random.random(input_shape).astype(np.float32)
            yield [data]
    
    return representative_data_gen

# PyTorch Quantization
class PyTorchQuantization:
    """PyTorch quantization implementation"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def dynamic_quantization(self):
        """Dynamic quantization for PyTorch"""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), 'pytorch_quantized_dynamic.pth')
        return quantized_model
    
    def static_quantization(self, calibration_loader):
        """Static quantization with calibration dataset"""
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate with representative data
        prepared_model.eval()
        with torch.no_grad():
            for data, _ in calibration_loader:
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), 'pytorch_quantized_static.pth')
        return quantized_model

# Usage example
def quantization_comparison():
    """Compare different quantization methods"""
    
    # Load baseline model
    quantizer = QuantizationOptimizer('baseline_model')
    
    # Create representative dataset
    input_shape = (1, 224, 224, 3)
    representative_dataset = create_representative_dataset(input_shape)
    
    results = {}
    
    # Test different quantization methods
    print("=== Quantization Comparison ===")
    
    # Dynamic range quantization
    dynamic_model = quantizer.dynamic_range_quantization()
    results['dynamic'] = len(dynamic_model)
    
    # Float16 quantization
    fp16_model = quantizer.float16_quantization()
    results['fp16'] = len(fp16_model)
    
    # Full integer quantization
    int8_model = quantizer.full_integer_quantization(representative_dataset)
    results['int8'] = len(int8_model)
    
    # Compare sizes
    print("\n=== Size Comparison ===")
    for method, size in results.items():
        size_mb = size / 1024 / 1024
        print(f"{method.upper()}: {size_mb:.2f} MB")
    
    return results

if __name__ == "__main__":
    quantization_results = quantization_comparison()
```

### 2.2 Model Pruning Implementation

Pruning removes redundant parameters to reduce model size and computation.

```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import torch
import torch.nn.utils.prune as prune

class ModelPruner:
    """Comprehensive model pruning implementation"""
    
    def __init__(self, model, framework='tensorflow'):
        self.model = model
        self.framework = framework
    
    def magnitude_based_pruning_tf(self, target_sparsity=0.5, epochs=10):
        """
        Magnitude-based pruning for TensorFlow
        Removes weights with smallest absolute values
        """
        
        # Define pruning schedule
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=epochs * 100  # Assuming 100 steps per epoch
            )
        }
        
        # Apply pruning to the model
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )
        
        # Compile the pruned model
        model_for_pruning.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        return model_for_pruning
    
    def structured_pruning_tf(self, target_sparsity=0.3):
        """
        Structured pruning - removes entire neurons/channels
        Better for hardware acceleration
        """
        
        def apply_structured_pruning(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                # Prune neurons in Dense layers
                original_units = layer.units
                new_units = int(original_units * (1 - target_sparsity))
                
                return tf.keras.layers.Dense(
                    new_units,
                    activation=layer.activation,
                    use_bias=layer.use_bias
                )
            
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # Prune channels in Conv2D layers
                original_filters = layer.filters
                new_filters = int(original_filters * (1 - target_sparsity))
                
                return tf.keras.layers.Conv2D(
                    new_filters,
                    layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    use_bias=layer.use_bias
                )
            
            return layer
        
        # Create new model with pruned structure
        pruned_model = tf.keras.Sequential([
            apply_structured_pruning(layer) for layer in self.model.layers
        ])
        
        return pruned_model

    def gradual_magnitude_pruning(self, train_dataset, val_dataset, target_sparsity=0.8):
        """
        Gradual magnitude-based pruning with training
        """
        
        # Pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='./logs/pruning'),
        ]
        
        # Apply pruning
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }
        
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )
        
        model_for_pruning.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train with pruning
        model_for_pruning.fit(
            train_dataset,
            epochs=10,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        
        # Strip pruning wrappers and export final model
        final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        
        return final_model

class PyTorchPruner:
    """PyTorch pruning implementation"""
    
    def __init__(self, model):
        self.model = model
    
    def unstructured_pruning(self, amount=0.3):
        """
        Unstructured magnitude-based pruning
        """
        
        # Apply pruning to all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
        
        return self.model
    
    def structured_pruning(self, amount=0.2):
        """
        Structured pruning - removes entire channels/neurons
        """
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Prune channels based on L2 norm
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=amount, 
                    n=2, 
                    dim=0  # Prune output channels
                )
            elif isinstance(module, torch.nn.Linear):
                # Prune neurons
                prune.ln_structured(
                    module, 
                    name='weight', 
                    amount=amount, 
                    n=2, 
                    dim=0
                )
        
        return self.model
    
    def global_pruning(self, amount=0.2):
        """
        Global pruning across all layers
        """
        
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        return self.model
    
    def calculate_sparsity(self):
        """Calculate overall model sparsity"""
        
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        sparsity = zero_params / total_params
        return sparsity

# Pruning analysis and comparison
def pruning_analysis():
    """Comprehensive pruning analysis"""
    
    print("=== Model Pruning Analysis ===")
    
    # Load baseline model
    model = tf.keras.models.load_model('baseline_model')
    pruner = ModelPruner(model, 'tensorflow')
    
    # Original model stats
    original_params = model.count_params()
    print(f"Original model parameters: {original_params:,}")
    
    # Test different pruning methods
    sparsity_levels = [0.3, 0.5, 0.7, 0.9]
    
    for sparsity in sparsity_levels:
        print(f"\n--- Sparsity Level: {sparsity*100}% ---")
        
        # Magnitude-based pruning
        pruned_model = pruner.magnitude_based_pruning_tf(target_sparsity=sparsity)
        
        # Structured pruning
        structured_model = pruner.structured_pruning_tf(target_sparsity=sparsity)
        structured_params = structured_model.count_params()
        
        reduction = (1 - structured_params / original_params) * 100
        print(f"Structured pruning - Parameters reduced by: {reduction:.1f}%")
        print(f"Remaining parameters: {structured_params:,}")

if __name__ == "__main__":
    pruning_analysis()
```

## Part 3: Hardware-Specific Optimization

### 3.1 GPU Optimization with TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import tensorflow as tf
from typing import Tuple, List

class TensorRTOptimizer:
    """TensorRT optimization for NVIDIA GPUs"""
    
    def __init__(self, onnx_model_path: str = None, precision: str = 'fp32'):
        self.onnx_model_path = onnx_model_path
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
    def build_engine_from_onnx(self, max_batch_size: int = 1, max_workspace_size: int = 1 << 30):
        """
        Build TensorRT engine from ONNX model
        """
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX file
        with open(self.onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        # Set precision
        if self.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif self.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
            # Note: INT8 requires calibration dataset
        
        # Build engine
        print("Building TensorRT engine... This may take a while.")
        self.engine = builder.build_engine(network, config)
        
        if self.engine is None:
            print("ERROR: Failed to build engine")
            return None
        
        print("TensorRT engine built successfully")
        return self.engine
    
    def save_engine(self, engine_path: str):
        """Save TensorRT engine to file"""
        if self.engine is None:
            print("No engine to save")
            return
        
        with open(engine_path, "wb") as f:
            f.write(self.engine.serialize())
        print(f"Engine saved to {engine_path}")
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        print(f"Engine loaded from {engine_path}")
        return self.engine
    
    def create_execution_context(self):
        """Create execution context for inference"""
        if self.engine is None:
            print("No engine available")
            return None
        
        self.context = self.engine.create_execution_context()
        return self.context
    
    def allocate_buffers(self):
        """Allocate GPU memory for input/output tensors"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference using TensorRT engine
        """
        if self.context is None:
            self.create_execution_context()
        
        inputs, outputs, bindings, stream = self.allocate_buffers()
        
        # Copy input data to host buffer
        np.copyto(inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Transfer output data from GPU
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        
        # Synchronize stream
        stream.synchronize()
        
        # Return output
        return outputs[0]['host']
    
    def benchmark_tensorrt(self, input_shape: Tuple, num_iterations: int = 1000):
        """Benchmark TensorRT inference performance"""
        
        if self.engine is None:
            print("No engine available for benchmarking")
            return
        
        # Create dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.inference(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.inference(dummy_input)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"TensorRT Benchmark Results:")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time
        }

def convert_pytorch_to_onnx(model, input_shape, onnx_path):
    """Convert PyTorch model to ONNX format"""
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to ONNX: {onnx_path}")

def convert_tensorflow_to_onnx(model_path, onnx_path):
    """Convert TensorFlow model to ONNX format"""
    
    import tf2onnx
    
    # Convert TensorFlow model to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model_path, opset=11)
    
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"TensorFlow model converted to ONNX: {onnx_path}")

# Usage example
def tensorrt_optimization_example():
    """Complete TensorRT optimization workflow"""
    
    print("=== TensorRT Optimization Example ===")
    
    # Step 1: Convert model to ONNX (example with dummy model)
    onnx_path = "model.onnx"
    
    # Step 2: Create TensorRT optimizer
    trt_optimizer = TensorRTOptimizer(onnx_path, precision='fp16')
    
    # Step 3: Build TensorRT engine
    engine = trt_optimizer.build_engine_from_onnx()
    
    if engine:
        # Step 4: Save engine for future use
        trt_optimizer.save_engine("model.trt")
        
        # Step 5: Benchmark performance
        input_shape = (1, 3, 224, 224)  # Batch, Channels, Height, Width
        results = trt_optimizer.benchmark_tensorrt(input_shape)
        
        return results
    
    return None

if __name__ == "__main__":
    tensorrt_results = tensorrt_optimization_example()
```

### 3.2 CPU Optimization with OpenVINO

```python
from openvino.inference_engine import IECore
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple

class OpenVINOOptimizer:
    """Intel OpenVINO optimization for CPU/VPU inference"""
    
    def __init__(self, model_xml: str, model_bin: str, device: str = 'CPU'):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device = device
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.input_blob = None
        self.output_blob = None
        
    def load_model(self):
        """Load OpenVINO IR model"""
        
        # Read the network
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        
        # Get input and output layer names
        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))
        
        # Load network to device
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)
        
        print(f"Model loaded on {self.device}")
        
        # Print model info
        input_shape = self.net.input_info[self.input_blob].input_data.shape
        output_shape = self.net.outputs[self.output_blob].shape
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")
        
        return self.exec_net
    
    def optimize_for_throughput(self):
        """Configure for maximum throughput"""
        
        if self.device == 'CPU':
            # CPU-specific optimizations
            self.ie.set_config({
                'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO',
                'CPU_BIND_THREAD': 'NUMA',
                'CPU_THREADS_NUM': '0'  # Use all available cores
            }, device_name='CPU')
        
        elif self.device == 'GPU':
            # GPU-specific optimizations
            self.ie.set_config({
                'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'
            }, device_name='GPU')
        
        print(f"Optimized for throughput on {self.device}")
    
    def optimize_for_latency(self):
        """Configure for minimum latency"""
        
        if self.device == 'CPU':
            # CPU latency optimizations
            self.ie.set_config({
                'CPU_THROUGHPUT_STREAMS': '1',
                'CPU_BIND_THREAD': 'YES'
            }, device_name='CPU')
        
        elif self.device == 'GPU':
            # GPU latency optimizations
            self.ie.set_config({
                'GPU_THROUGHPUT_STREAMS': '1'
            }, device_name='GPU')
        
        print(f"Optimized for latency on {self.device}")
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        
        if self.exec_net is None:
            raise ValueError("Model not loaded")
        
        # Prepare input
        input_dict = {self.input_blob: input_data}
        
        # Run inference
        result = self.exec_net.infer(inputs=input_dict)
        
        # Extract output
        output = result[self.output_blob]
        
        return output
    
    def benchmark_openvino(self, input_shape: Tuple, num_iterations: int = 1000):
        """Benchmark OpenVINO inference"""
        
        if self.exec_net is None:
            self.load_model()
        
        # Create dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.inference(dummy_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            self.inference(dummy_input)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"OpenVINO Benchmark Results ({self.device}):")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time,
            'device': self.device
        }
    
    def async_inference(self, input_data_list: List[np.ndarray]) -> List[np.ndarray]:
        """Asynchronous inference for better throughput"""
        
        if self.exec_net is None:
            raise ValueError("Model not loaded")
        
        # Get number of requests
        num_requests = len(input_data_list)
        
        # Create inference requests
        infer_requests = []
        for i in range(min(num_requests, 4)):  # Limit concurrent requests
            infer_requests.append(self.exec_net.create_infer_request())
        
        # Submit all requests
        results = []
        for i, input_data in enumerate(input_data_list):
            request_id = i % len(infer_requests)
            request = infer_requests[request_id]
            
            # Wait for previous request to complete if needed
            if i >= len(infer_requests):
                request.wait()
                results.append(request.output_blobs[self.output_blob].buffer.copy())
            
            # Start new inference
            request.async_infer({self.input_blob: input_data})
        
        # Wait for remaining requests
        for i in range(len(input_data_list) - len(infer_requests), len(input_data_list)):
            request_id = i % len(infer_requests)
            request = infer_requests[request_id]
            request.wait()
            results.append(request.output_blobs[self.output_blob].buffer.copy())
        
        return results

def convert_onnx_to_openvino(onnx_path: str, output_dir: str):
    """Convert ONNX model to OpenVINO IR format"""
    
    import subprocess
    
    # Model Optimizer command
    mo_command = [
        'mo',
        '--input_model', onnx_path,
        '--output_dir', output_dir,
        '--data_type', 'FP32',
        '--model_name', 'optimized_model'
    ]
    
    try:
        result = subprocess.run(mo_command, capture_output=True, text=True, check=True)
        print("Model conversion successful!")
        print(f"IR files saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Model conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

# Usage example
def openvino_optimization_example():
    """Complete OpenVINO optimization workflow"""
    
    print("=== OpenVINO Optimization Example ===")
    
    # Assume we have converted model files
    model_xml = "optimized_model.xml"
    model_bin = "optimized_model.bin"
    
    # Test different device configurations
    devices = ['CPU']  # Add 'GPU', 'MYRIAD' if available
    
    results = {}
    
    for device in devices:
        print(f"\n--- Testing {device} ---")
        
        # Create optimizer
        optimizer = OpenVINOOptimizer(model_xml, model_bin, device)
        
        # Load model
        optimizer.load_model()
        
        # Test throughput optimization
        optimizer.optimize_for_throughput()
        throughput_results = optimizer.benchmark_openvino((1, 3, 224, 224))
        
        # Test latency optimization
        optimizer.optimize_for_latency()
        latency_results = optimizer.benchmark_openvino((1, 3, 224, 224))
        
        results[device] = {
            'throughput': throughput_results,
            'latency': latency_results
        }
    
    return results

if __name__ == "__main__":
    openvino_results = openvino_optimization_example()
```

### 3.3 Mobile and Edge Optimization

```python
import tensorflow as tf
import numpy as np
import time
from typing import Dict, Any

class MobileOptimizer:
    """Optimization for mobile and edge devices"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
    
    def optimize_for_mobile(self, target_size_mb: float = 5.0):
        """
        Comprehensive mobile optimization pipeline
        """
        
        print("=== Mobile Optimization Pipeline ===")
        
        # Step 1: Model architecture optimization
        optimized_model = self._optimize_architecture()
        
        # Step 2: Quantization
        quantized_model = self._apply_mobile_quantization(optimized_model)
        
        # Step 3: Graph optimization
        final_model = self._optimize_graph(quantized_model)
        
        # Check final size
        model_size = self._get_model_size(final_model)
        print(f"Final model size: {model_size:.2f} MB")
        
        if model_size > target_size_mb:
            print(f"Warning: Model size ({model_size:.2f} MB) exceeds target ({target_size_mb} MB)")
        
        return final_model
    
    def _optimize_architecture(self):
        """Optimize model architecture for mobile"""
        
        # Replace expensive operations with mobile-friendly alternatives
        mobile_model = tf.keras.Sequential()
        
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Use depthwise separable convolutions
                mobile_model.add(tf.keras.layers.SeparableConv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation
                ))
            elif isinstance(layer, tf.keras.layers.Dense) and layer.units > 128:
                # Reduce dense layer size
                mobile_model.add(tf.keras.layers.Dense(
                    min(layer.units, 128),
                    activation=layer.activation
                ))
            else:
                # Keep other layers as is
                mobile_model.add(layer)
        
        return mobile_model
    
    def _apply_mobile_quantization(self, model):
        """Apply quantization optimized for mobile"""
        
        # Use dynamic range quantization for mobile
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Mobile-specific settings
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Optimize for size and latency
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save mobile-optimized model
        with open('mobile_optimized.tflite', 'wb') as f:
            f.write(tflite_model)
        
        return tflite_model
    
    def _optimize_graph(self, tflite_model):
        """Additional graph optimizations"""
        
        # Note: TensorFlow Lite already performs graph optimizations
        # This is where you'd add custom optimizations if needed
        
        return tflite_model
    
    def _get_model_size(self, tflite_model):
        """Get model size in MB"""
        return len(tflite_model) / (1024 * 1024)
    
    def benchmark_mobile_inference(self, tflite_model, input_shape, num_iterations=100):
        """Benchmark mobile inference performance"""
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"Mobile TFLite Benchmark:")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        print(f"Model size: {self._get_model_size(tflite_model):.2f} MB")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'model_size_mb': self._get_model_size(tflite_model)
        }

class EdgeTPUOptimizer:
    """Google Edge TPU optimization"""
    
    def __init__(self, tflite_model_path: str):
        self.tflite_model_path = tflite_model_path
    
    def compile_for_edge_tpu(self, output_path: str = 'model_edgetpu.tflite'):
        """
        Compile TensorFlow Lite model for Edge TPU
        Requires Edge TPU Compiler to be installed
        """
        
        import subprocess
        
        try:
            # Compile for Edge TPU
            cmd = ['edgetpu_compiler', self.tflite_model_path, '-o', 'output']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            print("Edge TPU compilation successful!")
            print(f"Compiled model saved as: {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"Edge TPU compilation failed: {e}")
            return None
        except FileNotFoundError:
            print("Edge TPU Compiler not found. Please install it first.")
            return None
    
    def benchmark_edge_tpu(self, compiled_model_path: str, input_shape, num_iterations=100):
        """Benchmark Edge TPU inference"""
        
        try:
            import tflite_runtime.interpreter as tflite
            
            # Create interpreter with Edge TPU delegate
            interpreter = tflite.Interpreter(
                model_path=compiled_model_path,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
            )
            
        except ImportError:
            print("TensorFlow Lite Runtime not available")
            return None
        except Exception as e:
            print(f"Edge TPU delegate not available: {e}")
            return None
        
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        print(f"Edge TPU Benchmark:")
        print(f"Average inference time: {avg_time * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'device': 'Edge TPU'
        }

# Usage example
def mobile_edge_optimization():
    """Complete mobile and edge optimization workflow"""
    
    print("=== Mobile and Edge Optimization ===")
    
    # Mobile optimization
    mobile_optimizer = MobileOptimizer('baseline_model')
    mobile_model = mobile_optimizer.optimize_for_mobile(target_size_mb=3.0)
    
    # Benchmark mobile performance
    mobile_results = mobile_optimizer.benchmark_mobile_inference(
        mobile_model, 
        (1, 224, 224, 3)
    )
    
    # Edge TPU optimization (if available)
    edge_optimizer = EdgeTPUOptimizer('mobile_optimized.tflite')
    compiled_model = edge_optimizer.compile_for_edge_tpu()
    
    if compiled_model:
        edge_results = edge_optimizer.benchmark_edge_tpu(
            compiled_model,
            (1, 224, 224, 3)
        )
        return {'mobile': mobile_results, 'edge_tpu': edge_results}
    
    return {'mobile': mobile_results}

if __name__ == "__main__":
    mobile_edge_results = mobile_edge_optimization()
```

## Part 4: Comprehensive Benchmarking and Analysis

### 4.1 Cross-Platform Performance Comparison

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from typing import Dict, List
import psutil

class HardwareBenchmark:
    """Comprehensive hardware benchmarking suite"""
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict:
        """Collect system information"""
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = [{'name': gpu.name, 'memory': gpu.memoryTotal} for gpu in gpus]
        except ImportError:
            gpu_info = []
        
        return {
            'cpu': {
                'model': psutil.cpu_freq(),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown'
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            },
            'gpu': gpu_info
        }
    
    def run_comprehensive_benchmark(self, model_variants: Dict) -> Dict:
        """
        Run comprehensive benchmark across all optimizations
        
        Args:
            model_variants: Dictionary of model name -> model/optimizer object
        """
        
        print("=== Comprehensive Hardware Benchmark ===")
        print(f"System Info: {json.dumps(self.system_info, indent=2)}")
        
        benchmark_results = {}
        input_shape = (1, 224, 224, 3)
        
        for variant_name, variant_info in model_variants.items():
            print(f"\n--- Benchmarking {variant_name} ---")
            
            try:
                # Run benchmark
                if 'benchmark_func' in variant_info:
                    results = variant_info['benchmark_func'](input_shape)
                else:
                    results = self._default_benchmark(variant_info['model'], input_shape)
                
                # Add metadata
                results['variant'] = variant_name
                results['optimization_type'] = variant_info.get('type', 'unknown')
                results['hardware_target'] = variant_info.get('hardware', 'cpu')
                
                # Memory profiling
                memory_usage = self._profile_memory_usage(variant_info['model'], input_shape)
                results.update(memory_usage)
                
                benchmark_results[variant_name] = results
                
            except Exception as e:
                print(f"Benchmark failed for {variant_name}: {e}")
                benchmark_results[variant_name] = {'error': str(e)}
        
        self.results = benchmark_results
        return benchmark_results
    
    def _default_benchmark(self, model, input_shape, num_iterations=100):
        """Default benchmarking method"""
        
        # Create dummy input
        if hasattr(model, 'predict'):  # TensorFlow/Keras model
            import tensorflow as tf
            dummy_input = tf.random.normal(input_shape)
            
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = model(dummy_input)
            end_time = time.time()
            
        else:  # Assume PyTorch model
            import torch
            dummy_input = torch.randn(input_shape)
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
            end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time
        }
    
    def _profile_memory_usage(self, model, input_shape):
        """Profile memory usage during inference"""
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run a few inferences to get peak memory
        if hasattr(model, 'predict'):  # TensorFlow/Keras
            import tensorflow as tf
            dummy_input = tf.random.normal(input_shape)
            for _ in range(5):
                _ = model(dummy_input)
        else:  # PyTorch
            import torch
            dummy_input = torch.randn(input_shape)
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy_input)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - baseline_memory
        }
    
    def generate_performance_report(self, save_path: str = 'benchmark_report.html'):
        """Generate comprehensive performance report"""
        
        if not self.results:
            print("No benchmark results available")
            return
        
        # Convert results to DataFrame
        df_data = []
        for variant, results in self.results.items():
            if 'error' not in results:
                df_data.append(results)
        
        df = pd.DataFrame(df_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Inference Time Comparison
        sns.barplot(data=df, x='variant', y='avg_time_ms', ax=axes[0,0])
        axes[0,0].set_title('Average Inference Time (ms)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput Comparison
        sns.barplot(data=df, x='variant', y='throughput_fps', ax=axes[0,1])
        axes[0,1].set_title('Throughput (FPS)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Memory Usage
        sns.barplot(data=df, x='variant', y='peak_memory_mb', ax=axes[1,0])
        axes[1,0].set_title('Peak Memory Usage (MB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Performance vs Memory Trade-off
        sns.scatterplot(data=df, x='peak_memory_mb', y='throughput_fps', 
                       hue='optimization_type', size='avg_time_ms', ax=axes[1,1])
        axes[1,1].set_title('Performance vs Memory Trade-off')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        
        # Generate HTML report
        html_report = self._generate_html_report(df)
        
        with open(save_path, 'w') as f:
            f.write(html_report)
        
        print(f"Performance report saved to: {save_path}")
        
        return df
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hardware-Accelerated Inference Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .best {{ background-color: #90EE90; }}
                .worst {{ background-color: #FFB6C1; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hardware-Accelerated Inference Benchmark Report</h1>
                <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>System Information</h2>
                <pre>{json.dumps(self.system_info, indent=2)}</pre>
            </div>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <ul>
                    <li><strong>Fastest Model:</strong> {df.loc[df['avg_time_ms'].idxmin(), 'variant']} 
                        ({df['avg_time_ms'].min():.2f} ms)</li>
                    <li><strong>Highest Throughput:</strong> {df.loc[df['throughput_fps'].idxmax(), 'variant']} 
                        ({df['throughput_fps'].max():.2f} FPS)</li>
                    <li><strong>Lowest Memory:</strong> {df.loc[df['peak_memory_mb'].idxmin(), 'variant']} 
                        ({df['peak_memory_mb'].min():.2f} MB)</li>
                    <li><strong>Models Tested:</strong> {len(df)}</li>
                </ul>
            </div>
            
            <h2>Detailed Results</h2>
            {df.to_html(classes='table', table_id='results_table')}
            
            <h2>Performance Visualization</h2>
            <img src="benchmark_results.png" alt="Benchmark Results" style="max-width: 100%;">
            
        </body>
        </html>
        """
        
        return html
    
    def export_results(self, format: str = 'json', filename: str = 'benchmark_results'):
        """Export results in various formats"""
        
        if format.lower() == 'json':
            with open(f'{filename}.json', 'w') as f:
                json.dump(self.results, f, indent=2)
        
        elif format.lower() == 'csv':
            df_data = []
            for variant, results in self.results.items():
                if 'error' not in results:
                    df_data.append(results)
            
            df = pd.DataFrame(df_data)
            df.to_csv(f'{filename}.csv', index=False)
        
        print(f"Results exported to: {filename}.{format}")

# Complete benchmark example
def run_complete_benchmark():
    """Run complete benchmark across all optimization techniques"""
    
    print("=== Complete Hardware Acceleration Benchmark ===")
    
    # Define model variants to test
    model_variants = {
        'baseline_tf': {
            'model': tf.keras.models.load_model('baseline_model'),
            'type': 'baseline',
            'hardware': 'cpu'
        },
        'quantized_dynamic': {
            'model': 'quantized_dynamic.tflite',
            'type': 'quantization',
            'hardware': 'cpu',
            'benchmark_func': lambda shape: benchmark_tflite_model('quantized_dynamic.tflite', shape)
        },
        'quantized_int8': {
            'model': 'quantized_int8.tflite',
            'type': 'quantization',
            'hardware': 'cpu',
            'benchmark_func': lambda shape: benchmark_tflite_model('quantized_int8.tflite', shape)
        },
        'mobile_optimized': {
            'model': 'mobile_optimized.tflite',
            'type': 'mobile',
            'hardware': 'mobile',
            'benchmark_func': lambda shape: benchmark_tflite_model('mobile_optimized.tflite', shape)
        }
    }
    
    # Run benchmark
    benchmark = HardwareBenchmark()
    results = benchmark.run_comprehensive_benchmark(model_variants)
    
    # Generate report
    df = benchmark.generate_performance_report()
    
    # Export results
    benchmark.export_results('json')
    benchmark.export_results('csv')
    
    return results, df

def benchmark_tflite_model(model_path: str, input_shape: tuple, num_iterations: int = 100):
    """Benchmark TensorFlow Lite model"""
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_iterations / total_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput_fps': throughput,
        'total_time_s': total_time
    }

if __name__ == "__main__":
    benchmark_results, benchmark_df = run_complete_benchmark()
```

## Part 5: Production Deployment

### 5.1 API Development and Containerization

```python
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import time
import logging
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceAPI:
    """Production-ready inference API"""
    
    def __init__(self, model_path: str, model_type: str = 'tflite'):
        self.model_path = model_path
        self.model_type = model_type
        self.model = self._load_model()
        self.request_count = 0
        self.total_inference_time = 0
        
    def _load_model(self):
        """Load the optimized model"""
        
        if self.model_type == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            return interpreter
        
        elif self.model_type == 'saved_model':
            return tf.saved_model.load(self.model_path)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run inference and return results"""
        
        start_time = time.time()
        
        try:
            if self.model_type == 'tflite':
                # TensorFlow Lite inference
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                
                self.model.set_tensor(input_details[0]['index'], input_data)
                self.model.invoke()
                
                output = self.model.get_tensor(output_details[0]['index'])
                
            else:
                # TensorFlow SavedModel inference
                output = self.model(input_data)
                
            inference_time = time.time() - start_time
            
            # Update statistics
            self.request_count += 1
            self.total_inference_time += inference_time
            
            return {
                'prediction': output.tolist(),
                'inference_time_ms': inference_time * 1000,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def get_stats(self) -> Dict[str, float]:
        """Get API statistics"""
        
        avg_time = (self.total_inference_time / self.request_count 
                   if self.request_count > 0 else 0)
        
        return {
            'total_requests': self.request_count,
            'average_inference_time_ms': avg_time * 1000,
            'total_inference_time_s': self.total_inference_time
        }

# Flask application
app = Flask(__name__)

# Initialize inference API
MODEL_PATH = os.getenv('MODEL_PATH', 'mobile_optimized.tflite')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'tflite')

inference_api = InferenceAPI(MODEL_PATH, MODEL_TYPE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    
    try:
        # Get input data
        data = request.get_json()
        
        if 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to numpy array
        input_array = np.array(data['input'], dtype=np.float32)
        
        # Validate input shape
        if len(input_array.shape) != 4:  # Assuming NHWC format
            return jsonify({'error': 'Invalid input shape'}), 400
        
        # Run inference
        result = inference_api.predict(input_array)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/stats', methods=['GET'])
def stats():
    """Statistics endpoint"""
    return jsonify(inference_api.get_stats())

@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    return jsonify({
        'model_path': MODEL_PATH,
        'model_type': MODEL_TYPE,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

**Dockerfile for containerization:**

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

**requirements.txt:**
```
Flask==2.3.3
tensorflow==2.13.0
numpy==1.24.3
gunicorn==21.2.0
```

**Docker Compose for deployment:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  inference-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MODEL_PATH=/app/models/mobile_optimized.tflite
      - MODEL_TYPE=tflite
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - inference-api
    restart: unless-stopped
```

## Project Summary and Key Learnings

### 🎯 Project Achievements

By completing this project, you have:

1. **Mastered Model Optimization Techniques**
   - Implemented quantization (dynamic, static, QAT)
   - Applied pruning (structured and unstructured)
   - Used knowledge distillation for model compression

2. **Deployed Across Multiple Hardware Platforms**
   - GPU optimization with TensorRT
   - CPU acceleration with OpenVINO
   - Mobile deployment with TensorFlow Lite
   - Edge computing with TPU

3. **Built Production-Ready Systems**
   - Created RESTful APIs for model serving
   - Implemented containerized deployment
   - Added monitoring and logging capabilities

4. **Performed Comprehensive Analysis**
   - Benchmarked performance across platforms
   - Analyzed trade-offs between speed, size, and accuracy
   - Generated detailed performance reports

### 📊 Performance Comparison Summary

| Optimization | Model Size | Inference Speed | Accuracy Retention | Hardware Target |
|-------------|------------|-----------------|-------------------|-----------------|
| **Baseline** | 100% | 1x | 100% | CPU/GPU |
| **Dynamic Quantization** | ~25% | 2-3x | 99% | CPU |
| **INT8 Quantization** | ~25% | 3-4x | 95-98% | CPU/Mobile |
| **Pruning (50%)** | ~50% | 1.5-2x | 97-99% | All |
| **Knowledge Distillation** | 10-30% | 2-5x | 90-95% | All |
| **TensorRT (FP16)** | ~50% | 3-5x | 99% | NVIDIA GPU |
| **Mobile Optimized** | ~15% | 2-4x | 92-96% | Mobile/Edge |

### 🔑 Key Technical Insights

1. **Quantization Trade-offs**
   - Dynamic quantization: Easy to implement, good for CPU
   - Static quantization: Better compression, requires calibration data
   - QAT: Best accuracy retention, requires training

2. **Hardware-Specific Optimizations**
   - GPUs benefit most from FP16 and batch processing
   - CPUs excel with quantization and vectorization
   - Mobile devices need aggressive compression

3. **Model Architecture Considerations**
   - Some operations don't quantize well (e.g., attention mechanisms)
   - Structured pruning is better for hardware acceleration
   - Depthwise separable convolutions are mobile-friendly

### 🚀 Next Steps and Extensions

1. **Advanced Techniques**
   - Neural Architecture Search (NAS) for optimal mobile architectures
   - Advanced pruning techniques (gradual magnitude, lottery ticket hypothesis)
   - Mixed-precision training and inference

2. **Additional Hardware Targets**
   - AMD ROCm for AMD GPUs
   - Intel Neural Compute Stick
   - ARM NN for ARM processors

3. **Production Enhancements**
   - A/B testing for model versions
   - Auto-scaling based on load
   - Model versioning and rollback capabilities

### 📚 Additional Resources

**Books:**
- "Efficient Deep Learning" by Gaurav Menghani
- "TinyML" by Pete Warden and Daniel Situnayake

**Papers:**
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"

**Tools and Frameworks:**
- TensorFlow Model Optimization Toolkit
- PyTorch Mobile
- ONNX Runtime
- Apache TVM

This project provides a solid foundation for deploying machine learning models efficiently across diverse hardware platforms. The techniques and insights gained here are directly applicable to real-world production systems.
