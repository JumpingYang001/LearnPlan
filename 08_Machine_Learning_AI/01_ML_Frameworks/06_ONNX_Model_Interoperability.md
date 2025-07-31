# ONNX and Model Interoperability

*Duration: 2 weeks*

## ONNX Format

### What is ONNX?

**Open Neural Network Exchange (ONNX)** is an open-source format for representing machine learning models. It enables interoperability between different ML frameworks, allowing you to train a model in one framework and deploy it in another.

#### Key Benefits of ONNX:
- 🔄 **Framework Independence**: Train in PyTorch, deploy in TensorFlow
- 🚀 **Deployment Flexibility**: Run models on various hardware and platforms
- 🎯 **Optimization**: Leverage framework-specific optimizations
- 📊 **Standardization**: Common format for model exchange

### ONNX Model Structure

An ONNX model consists of several key components:

```
ONNX Model
├── Model Metadata
│   ├── Model version
│   ├── Producer name
│   └── Domain
├── Graph
│   ├── Nodes (Operations)
│   ├── Inputs
│   ├── Outputs
│   └── Initializers (weights)
├── Value Info
│   └── Type information
└── Operator Sets
    └── Version information
```

#### Detailed Model Structure Example:

```python
import onnx
import numpy as np

# Load and inspect an ONNX model
model = onnx.load("model.onnx")

# Model metadata
print("Model Information:")
print(f"IR Version: {model.ir_version}")
print(f"Producer: {model.producer_name}")
print(f"Producer Version: {model.producer_version}")
print(f"Domain: {model.domain}")
print(f"Model Version: {model.model_version}")

# Graph structure
graph = model.graph
print(f"\nGraph Name: {graph.name}")

# Inputs
print("\nModel Inputs:")
for input_tensor in graph.input:
    print(f"  Name: {input_tensor.name}")
    print(f"  Type: {input_tensor.type}")
    # Get shape information
    if input_tensor.type.tensor_type.shape.dim:
        shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
        print(f"  Shape: {shape}")

# Outputs
print("\nModel Outputs:")
for output_tensor in graph.output:
    print(f"  Name: {output_tensor.name}")
    print(f"  Type: {output_tensor.type}")

# Nodes (operations)
print(f"\nNumber of nodes: {len(graph.node)}")
print("First 5 operations:")
for i, node in enumerate(graph.node[:5]):
    print(f"  {i}: {node.op_type} - {node.name}")
    print(f"      Inputs: {list(node.input)}")
    print(f"      Outputs: {list(node.output)}")
```

### Operator Sets

ONNX uses **operator sets (opsets)** to define the available operations and their semantics. Different opset versions support different operators and may have different behavior.

```python
# Check supported opsets
print("Supported Operator Sets:")
for opset in model.opset_import:
    print(f"  Domain: {opset.domain}")
    print(f"  Version: {opset.version}")

# Create model with specific opset
import onnx.helper as helper

# Define a simple model with specific opset version
def create_linear_model():
    # Input
    X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, 3])
    
    # Output
    Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, 1])
    
    # Weights
    W = helper.make_tensor('W', onnx.TensorProto.FLOAT, [3, 1], 
                          np.random.randn(3, 1).astype(np.float32).flatten())
    
    # Bias
    B = helper.make_tensor('B', onnx.TensorProto.FLOAT, [1], 
                          np.random.randn(1).astype(np.float32))
    
    # MatMul node
    matmul_node = helper.make_node('MatMul', ['X', 'W'], ['XW'])
    
    # Add node
    add_node = helper.make_node('Add', ['XW', 'B'], ['Y'])
    
    # Create graph
    graph = helper.make_graph([matmul_node, add_node], 'linear_model',
                             [X], [Y], [W, B])
    
    # Create model with specific opset
    model = helper.make_model(graph, producer_name='custom_producer')
    model.opset_import[0].version = 13  # Use opset version 13
    
    return model

# Create and save the model
linear_model = create_linear_model()
onnx.save(linear_model, 'linear_model.onnx')
print("Linear model created and saved!")
```

### Framework Independence

ONNX serves as a bridge between different ML frameworks:

```python
# Example: Cross-framework workflow
import torch
import tensorflow as tf
import onnxruntime as ort
import numpy as np

# 1. Train model in PyTorch
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Train PyTorch model
pytorch_model = SimpleNet()
# ... training code ...

# 2. Export to ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(pytorch_model, dummy_input, "pytorch_to_onnx.onnx",
                  export_params=True, opset_version=11,
                  input_names=['input'], output_names=['output'])

# 3. Load in ONNX Runtime (framework-agnostic)
ort_session = ort.InferenceSession("pytorch_to_onnx.onnx")

# 4. Run inference
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = ort_session.run(None, {'input': input_data})
print(f"ONNX Runtime output: {outputs[0]}")

# 5. Optionally convert to TensorFlow
# (using tf2onnx and onnx-tf libraries)
```

### Model Validation

Always validate your ONNX models to ensure correctness:

```python
import onnx

def validate_onnx_model(model_path):
    """Comprehensive ONNX model validation"""
    try:
        # Load model
        model = onnx.load(model_path)
        
        # Basic validation
        onnx.checker.check_model(model)
        print("✓ Model structure is valid")
        
        # Shape inference
        inferred_model = onnx.shape_inference.infer_shapes(model)
        print("✓ Shape inference successful")
        
        # Check for common issues
        issues = []
        
        # Check input/output names
        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        
        if not input_names:
            issues.append("No input names specified")
        if not output_names:
            issues.append("No output names specified")
        
        # Check for unsupported operations
        unsupported_ops = []
        for node in model.graph.node:
            if node.op_type in ['Custom', 'Unknown']:
                unsupported_ops.append(node.op_type)
        
        if unsupported_ops:
            issues.append(f"Unsupported operations: {unsupported_ops}")
        
        if issues:
            print("⚠️ Potential issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ No issues found")
        
        return True
        
    except onnx.checker.ValidationError as e:
        print(f"❌ Validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Validate model
validate_onnx_model("model.onnx")
```

## ONNX Model Conversion

### Framework to ONNX Conversion

Converting models between frameworks is one of ONNX's primary use cases. Each framework has its own conversion tools and considerations.

#### TensorFlow to ONNX

**Method 1: Using tf2onnx**
```python
import tensorflow as tf
import tf2onnx
import onnx

# Create a simple TensorFlow model
def create_tf_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Create and train model
tf_model = create_tf_model()

# Generate sample data for training
import numpy as np
X_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# Train the model
tf_model.fit(X_train, y_train, epochs=5, verbose=0)

# Convert to ONNX using tf2onnx
import tf2onnx.convert

def convert_tf_to_onnx(tf_model, model_path="tf_model.onnx"):
    # Method 1: Direct conversion from Keras model
    spec = (tf.TensorSpec((None, 10), tf.float32, name="input"),)
    output_path = model_path
    
    model_proto, _ = tf2onnx.convert.from_keras(tf_model, 
                                               input_signature=spec, 
                                               opset=13)
    
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    print(f"TensorFlow model converted to ONNX: {output_path}")
    return output_path

# Convert model
onnx_path = convert_tf_to_onnx(tf_model)

# Method 2: From SavedModel format
def convert_savedmodel_to_onnx():
    # Save TensorFlow model
    tf_model.save("saved_model")
    
    # Convert using command line tool
    import subprocess
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", "saved_model",
        "--output", "tf_savedmodel.onnx",
        "--opset", "13"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("SavedModel converted to ONNX successfully")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")

# Validate conversion
def validate_tf_onnx_conversion(tf_model, onnx_path):
    import onnxruntime as ort
    
    # Test data
    test_input = np.random.random((1, 10)).astype(np.float32)
    
    # TensorFlow prediction
    tf_output = tf_model.predict(test_input, verbose=0)
    
    # ONNX prediction
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'input': test_input})[0]
    
    # Compare outputs
    diff = np.abs(tf_output - onnx_output).max()
    print(f"Max difference between TF and ONNX: {diff}")
    
    if diff < 1e-5:
        print("✓ Conversion validated successfully")
    else:
        print("⚠️ Significant difference detected")

validate_tf_onnx_conversion(tf_model, onnx_path)
```

#### PyTorch to ONNX

```python
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Define a custom PyTorch model
class CustomNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def convert_pytorch_to_onnx():
    # Create model
    model = CustomNet()
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Model input
        "pytorch_custom.onnx",          # Output file
        export_params=True,             # Store trained weights
        opset_version=11,              # ONNX version
        do_constant_folding=True,       # Constant folding optimization
        input_names=['input'],          # Input names
        output_names=['output'],        # Output names
        dynamic_axes={                  # Dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print("PyTorch model exported to ONNX successfully")

# Advanced PyTorch to ONNX with preprocessing
class ModelWithPreprocessing(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Add preprocessing layers
        self.normalize = lambda x: (x - 0.485) / 0.229  # Example normalization
    
    def forward(self, x):
        x = self.normalize(x)
        return self.base_model(x)

def convert_with_preprocessing():
    # Load pretrained model
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    
    # Wrap with preprocessing
    model_with_prep = ModelWithPreprocessing(resnet)
    
    # Export
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model_with_prep,
        dummy_input,
        "resnet18_with_preprocessing.onnx",
        export_params=True,
        opset_version=11,
        input_names=['image'],
        output_names=['classification'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'classification': {0: 'batch_size'}
        }
    )

# Handle custom operations
class CustomOperation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Custom operation implementation
        return input * 2 + 1
    
    @staticmethod
    def symbolic(g, input):
        # Define ONNX symbolic representation
        two = g.op("Constant", value_t=torch.tensor(2.0))
        one = g.op("Constant", value_t=torch.tensor(1.0))
        mul = g.op("Mul", input, two)
        return g.op("Add", mul, one)

# Register custom operation
torch.onnx.register_custom_op_symbolic(
    "custom::custom_op", CustomOperation.symbolic, 11
)

convert_pytorch_to_onnx()
```

#### ONNX to Other Formats

```python
# ONNX to TensorFlow Lite
def convert_onnx_to_tflite(onnx_path, tflite_path):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("temp_tf_model")
    
    # Convert TensorFlow to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Enable quantization
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save TensorFlow Lite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"ONNX model converted to TensorFlow Lite: {tflite_path}")

def representative_dataset_gen():
    """Generate representative dataset for quantization"""
    for _ in range(100):
        yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

# ONNX to Core ML (iOS deployment)
def convert_onnx_to_coreml(onnx_path, coreml_path):
    try:
        from onnx_coreml import convert
        
        # Convert ONNX to Core ML
        coreml_model = convert(
            model=onnx_path,
            minimum_ios_deployment_target='13.0',
            image_input_names=['input'],
            preprocessing_args={
                'image_scale': 1.0/255.0,
                'red_bias': -0.485/0.229,
                'green_bias': -0.456/0.224,
                'blue_bias': -0.406/0.225
            }
        )
        
        # Save Core ML model
        coreml_model.save(coreml_path)
        print(f"ONNX model converted to Core ML: {coreml_path}")
        
    except ImportError:
        print("onnx-coreml not installed. Install with: pip install onnx-coreml")

# ONNX to OpenVINO (Intel optimization)
def convert_onnx_to_openvino(onnx_path, output_dir):
    import subprocess
    
    cmd = [
        "mo",  # Model Optimizer
        "--input_model", onnx_path,
        "--output_dir", output_dir,
        "--data_type", "FP16",  # Half precision
        "--compress_to_fp16"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"ONNX model converted to OpenVINO IR: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"OpenVINO conversion failed: {e}")
    except FileNotFoundError:
        print("OpenVINO Model Optimizer not found. Install OpenVINO toolkit.")
```

### Validation and Verification

```python
import onnx
import onnxruntime as ort
import numpy as np

def comprehensive_model_validation(original_model, onnx_path, framework='pytorch'):
    """Comprehensive validation of ONNX conversion"""
    
    print("Starting comprehensive ONNX model validation...")
    
    # 1. Load ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        return False
    
    # 2. Create ONNX Runtime session
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print("✓ ONNX Runtime can load the model")
    except Exception as e:
        print(f"❌ ONNX Runtime failed to load model: {e}")
        return False
    
    # 3. Test with multiple inputs
    input_info = ort_session.get_inputs()[0]
    input_shape = input_info.shape
    input_name = input_info.name
    
    # Handle dynamic shapes
    test_shapes = []
    if any(dim is None or isinstance(dim, str) for dim in input_shape):
        # Dynamic shape - test with different sizes
        static_shape = [1 if (dim is None or isinstance(dim, str)) else dim 
                       for dim in input_shape]
        test_shapes = [static_shape, 
                      [2] + static_shape[1:] if len(static_shape) > 1 else [2]]
    else:
        test_shapes = [input_shape]
    
    for test_shape in test_shapes:
        print(f"Testing with shape: {test_shape}")
        
        # Generate test input
        test_input = np.random.randn(*test_shape).astype(np.float32)
        
        try:
            # Original model prediction
            if framework == 'pytorch':
                import torch
                original_model.eval()
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input)
                    original_output = original_model(torch_input).numpy()
            elif framework == 'tensorflow':
                original_output = original_model.predict(test_input, verbose=0)
            
            # ONNX model prediction
            onnx_outputs = ort_session.run(None, {input_name: test_input})
            onnx_output = onnx_outputs[0]
            
            # Compare outputs
            diff = np.abs(original_output - onnx_output).max()
            relative_diff = diff / (np.abs(original_output).max() + 1e-8)
            
            print(f"  Max absolute difference: {diff:.2e}")
            print(f"  Max relative difference: {relative_diff:.2e}")
            
            if diff < 1e-4:
                print("  ✓ Outputs match within tolerance")
            else:
                print("  ⚠️ Significant difference detected")
                
        except Exception as e:
            print(f"  ❌ Error during inference: {e}")
            return False
    
    # 4. Performance comparison
    print("\nPerformance Comparison:")
    test_input = np.random.randn(*test_shapes[0]).astype(np.float32)
    
    # Benchmark original model
    import time
    
    if framework == 'pytorch':
        torch_input = torch.from_numpy(test_input)
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = original_model(torch_input)
        
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = original_model(torch_input)
        original_time = (time.time() - start_time) / 100
    
    # Benchmark ONNX model
    # Warmup
    for _ in range(10):
        _ = ort_session.run(None, {input_name: test_input})
    
    start_time = time.time()
    for _ in range(100):
        _ = ort_session.run(None, {input_name: test_input})
    onnx_time = (time.time() - start_time) / 100
    
    print(f"Original model avg time: {original_time*1000:.2f} ms")
    print(f"ONNX model avg time: {onnx_time*1000:.2f} ms")
    print(f"Speedup: {original_time/onnx_time:.2f}x")
    
    return True

# Example usage
# comprehensive_model_validation(pytorch_model, "model.onnx", "pytorch")
```

## ONNX Runtime

### Architecture Overview

ONNX Runtime is a high-performance inference engine for ONNX models. It provides optimized execution across different hardware platforms and offers APIs for multiple programming languages.

```
ONNX Runtime Architecture

┌─────────────────────────────────────────────────────────┐
│                    Client APIs                          │
│  Python │ C++ │ C# │ Java │ JavaScript │ C │ WinRT      │
├─────────────────────────────────────────────────────────┤
│                 Session & Graph                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Session   │  │ Graph Opt.   │  │   Memory Mgmt   │ │
│  │ Management  │  │ & Transform  │  │   & Allocation  │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                Execution Providers                      │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │ CPU  │ │ CUDA │ │ TRT  │ │ DML  │ │ OV   │ │ ACL  │  │
│ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │
├─────────────────────────────────────────────────────────┤
│                    Hardware                             │
│     x86/x64    │    ARM    │    GPU    │    NPU         │
└─────────────────────────────────────────────────────────┘
```

### Basic ONNX Runtime Usage

```python
import onnxruntime as ort
import numpy as np

# Basic inference session
def basic_inference_example():
    # Create inference session
    session = ort.InferenceSession("model.onnx")
    
    # Get input/output metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    
    print(f"Input: {input_name}, Shape: {input_shape}, Type: {input_type}")
    print(f"Output: {output_name}, Shape: {output_shape}")
    
    # Prepare input data
    # Handle dynamic shapes
    if input_shape[0] == 'batch_size' or input_shape[0] is None:
        actual_shape = [1] + input_shape[1:]
    else:
        actual_shape = input_shape
    
    input_data = np.random.randn(*actual_shape).astype(np.float32)
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output: {outputs[0]}")
    
    return outputs[0]

basic_inference_example()
```

### Execution Providers

ONNX Runtime supports multiple execution providers for different hardware acceleration:

```python
# List available execution providers
def list_available_providers():
    providers = ort.get_available_providers()
    print("Available Execution Providers:")
    for provider in providers:
        print(f"  - {provider}")
    return providers

# Configure execution providers
def create_optimized_session(model_path, use_gpu=True):
    providers = []
    
    if use_gpu:
        # GPU providers (in order of preference)
        available_providers = ort.get_available_providers()
        
        # NVIDIA TensorRT (highest performance for NVIDIA GPUs)
        if 'TensorrtExecutionProvider' in available_providers:
            providers.append(('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 1 << 30,  # 1GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }))
        
        # NVIDIA CUDA
        if 'CUDAExecutionProvider' in available_providers:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        
        # DirectML (Windows)
        if 'DmlExecutionProvider' in available_providers:
            providers.append(('DmlExecutionProvider', {
                'device_id': 0,
            }))
        
        # OpenVINO (Intel)
        if 'OpenVINOExecutionProvider' in available_providers:
            providers.append(('OpenVINOExecutionProvider', {
                'device_type': 'GPU_FP16',  # or 'CPU_FP32'
                'precision': 'FP16'
            }))
    
    # Always include CPU as fallback
    providers.append('CPUExecutionProvider')
    
    # Create session with specified providers
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # Error level
    
    session = ort.InferenceSession(
        model_path, 
        sess_options=session_options,
        providers=providers
    )
    
    print(f"Using execution providers: {session.get_providers()}")
    return session

# Example: GPU-optimized session
optimized_session = create_optimized_session("model.onnx", use_gpu=True)
```

### Performance Optimization

```python
import onnxruntime as ort
import time
import numpy as np

class ONNXOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.optimized_session = None
    
    def create_baseline_session(self):
        """Create baseline session with default settings"""
        self.session = ort.InferenceSession(self.model_path)
        return self.session
    
    def create_optimized_session(self):
        """Create optimized session with performance tuning"""
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        
        # Graph optimization level
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        
        # Enable CPU memory arena
        sess_options.enable_cpu_mem_arena = True
        
        # Parallel execution
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Set thread count (usually number of cores)
        import os
        sess_options.intra_op_num_threads = os.cpu_count()
        sess_options.inter_op_num_threads = 1  # For most models
        
        # Optimization for specific scenarios
        sess_options.add_session_config_entry('session.set_denormal_as_zero', '1')
        
        # Create session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.optimized_session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        return self.optimized_session
    
    def benchmark_sessions(self, num_runs=100):
        """Benchmark baseline vs optimized sessions"""
        
        # Get input info
        input_info = self.session.get_inputs()[0]
        input_shape = input_info.shape
        input_name = input_info.name
        
        # Handle dynamic shapes
        if any(dim is None or isinstance(dim, str) for dim in input_shape):
            actual_shape = [1] + [d if isinstance(d, int) else 224 for d in input_shape[1:]]
        else:
            actual_shape = input_shape
        
        # Generate test data
        test_input = np.random.randn(*actual_shape).astype(np.float32)
        
        # Benchmark baseline
        baseline_times = []
        for _ in range(10):  # Warmup
            _ = self.session.run(None, {input_name: test_input})
        
        for _ in range(num_runs):
            start = time.time()
            _ = self.session.run(None, {input_name: test_input})
            baseline_times.append(time.time() - start)
        
        # Benchmark optimized
        optimized_times = []
        for _ in range(10):  # Warmup
            _ = self.optimized_session.run(None, {input_name: test_input})
        
        for _ in range(num_runs):
            start = time.time()
            _ = self.optimized_session.run(None, {input_name: test_input})
            optimized_times.append(time.time() - start)
        
        # Calculate statistics
        baseline_avg = np.mean(baseline_times) * 1000  # ms
        baseline_std = np.std(baseline_times) * 1000
        
        optimized_avg = np.mean(optimized_times) * 1000  # ms
        optimized_std = np.std(optimized_times) * 1000
        
        speedup = baseline_avg / optimized_avg
        
        print(f"Baseline:  {baseline_avg:.2f} ± {baseline_std:.2f} ms")
        print(f"Optimized: {optimized_avg:.2f} ± {optimized_std:.2f} ms")
        print(f"Speedup:   {speedup:.2f}x")
        
        return {
            'baseline': {'mean': baseline_avg, 'std': baseline_std},
            'optimized': {'mean': optimized_avg, 'std': optimized_std},
            'speedup': speedup
        }

# Usage example
# optimizer = ONNXOptimizer("model.onnx")
# optimizer.create_baseline_session()
# optimizer.create_optimized_session()
# results = optimizer.benchmark_sessions()
```

### C++ API Usage

```cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <memory>

class ONNXInference {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

public:
    ONNXInference(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference") {
        
        // Configure session options
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Add execution providers
        #ifdef USE_CUDA
        session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
        #endif
        
        // Create session
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        // Get input/output info
        initializeIO();
    }
    
    void initializeIO() {
        // Get input info
        size_t num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = session->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            input_names.push_back(input_name.get());
            
            auto input_info = session->GetInputTypeInfo(i);
            auto tensor_info = input_info.GetTensorTypeAndShapeInfo();
            input_shapes.push_back(tensor_info.GetShape());
        }
        
        // Get output info
        size_t num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = session->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
            output_names.push_back(output_name.get());
            
            auto output_info = session->GetOutputTypeInfo(i);
            auto tensor_info = output_info.GetTensorTypeAndShapeInfo();
            output_shapes.push_back(tensor_info.GetShape());
        }
    }
    
    std::vector<Ort::Value> inference(const std::vector<float>& input_data) {
        // Create memory info
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create input tensor
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shapes[0].data(),
            input_shapes[0].size()
        ));
        
        // Convert input names to const char*
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names.size()
        );
        
        return output_tensors;
    }
    
    void printModelInfo() {
        std::cout << "Model Information:" << std::endl;
        std::cout << "Inputs:" << std::endl;
        for (size_t i = 0; i < input_names.size(); ++i) {
            std::cout << "  " << input_names[i] << ": [";
            for (size_t j = 0; j < input_shapes[i].size(); ++j) {
                std::cout << input_shapes[i][j];
                if (j < input_shapes[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Outputs:" << std::endl;
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::cout << "  " << output_names[i] << ": [";
            for (size_t j = 0; j < output_shapes[i].size(); ++j) {
                std::cout << output_shapes[i][j];
                if (j < output_shapes[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
};

// Usage example
int main() {
    try {
        ONNXInference inferencer("model.onnx");
        inferencer.printModelInfo();
        
        // Create sample input data
        std::vector<float> input_data(224 * 224 * 3, 1.0f);  // Example for image input
        
        // Run inference
        auto outputs = inferencer.inference(input_data);
        
        // Process outputs
        float* output_data = outputs[0].GetTensorMutableData<float>();
        size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::cout << "Output size: " << output_size << std::endl;
        std::cout << "First 5 values: ";
        for (size_t i = 0; i < std::min(size_t(5), output_size); ++i) {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Advanced Features

```python
# Custom operators
def register_custom_op():
    """Example of registering custom operators"""
    import onnxruntime as ort
    
    # This would be implemented in C++ and compiled as a shared library
    # custom_op_library = "path/to/custom_ops.so"
    # session_options = ort.SessionOptions()
    # session_options.register_custom_ops_library(custom_op_library)
    
    pass

# Model optimization
def optimize_model_for_inference():
    """Optimize ONNX model for better inference performance"""
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    import onnx
    
    # Load model
    model = onnx.load("original_model.onnx")
    
    # Shape inference
    model_with_shapes = SymbolicShapeInference.infer_shapes(model)
    
    # Save optimized model
    onnx.save(model_with_shapes, "optimized_model.onnx")
    
    return "optimized_model.onnx"

# Quantization
def quantize_model():
    """Quantize ONNX model for faster inference and smaller size"""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    model_input = "model_fp32.onnx"
    model_output = "model_int8.onnx"
    
    quantize_dynamic(
        model_input,
        model_output,
        weight_type=QuantType.QInt8
    )
    
    print(f"Model quantized: {model_output}")
    return model_output
```

## Model Interoperability

### Framework-Specific Challenges

Model interoperability between different ML frameworks presents unique challenges that developers must understand and address.

#### Data Format and Preprocessing Differences

```python
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import cv2

class FrameworkCompatibilityHandler:
    """Handle common compatibility issues between frameworks"""
    
    def __init__(self):
        self.preprocessing_pipelines = {
            'pytorch': self.pytorch_preprocessing,
            'tensorflow': self.tensorflow_preprocessing,
            'onnx': self.onnx_preprocessing
        }
    
    def pytorch_preprocessing(self, image_path):
        """PyTorch-style preprocessing (CHW format)"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to CHW and [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.numpy()  # Shape: (1, 3, 224, 224)
    
    def tensorflow_preprocessing(self, image_path):
        """TensorFlow-style preprocessing (HWC format)"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
        
        # TensorFlow normalization (different from PyTorch)
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image = tf.expand_dims(image, 0)  # Add batch dimension
        return image.numpy()  # Shape: (1, 224, 224, 3)
    
    def onnx_preprocessing(self, image_path):
        """ONNX-compatible preprocessing"""
        # Usually follows the original training framework's format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Normalize (using ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert HWC to CHW (channel first)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)  # Add batch dimension
        
        return image  # Shape: (1, 3, 224, 224)
    
    def convert_tensor_format(self, tensor, from_format, to_format):
        """Convert tensor between different formats"""
        if from_format == to_format:
            return tensor
        
        if from_format == 'NHWC' and to_format == 'NCHW':
            # TensorFlow to PyTorch format
            return np.transpose(tensor, (0, 3, 1, 2))
        elif from_format == 'NCHW' and to_format == 'NHWC':
            # PyTorch to TensorFlow format
            return np.transpose(tensor, (0, 2, 3, 1))
        else:
            raise ValueError(f"Conversion from {from_format} to {to_format} not supported")

# Example usage
handler = FrameworkCompatibilityHandler()

# Test preprocessing differences
image_path = "test_image.jpg"
pytorch_input = handler.pytorch_preprocessing(image_path)
tf_input = handler.tensorflow_preprocessing(image_path)
onnx_input = handler.onnx_preprocessing(image_path)

print(f"PyTorch input shape: {pytorch_input.shape}")  # (1, 3, 224, 224)
print(f"TensorFlow input shape: {tf_input.shape}")    # (1, 224, 224, 3)
print(f"ONNX input shape: {onnx_input.shape}")        # (1, 3, 224, 224)
```

#### Data Type Mismatches

```python
import numpy as np
import torch
import tensorflow as tf

class DataTypeHandler:
    """Handle data type conversions between frameworks"""
    
    @staticmethod
    def get_framework_dtype_mapping():
        """Get mapping between framework data types"""
        return {
            'float32': {
                'numpy': np.float32,
                'torch': torch.float32,
                'tensorflow': tf.float32
            },
            'float16': {
                'numpy': np.float16,
                'torch': torch.float16,
                'tensorflow': tf.float16
            },
            'int32': {
                'numpy': np.int32,
                'torch': torch.int32,
                'tensorflow': tf.int32
            },
            'int64': {
                'numpy': np.int64,
                'torch': torch.int64,
                'tensorflow': tf.int64
            }
        }
    
    @staticmethod
    def convert_dtype(data, target_framework, target_dtype='float32'):
        """Convert data type for target framework"""
        dtype_mapping = DataTypeHandler.get_framework_dtype_mapping()
        target_type = dtype_mapping[target_dtype][target_framework]
        
        if target_framework == 'numpy':
            return data.astype(target_type)
        elif target_framework == 'torch':
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).type(target_type)
            else:
                return data.type(target_type)
        elif target_framework == 'tensorflow':
            return tf.cast(data, target_type)
    
    @staticmethod
    def check_precision_loss(original_data, converted_data, tolerance=1e-6):
        """Check for precision loss during conversion"""
        if isinstance(converted_data, torch.Tensor):
            converted_data = converted_data.numpy()
        elif hasattr(converted_data, 'numpy'):
            converted_data = converted_data.numpy()
        
        if isinstance(original_data, torch.Tensor):
            original_data = original_data.numpy()
        elif hasattr(original_data, 'numpy'):
            original_data = original_data.numpy()
        
        max_diff = np.abs(original_data - converted_data).max()
        relative_diff = max_diff / (np.abs(original_data).max() + 1e-8)
        
        print(f"Max absolute difference: {max_diff}")
        print(f"Max relative difference: {relative_diff}")
        
        if max_diff > tolerance:
            print("⚠️ Significant precision loss detected!")
            return False
        else:
            print("✓ Conversion successful within tolerance")
            return True

# Example: Handle precision issues
original_data = np.random.randn(1000).astype(np.float32)

# Convert to different precisions
fp16_data = DataTypeHandler.convert_dtype(original_data, 'numpy', 'float16')
int32_data = DataTypeHandler.convert_dtype(original_data, 'numpy', 'int32')

# Check precision loss
print("Float32 to Float16 conversion:")
DataTypeHandler.check_precision_loss(original_data, fp16_data)

print("\nFloat32 to Int32 conversion:")
DataTypeHandler.check_precision_loss(original_data, int32_data)
```

### Common Conversion Issues

#### 1. Operator Support Differences

```python
import onnx
import onnxruntime as ort

class OperatorCompatibilityChecker:
    """Check and handle operator compatibility issues"""
    
    def __init__(self):
        self.unsupported_ops = {
            'pytorch_to_onnx': [
                'aten::upsample_bilinear2d',  # Use F.interpolate instead
                'aten::adaptive_avg_pool2d',  # Limited support
                'prim::PythonOp'              # Custom Python operations
            ],
            'tensorflow_to_onnx': [
                'tf.py_function',             # Python functions
                'tf.raw_ops',                 # Low-level operations
                'tf.custom_gradient'          # Custom gradients
            ]
        }
        
        self.workarounds = {
            'aten::upsample_bilinear2d': self.fix_upsample_bilinear2d,
            'aten::adaptive_avg_pool2d': self.fix_adaptive_avg_pool2d
        }
    
    def check_model_compatibility(self, onnx_model_path):
        """Check ONNX model for compatibility issues"""
        model = onnx.load(onnx_model_path)
        
        issues = []
        for node in model.graph.node:
            if node.op_type in ['Loop', 'If', 'Scan']:
                issues.append(f"Control flow operator '{node.op_type}' may have limited support")
            
            # Check for custom operators
            if '::' in node.op_type:
                issues.append(f"Custom operator '{node.op_type}' detected")
        
        return issues
    
    def fix_upsample_bilinear2d(self, model_path):
        """Fix upsample_bilinear2d compatibility issues"""
        print("Fixing upsample_bilinear2d operator...")
        # Implementation would involve graph transformation
        # This is a simplified example
        
        model = onnx.load(model_path)
        
        for node in model.graph.node:
            if node.op_type == 'Upsample':
                # Ensure mode is set correctly
                for attr in node.attribute:
                    if attr.name == 'mode':
                        attr.s = b'linear'
        
        onnx.save(model, model_path.replace('.onnx', '_fixed.onnx'))
        return model_path.replace('.onnx', '_fixed.onnx')
    
    def fix_adaptive_avg_pool2d(self, model_path):
        """Replace adaptive pooling with regular pooling"""
        print("Replacing AdaptiveAvgPool2d with GlobalAveragePool...")
        
        model = onnx.load(model_path)
        
        for node in model.graph.node:
            if node.op_type == 'AveragePool' and 'adaptive' in node.name.lower():
                # Replace with GlobalAveragePool
                node.op_type = 'GlobalAveragePool'
                # Remove kernel size and stride attributes
                node.attribute[:] = [attr for attr in node.attribute 
                                   if attr.name not in ['kernel_shape', 'strides']]
        
        onnx.save(model, model_path.replace('.onnx', '_fixed.onnx'))
        return model_path.replace('.onnx', '_fixed.onnx')

# Usage example
checker = OperatorCompatibilityChecker()
issues = checker.check_model_compatibility("model.onnx")
for issue in issues:
    print(f"⚠️ {issue}")
```

#### 2. Dynamic Shape Handling

```python
import onnx
import onnx.helper as helper

class DynamicShapeHandler:
    """Handle dynamic shape conversion issues"""
    
    @staticmethod
    def make_dynamic_shape_compatible(model_path, output_path=None):
        """Convert static shapes to dynamic shapes"""
        if output_path is None:
            output_path = model_path.replace('.onnx', '_dynamic.onnx')
        
        model = onnx.load(model_path)
        
        # Update input shapes to be dynamic
        for input_tensor in model.graph.input:
            if input_tensor.type.tensor_type.shape.dim:
                # Make batch size dynamic
                input_tensor.type.tensor_type.shape.dim[0].dim_param = "batch_size"
        
        # Update output shapes to be dynamic
        for output_tensor in model.graph.output:
            if output_tensor.type.tensor_type.shape.dim:
                # Make batch size dynamic
                output_tensor.type.tensor_type.shape.dim[0].dim_param = "batch_size"
        
        onnx.save(model, output_path)
        print(f"Dynamic shape model saved: {output_path}")
        return output_path
    
    @staticmethod
    def test_dynamic_shapes(model_path, test_shapes):
        """Test model with different input shapes"""
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        for shape in test_shapes:
            try:
                test_input = np.random.randn(*shape).astype(np.float32)
                outputs = session.run(None, {input_name: test_input})
                print(f"✓ Shape {shape}: Output shape {outputs[0].shape}")
            except Exception as e:
                print(f"❌ Shape {shape}: Error - {e}")

# Example usage
handler = DynamicShapeHandler()
dynamic_model = handler.make_dynamic_shape_compatible("static_model.onnx")

# Test with different batch sizes
test_shapes = [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)]
handler.test_dynamic_shapes(dynamic_model, test_shapes)
```

### Custom Operation Handling

```python
import onnx
import onnx.helper as helper
import numpy as np

class CustomOperationHandler:
    """Handle custom operations in ONNX models"""
    
    @staticmethod
    def register_custom_operation():
        """Example of how to register custom operations"""
        
        # Define custom operation schema
        from onnx import defs
        
        # This would typically be done in C++ for performance
        custom_op_schema = """
        def custom_relu6(input):
            return np.clip(input, 0, 6)
        """
        
        print("Custom operation registered (example)")
    
    @staticmethod
    def replace_custom_ops_with_standard(model_path, output_path=None):
        """Replace custom operations with standard ONNX operations"""
        if output_path is None:
            output_path = model_path.replace('.onnx', '_standard.onnx')
        
        model = onnx.load(model_path)
        graph = model.graph
        
        nodes_to_remove = []
        nodes_to_add = []
        
        for i, node in enumerate(graph.node):
            if node.op_type == 'CustomReLU6':
                # Replace CustomReLU6 with Clip operation
                clip_node = helper.make_node(
                    'Clip',
                    inputs=node.input,
                    outputs=node.output,
                    min=0.0,
                    max=6.0
                )
                nodes_to_remove.append(i)
                nodes_to_add.append(clip_node)
            
            elif node.op_type == 'CustomBatchNorm':
                # Replace with standard BatchNormalization
                bn_node = helper.make_node(
                    'BatchNormalization',
                    inputs=node.input,
                    outputs=node.output,
                    epsilon=1e-5,
                    momentum=0.9
                )
                nodes_to_remove.append(i)
                nodes_to_add.append(bn_node)
        
        # Remove old nodes (in reverse order to maintain indices)
        for idx in sorted(nodes_to_remove, reverse=True):
            del graph.node[idx]
        
        # Add new nodes
        graph.node.extend(nodes_to_add)
        
        onnx.save(model, output_path)
        print(f"Standard operations model saved: {output_path}")
        return output_path
    
    @staticmethod
    def decompose_complex_operations(model_path, output_path=None):
        """Decompose complex operations into simpler ones"""
        if output_path is None:
            output_path = model_path.replace('.onnx', '_decomposed.onnx')
        
        model = onnx.load(model_path)
        graph = model.graph
        
        # Example: Decompose GELU into simpler operations
        # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        
        nodes_to_replace = []
        for i, node in enumerate(graph.node):
            if node.op_type == 'GELU' or (hasattr(node, 'domain') and 'gelu' in node.name.lower()):
                # Decompose GELU
                input_name = node.input[0]
                output_name = node.output[0]
                
                # Create intermediate tensor names
                x_cubed = f"{input_name}_cubed"
                scaled_x_cubed = f"{input_name}_scaled_cubed"
                inner_sum = f"{input_name}_inner_sum"
                tanh_input = f"{input_name}_tanh_input"
                tanh_output = f"{input_name}_tanh_output"
                one_plus_tanh = f"{input_name}_one_plus_tanh"
                half_result = f"{input_name}_half_result"
                
                decomposed_nodes = [
                    # x^3
                    helper.make_node('Mul', [input_name, input_name], [f"{input_name}_squared"]),
                    helper.make_node('Mul', [f"{input_name}_squared", input_name], [x_cubed]),
                    
                    # 0.044715 * x^3
                    helper.make_node('Constant', [], [f"{input_name}_coeff"], 
                                   value=helper.make_tensor('coeff', onnx.TensorProto.FLOAT, [], [0.044715])),
                    helper.make_node('Mul', [f"{input_name}_coeff", x_cubed], [scaled_x_cubed]),
                    
                    # x + 0.044715 * x^3
                    helper.make_node('Add', [input_name, scaled_x_cubed], [inner_sum]),
                    
                    # sqrt(2/π) * (x + 0.044715 * x^3)
                    helper.make_node('Constant', [], [f"{input_name}_sqrt_coeff"],
                                   value=helper.make_tensor('sqrt_coeff', onnx.TensorProto.FLOAT, [], [0.7978845608])),
                    helper.make_node('Mul', [f"{input_name}_sqrt_coeff", inner_sum], [tanh_input]),
                    
                    # tanh(sqrt(2/π) * (x + 0.044715 * x^3))
                    helper.make_node('Tanh', [tanh_input], [tanh_output]),
                    
                    # 1 + tanh(...)
                    helper.make_node('Constant', [], [f"{input_name}_one"],
                                   value=helper.make_tensor('one', onnx.TensorProto.FLOAT, [], [1.0])),
                    helper.make_node('Add', [f"{input_name}_one", tanh_output], [one_plus_tanh]),
                    
                    # 0.5 * (1 + tanh(...))
                    helper.make_node('Constant', [], [f"{input_name}_half"],
                                   value=helper.make_tensor('half', onnx.TensorProto.FLOAT, [], [0.5])),
                    helper.make_node('Mul', [f"{input_name}_half", one_plus_tanh], [half_result]),
                    
                    # x * 0.5 * (1 + tanh(...))
                    helper.make_node('Mul', [input_name, half_result], [output_name])
                ]
                
                nodes_to_replace.append((i, decomposed_nodes))
        
        # Replace nodes
        for original_idx, new_nodes in reversed(nodes_to_replace):
            # Remove original node
            del graph.node[original_idx]
            # Insert new nodes
            for j, new_node in enumerate(new_nodes):
                graph.node.insert(original_idx + j, new_node)
        
        onnx.save(model, output_path)
        print(f"Decomposed model saved: {output_path}")
        return output_path

# Usage example
custom_handler = CustomOperationHandler()
custom_handler.register_custom_operation()
standard_model = custom_handler.replace_custom_ops_with_standard("model_with_custom_ops.onnx")
decomposed_model = custom_handler.decompose_complex_operations(standard_model)
```

### Performance Implications

```python
import time
import numpy as np
import onnxruntime as ort
import psutil
import os

class PerformanceProfiler:
    """Profile performance across different deployment scenarios"""
    
    def __init__(self):
        self.results = {}
    
    def profile_model_variants(self, model_paths, test_input, num_runs=100):
        """Profile different model variants"""
        
        for model_name, model_path in model_paths.items():
            print(f"\nProfiling {model_name}...")
            
            try:
                # Create session
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                
                # Warmup
                for _ in range(10):
                    session.run(None, {input_name: test_input})
                
                # Measure performance
                times = []
                memory_usage = []
                
                for _ in range(num_runs):
                    # Measure memory before
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Run inference
                    start_time = time.time()
                    outputs = session.run(None, {input_name: test_input})
                    end_time = time.time()
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    times.append((end_time - start_time) * 1000)  # ms
                    memory_usage.append(memory_after - memory_before)
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_memory = np.mean(memory_usage)
                
                # Get model size
                model_size = os.path.getsize(model_path) / 1024 / 1024  # MB
                
                self.results[model_name] = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'avg_memory_mb': avg_memory,
                    'model_size_mb': model_size,
                    'throughput_fps': 1000 / avg_time
                }
                
                print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
                print(f"  Memory usage: {avg_memory:.2f} MB")
                print(f"  Model size: {model_size:.2f} MB")
                print(f"  Throughput: {1000/avg_time:.1f} FPS")
                
            except Exception as e:
                print(f"  Error profiling {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
    
    def compare_frameworks(self, pytorch_model, tf_model, onnx_model, test_input):
        """Compare performance across frameworks"""
        import torch
        import tensorflow as tf
        
        print("Framework Performance Comparison")
        print("=" * 50)
        
        # PyTorch
        if pytorch_model is not None:
            pytorch_model.eval()
            torch_input = torch.from_numpy(test_input)
            
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = pytorch_model(torch_input)
                
                # Measure
                times = []
                for _ in range(100):
                    start = time.time()
                    _ = pytorch_model(torch_input)
                    times.append((time.time() - start) * 1000)
                
                pytorch_time = np.mean(times)
                print(f"PyTorch: {pytorch_time:.2f} ms")
        
        # TensorFlow
        if tf_model is not None:
            # Warmup
            for _ in range(10):
                _ = tf_model(test_input)
            
            # Measure
            times = []
            for _ in range(100):
                start = time.time()
                _ = tf_model(test_input)
                times.append((time.time() - start) * 1000)
            
            tf_time = np.mean(times)
            print(f"TensorFlow: {tf_time:.2f} ms")
        
        # ONNX Runtime
        if onnx_model is not None:
            session = ort.InferenceSession(onnx_model)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: test_input})
            
            # Measure
            times = []
            for _ in range(100):
                start = time.time()
                _ = session.run(None, {input_name: test_input})
                times.append((time.time() - start) * 1000)
            
            onnx_time = np.mean(times)
            print(f"ONNX Runtime: {onnx_time:.2f} ms")
    
    def generate_report(self, output_file="performance_report.txt"):
        """Generate detailed performance report"""
        with open(output_file, 'w') as f:
            f.write("ONNX Model Performance Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                if 'error' in metrics:
                    f.write(f"Error: {metrics['error']}\n\n")
                    continue
                
                f.write(f"Average Inference Time: {metrics['avg_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms\n")
                f.write(f"Throughput: {metrics['throughput_fps']:.1f} FPS\n")
                f.write(f"Memory Usage: {metrics['avg_memory_mb']:.2f} MB\n")
                f.write(f"Model Size: {metrics['model_size_mb']:.2f} MB\n")
                f.write(f"Efficiency: {metrics['throughput_fps']/metrics['model_size_mb']:.2f} FPS/MB\n\n")
        
        print(f"Performance report saved: {output_file}")

# Example usage
profiler = PerformanceProfiler()

# Profile different model variants
model_variants = {
    'original': 'model_fp32.onnx',
    'quantized': 'model_int8.onnx',
    'optimized': 'model_optimized.onnx'
}

test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
profiler.profile_model_variants(model_variants, test_input)
profiler.generate_report()
```

## Learning Objectives

By the end of this section, you should be able to:
- **Understand ONNX format structure** and its role in ML model interoperability
- **Convert models between frameworks** (PyTorch ↔ TensorFlow ↔ ONNX) with proper validation
- **Deploy ONNX models efficiently** using ONNX Runtime across different hardware platforms
- **Handle common conversion issues** including data format mismatches and operator compatibility
- **Optimize ONNX models** for production deployment with quantization and graph optimization
- **Implement C++ inference pipelines** using ONNX Runtime for high-performance applications
- **Debug and profile model performance** across different deployment scenarios

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Export a PyTorch model to ONNX format with dynamic shapes  
□ Convert a TensorFlow model to ONNX and validate the conversion  
□ Load and run inference with ONNX Runtime in both Python and C++  
□ Identify and fix common operator compatibility issues  
□ Optimize ONNX models using quantization and graph optimization  
□ Handle data preprocessing differences between frameworks  
□ Profile model performance and identify bottlenecks  
□ Deploy ONNX models with different execution providers (CPU, GPU, etc.)  

### Practical Projects

**Project 1: Cross-Framework Model Pipeline**
```python
# TODO: Create a complete pipeline that:
# 1. Trains a model in PyTorch
# 2. Exports to ONNX
# 3. Validates conversion accuracy
# 4. Optimizes for deployment
# 5. Deploys with ONNX Runtime

class CrossFrameworkPipeline:
    def __init__(self, model_architecture):
        self.model = model_architecture
        
    def train_pytorch_model(self, train_data):
        # Your training code here
        pass
        
    def export_to_onnx(self, model_path):
        # Your ONNX export code here
        pass
        
    def validate_conversion(self, test_data):
        # Your validation code here
        pass
        
    def optimize_for_deployment(self):
        # Your optimization code here
        pass
```

**Project 2: Production Deployment System**
```cpp
// TODO: Implement a C++ inference service that:
// 1. Loads ONNX models efficiently
// 2. Handles batch processing
// 3. Provides REST API endpoints
// 4. Monitors performance metrics

class ONNXInferenceService {
public:
    ONNXInferenceService(const std::string& model_path);
    
    std::vector<float> predict(const std::vector<float>& input);
    void loadModel(const std::string& model_path);
    void optimizeForHardware();
    
private:
    // Your implementation here
};
```

**Project 3: Model Conversion Toolkit**
```python
# TODO: Create a comprehensive toolkit for:
# 1. Automated model conversion
# 2. Compatibility checking
# 3. Performance benchmarking
# 4. Deployment optimization

class ModelConversionToolkit:
    def __init__(self):
        self.supported_formats = ['pytorch', 'tensorflow', 'onnx']
        
    def convert_model(self, input_path, output_format):
        # Your conversion logic here
        pass
        
    def check_compatibility(self, model_path):
        # Your compatibility checking here
        pass
        
    def benchmark_performance(self, model_variants):
        # Your benchmarking code here
        pass
```

## Study Materials

### Essential Reading
- **Primary:** [ONNX Official Documentation](https://onnx.ai/onnx/)
- **Deep Dive:** [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- **Reference:** "Hands-On Machine Learning" - Chapter on Model Deployment
- **Advanced:** [ONNX Operator Schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

### Video Resources
- "ONNX: Open Neural Network Exchange" - Microsoft Build Conference
- "Cross-Framework Model Deployment" - PyTorch Developer Conference
- "ONNX Runtime Performance Optimization" - NVIDIA GTC Sessions
- "Production ML with ONNX" - MLOps World Conference

### Hands-on Labs
- **Lab 1:** Convert computer vision models between PyTorch and TensorFlow
- **Lab 2:** Deploy ONNX models on edge devices (Raspberry Pi, mobile)
- **Lab 3:** Build a multi-model inference service with ONNX Runtime
- **Lab 4:** Optimize models for specific hardware (CPU, GPU, NPU)

### Practice Scenarios

**Beginner Level:**
1. Convert a simple linear regression model from PyTorch to ONNX
2. Run basic inference with ONNX Runtime Python API
3. Compare inference times between frameworks
4. Export a pretrained ResNet model to ONNX format

**Intermediate Level:**
5. Handle dynamic input shapes in model conversion
6. Implement custom preprocessing in ONNX graphs
7. Quantize models for mobile deployment
8. Create C++ inference applications

**Advanced Level:**
9. Build custom ONNX operators for domain-specific operations
10. Implement distributed inference with ONNX Runtime
11. Optimize memory usage for large language models
12. Create automated CI/CD pipelines for model deployment

### Development Environment Setup

**Required Tools and Libraries:**
```bash
# Core ONNX tools
pip install onnx onnxruntime onnxruntime-gpu

# Framework conversion tools
pip install tf2onnx                    # TensorFlow to ONNX
pip install torch torchvision          # PyTorch (built-in ONNX export)
pip install onnx-tensorflow            # ONNX to TensorFlow

# Optimization tools
pip install onnxoptimizer             # Graph optimization
pip install onnxsim                   # Model simplification

# Quantization and compression
pip install onnxruntime-tools         # Quantization tools

# Visualization and debugging
pip install netron                    # Model visualization
pip install onnx-utils                # ONNX utilities

# Performance profiling
pip install psutil                    # System monitoring
pip install py-spy                    # Python profiler
```

**C++ Development Setup:**
```bash
# Download ONNX Runtime C++ libraries
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz

# CMake configuration
cmake -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime \
      -DCMAKE_BUILD_TYPE=Release \
      .

# Compilation flags
g++ -std=c++17 -O3 \
    -I/path/to/onnxruntime/include \
    -L/path/to/onnxruntime/lib \
    -lonnxruntime \
    inference.cpp -o inference
```

**Docker Environment:**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install onnx onnxruntime-gpu torch tensorflow

# Copy your code
COPY . /workspace
WORKDIR /workspace

# Run inference service
CMD ["python3", "inference_service.py"]
```

### Debugging and Troubleshooting

**Common Issues and Solutions:**

1. **Operator Not Supported**
   ```python
   # Check supported operators
   import onnxruntime as ort
   print(ort.get_available_providers())
   
   # Use model optimizer
   from onnxoptimizer import optimize
   optimized_model = optimize(model)
   ```

2. **Shape Inference Errors**
   ```python
   # Use shape inference
   import onnx.shape_inference as shape_inference
   inferred_model = shape_inference.infer_shapes(model)
   ```

3. **Performance Issues**
   ```python
   # Enable optimizations
   session_options = ort.SessionOptions()
   session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   session = ort.InferenceSession(model_path, session_options)
   ```

**Validation Commands:**
```bash
# Validate ONNX model
python -m onnx.checker model.onnx

# Visualize model structure
netron model.onnx

# Check model information
python -c "import onnx; model = onnx.load('model.onnx'); print(onnx.helper.printable_graph(model.graph))"

# Benchmark performance
python -m onnxruntime.tools.benchmark -m model.onnx -r 100
```

### Assessment Questions

**Conceptual Understanding:**
1. What are the main advantages of using ONNX for model deployment?
2. How does ONNX handle operator versioning and compatibility?
3. What are the trade-offs between different execution providers?
4. When would you choose ONNX Runtime over framework-native inference?

**Technical Implementation:**
5. How do you handle dynamic shapes during PyTorch to ONNX conversion?
6. What steps are needed to optimize an ONNX model for mobile deployment?
7. How can you implement custom preprocessing within an ONNX graph?
8. What are the best practices for C++ ONNX Runtime integration?

**Problem Solving:**
9. A PyTorch model with custom operations fails to export to ONNX. How do you fix this?
10. Your ONNX model runs slower than the original framework. How do you debug this?
11. Model accuracy drops after conversion to ONNX. What validation steps do you take?
12. You need to deploy the same model on CPU, GPU, and mobile devices. How do you approach this?
