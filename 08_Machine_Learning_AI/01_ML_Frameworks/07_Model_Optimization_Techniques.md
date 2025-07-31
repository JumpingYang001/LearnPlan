# Model Optimization Techniques

*Duration: 2 weeks*

## Overview

Model optimization is crucial for deploying machine learning models in production environments where computational resources, memory, and inference speed are constrained. This comprehensive guide covers the essential techniques for making models faster, smaller, and more efficient without significantly sacrificing accuracy.

## Quantization

Quantization reduces the precision of model weights and activations from 32-bit floating-point (FP32) to lower precision formats like 8-bit integers (INT8) or 16-bit floating-point (FP16).

### Understanding Quantization Fundamentals

**Why Quantization Works:**
- Neural networks are often over-parameterized
- Models can tolerate some precision loss
- Lower precision operations are faster and use less memory
- Hardware acceleration for low-precision arithmetic

**Quantization Formula:**
```
quantized_value = round((float_value - zero_point) / scale)
dequantized_value = scale * (quantized_value + zero_point)
```

### Post-Training Quantization (PTQ)

Post-training quantization converts a trained FP32 model to a quantized version without retraining.

**PyTorch Implementation:**
```python
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.quantization import get_default_qconfig
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

class ModelBenchmark:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def measure_inference_time(self, num_batches=100):
        """Measure average inference time"""
        self.model.eval()
        times = []
        
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                if i >= num_batches:
                    break
                
                start_time = time.time()
                _ = self.model(data)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return sum(times) / len(times)
    
    def measure_model_size(self):
        """Measure model size in MB"""
        model_size = 0
        for param in self.model.parameters():
            model_size += param.nelement() * param.element_size()
        return model_size / (1024 * 1024)  # Convert to MB

# Example: ResNet18 Quantization
def post_training_quantization_example():
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Prepare model for quantization
    model.qconfig = get_default_qconfig('fbgemm')  # x86 CPU backend
    model_prepared = tq.prepare(model, inplace=False)
    
    # Calibration data (representative dataset)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create calibration dataset (using dummy data for example)
    calibration_data = torch.randn(100, 3, 224, 224)
    
    # Calibration step
    print("Calibrating model...")
    with torch.no_grad():
        for i in range(10):  # Use subset for calibration
            batch = calibration_data[i*10:(i+1)*10]
            model_prepared(batch)
    
    # Convert to quantized model
    model_quantized = tq.convert(model_prepared, inplace=False)
    
    # Compare models
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Original model
    original_size = ModelBenchmark(model, None).measure_model_size()
    start_time = time.time()
    with torch.no_grad():
        output_fp32 = model(dummy_input)
    fp32_time = time.time() - start_time
    
    # Quantized model
    quantized_size = ModelBenchmark(model_quantized, None).measure_model_size()
    start_time = time.time()
    with torch.no_grad():
        output_int8 = model_quantized(dummy_input)
    int8_time = time.time() - start_time
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"FP32 inference time: {fp32_time*1000:.2f} ms")
    print(f"INT8 inference time: {int8_time*1000:.2f} ms")
    print(f"Speed improvement: {fp32_time/int8_time:.1f}x")
    
    return model, model_quantized

# Dynamic Quantization (easier approach)
def dynamic_quantization_example():
    """Dynamic quantization - weights quantized, activations computed in FP32"""
    model = models.resnet18(pretrained=True)
    
    # Apply dynamic quantization
    model_dynamic_quantized = tq.quantize_dynamic(
        model, 
        {nn.Conv2d, nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )
    
    return model_dynamic_quantized

if __name__ == "__main__":
    # Run examples
    original_model, quantized_model = post_training_quantization_example()
    dynamic_model = dynamic_quantization_example()
```

### Quantization-Aware Training (QAT)

QAT simulates quantization during training, allowing the model to adapt to quantization errors.

**Implementation Example:**
```python
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.quantization import QuantStub, DeQuantStub

class QuantizedResNetBlock(nn.Module):
    """Custom ResNet block with quantization support"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip_add = nn.quantized.FloatFunctional()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Use quantized addition
        out = self.skip_add.add(out, self.shortcut(x))
        out = self.relu(out)
        
        return out

class QuantizedModel(nn.Module):
    """Model designed for quantization-aware training"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Model layers
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(QuantizedResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(QuantizedResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Quantize input
        x = self.quant(x)
        
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # Dequantize output
        x = self.dequant(x)
        return x

def quantization_aware_training():
    """Complete QAT pipeline"""
    # Create model
    model = QuantizedModel(num_classes=10)
    
    # Set quantization config
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    
    # Prepare model for QAT
    model_prepared = tq.prepare_qat(model, inplace=False)
    
    # Training loop (simplified)
    optimizer = torch.optim.SGD(model_prepared.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Quantization-Aware Training...")
    model_prepared.train()
    
    # Dummy training data
    for epoch in range(5):
        for batch_idx in range(100):  # Simplified training
            data = torch.randn(32, 3, 224, 224)
            target = torch.randint(0, 10, (32,))
            
            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Convert to quantized model for inference
    model_prepared.eval()
    model_quantized = tq.convert(model_prepared, inplace=False)
    
    return model_quantized

# Advanced: Custom Quantization
class CustomQuantization:
    """Custom quantization implementation"""
    
    @staticmethod
    def linear_quantize(tensor, scale, zero_point, dtype=torch.qint8):
        """Manual linear quantization"""
        qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        
        # Scale and shift
        quantized = torch.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        quantized = torch.clamp(quantized, qmin, qmax)
        
        return quantized.to(dtype)
    
    @staticmethod
    def calculate_scale_zero_point(tensor, dtype=torch.qint8):
        """Calculate optimal scale and zero_point for tensor"""
        qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        
        min_val, max_val = tensor.min(), tensor.max()
        
        # Calculate scale
        scale = (max_val - min_val) / (qmax - qmin)
        
        # Calculate zero point
        zero_point = qmin - min_val / scale
        zero_point = torch.clamp(torch.round(zero_point), qmin, qmax)
        
        return scale, zero_point

# Example usage
if __name__ == "__main__":
    # Run QAT example
    qat_model = quantization_aware_training()
    
    # Custom quantization example
    custom_quant = CustomQuantization()
    sample_tensor = torch.randn(100, 100)
    scale, zero_point = custom_quant.calculate_scale_zero_point(sample_tensor)
    quantized_tensor = custom_quant.linear_quantize(sample_tensor, scale, zero_point)
    
    print(f"Original tensor range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]")
    print(f"Quantization scale: {scale:.6f}, zero_point: {zero_point}")
    print(f"Quantized tensor range: [{quantized_tensor.min()}, {quantized_tensor.max()}]")
```

### INT8/FP16 Computation

**FP16 (Half Precision) Example:**
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class FP16Training:
    """Mixed precision training with FP16"""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()  # For gradient scaling
    
    def train_step(self, data, target, criterion):
        """Single training step with mixed precision"""
        with autocast():  # Automatic mixed precision
            output = self.model(data)
            loss = criterion(output, target)
        
        # Scale loss to prevent gradient underflow
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()

# Example: Converting model to FP16
def fp16_inference_example():
    model = models.resnet18(pretrained=True)
    
    # Convert to FP16
    model = model.half()  # Convert weights to FP16
    
    # FP16 input
    input_fp16 = torch.randn(1, 3, 224, 224).half()
    
    # Inference
    with torch.no_grad():
        output = model(input_fp16)
    
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Input dtype: {input_fp16.dtype}")
    print(f"Output dtype: {output.dtype}")
    
    return model

# TensorRT optimization (NVIDIA GPUs)
def tensorrt_optimization():
    """Example of TensorRT optimization for INT8"""
    try:
        import tensorrt as trt
        
        # This is a conceptual example
        # Actual implementation requires more setup
        
        def build_engine_int8(onnx_file_path, engine_file_path, calibration_dataset):
            """Build TensorRT engine with INT8 precision"""
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_file_path, 'rb') as model:
                parser.parse(model.read())
            
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            config.set_flag(trt.BuilderFlag.INT8)  # Enable INT8
            
            # Set calibration data
            config.int8_calibrator = calibration_dataset
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            # Save engine
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
            
            return engine
        
        print("TensorRT INT8 optimization setup complete")
        
    except ImportError:
        print("TensorRT not available. Install with: pip install nvidia-tensorrt")

if __name__ == "__main__":
    fp16_model = fp16_inference_example()
    tensorrt_optimization()
```

### Symmetric vs. Asymmetric Quantization

**Symmetric Quantization:**
- Zero point is always 0
- Simpler computation: `quantized = round(float_value / scale)`
- May waste representation space if data is not centered around 0

**Asymmetric Quantization:**
- Zero point can be any value within quantization range
- Better utilizes available quantization levels
- More complex computation but better accuracy

```python
class QuantizationComparison:
    """Compare symmetric vs asymmetric quantization"""
    
    @staticmethod
    def symmetric_quantize(tensor, num_bits=8):
        """Symmetric quantization implementation"""
        n_levels = 2 ** num_bits
        max_val = tensor.abs().max()
        scale = max_val / (n_levels // 2 - 1)
        
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -(n_levels//2), n_levels//2-1)
        
        return quantized, scale, 0  # zero_point = 0
    
    @staticmethod
    def asymmetric_quantize(tensor, num_bits=8):
        """Asymmetric quantization implementation"""
        n_levels = 2 ** num_bits
        min_val, max_val = tensor.min(), tensor.max()
        
        scale = (max_val - min_val) / (n_levels - 1)
        zero_point = -min_val / scale
        zero_point = torch.clamp(torch.round(zero_point), 0, n_levels-1)
        
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, n_levels-1)
        
        return quantized, scale, zero_point
    
    @staticmethod
    def compare_quantization_methods(tensor):
        """Compare both methods on the same tensor"""
        print(f"Original tensor stats:")
        print(f"  Min: {tensor.min():.4f}, Max: {tensor.max():.4f}")
        print(f"  Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        
        # Symmetric quantization
        sym_q, sym_scale, sym_zp = QuantizationComparison.symmetric_quantize(tensor)
        sym_dequant = sym_scale * sym_q
        sym_error = torch.mean((tensor - sym_dequant) ** 2).item()
        
        # Asymmetric quantization
        asym_q, asym_scale, asym_zp = QuantizationComparison.asymmetric_quantize(tensor)
        asym_dequant = asym_scale * (asym_q - asym_zp)
        asym_error = torch.mean((tensor - asym_dequant) ** 2).item()
        
        print(f"\nSymmetric quantization:")
        print(f"  Scale: {sym_scale:.6f}, Zero point: {sym_zp}")
        print(f"  MSE error: {sym_error:.6f}")
        
        print(f"\nAsymmetric quantization:")
        print(f"  Scale: {asym_scale:.6f}, Zero point: {asym_zp:.1f}")
        print(f"  MSE error: {asym_error:.6f}")
        
        print(f"\nAsymmetric is {sym_error/asym_error:.2f}x better" if asym_error < sym_error 
              else f"Symmetric is {asym_error/sym_error:.2f}x better")

# Example usage
if __name__ == "__main__":
    # Test with different tensor distributions
    print("=== Testing with normal distribution (centered) ===")
    normal_tensor = torch.randn(1000)
    QuantizationComparison.compare_quantization_methods(normal_tensor)
    
    print("\n=== Testing with positive-skewed distribution ===")
    skewed_tensor = torch.abs(torch.randn(1000)) + 5  # Positive values only
    QuantizationComparison.compare_quantization_methods(skewed_tensor)
```

## Pruning

Pruning removes unnecessary weights and connections from neural networks to reduce model size and computational requirements while maintaining performance.

### Understanding Pruning Fundamentals

**Why Pruning Works:**
- Neural networks are typically over-parameterized
- Many weights contribute minimally to the final output
- Removing redundant parameters can improve generalization
- Significant reduction in memory usage and computation

**Types of Pruning:**
1. **Magnitude-based**: Remove weights with smallest absolute values
2. **Gradient-based**: Remove weights with smallest gradients
3. **Random**: Remove weights randomly (baseline)
4. **Structured**: Remove entire neurons, channels, or layers
5. **Unstructured**: Remove individual weights

### Weight Pruning Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class PruningToolkit:
    """Comprehensive pruning toolkit"""
    
    @staticmethod
    def magnitude_pruning(model, pruning_ratio=0.2):
        """Remove weights with smallest magnitude"""
        # Collect all weights
        weights = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights.append(module.weight.data.view(-1))
        
        # Concatenate all weights
        all_weights = torch.cat(weights)
        
        # Calculate threshold
        threshold = torch.quantile(torch.abs(all_weights), pruning_ratio)
        
        # Apply pruning
        pruned_params = 0
        total_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                mask = torch.abs(module.weight.data) > threshold
                module.weight.data *= mask.float()
                
                pruned_params += torch.sum(~mask).item()
                total_params += mask.numel()
        
        actual_pruning_ratio = pruned_params / total_params
        print(f"Pruned {pruned_params}/{total_params} parameters ({actual_pruning_ratio:.2%})")
        
        return model
    
    @staticmethod
    def gradual_magnitude_pruning(model, initial_sparsity=0.0, final_sparsity=0.9, 
                                num_iterations=10):
        """Gradually increase pruning over training iterations"""
        sparsity_schedule = np.linspace(initial_sparsity, final_sparsity, num_iterations)
        
        pruned_models = []
        for i, target_sparsity in enumerate(sparsity_schedule):
            print(f"Iteration {i+1}: Target sparsity = {target_sparsity:.2%}")
            
            # Clone model for this iteration
            model_copy = PruningToolkit._deep_copy_model(model)
            
            # Apply pruning
            pruned_model = PruningToolkit.magnitude_pruning(model_copy, target_sparsity)
            pruned_models.append(pruned_model)
            
            # Fine-tune model here (simplified for example)
            # fine_tune(pruned_model, train_loader, epochs=5)
        
        return pruned_models
    
    @staticmethod
    def _deep_copy_model(model):
        """Create a deep copy of the model"""
        model_copy = type(model)()
        model_copy.load_state_dict(model.state_dict())
        return model_copy

# Advanced Magnitude-based Pruning with Masks
class MagnitudePruner:
    """Advanced magnitude-based pruning with persistent masks"""
    
    def __init__(self, model):
        self.model = model
        self.masks = {}
        self._create_masks()
    
    def _create_masks(self):
        """Create initial masks (all ones - no pruning)"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.masks[name] = torch.ones_like(module.weight.data)
    
    def prune_by_percentage(self, pruning_percentage=20):
        """Prune by percentage globally"""
        # Collect all weights
        all_weights = []
        weight_names = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Only consider non-pruned weights
                weights = module.weight.data[self.masks[name] == 1]
                all_weights.append(weights.view(-1))
                weight_names.append(name)
        
        # Find global threshold
        all_weights_tensor = torch.cat(all_weights)
        threshold = torch.quantile(torch.abs(all_weights_tensor), 
                                 pruning_percentage / 100.0)
        
        # Update masks
        total_pruned = 0
        total_weights = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Update mask: keep weights above threshold
                new_mask = (torch.abs(module.weight.data) >= threshold).float()
                # Ensure we don't un-prune previously pruned weights
                self.masks[name] = self.masks[name] * new_mask
                
                # Apply mask
                module.weight.data *= self.masks[name]
                
                # Statistics
                pruned = torch.sum(self.masks[name] == 0).item()
                total = self.masks[name].numel()
                total_pruned += pruned
                total_weights += total
                
                print(f"Layer {name}: {pruned}/{total} pruned ({pruned/total:.2%})")
        
        overall_sparsity = total_pruned / total_weights
        print(f"Overall sparsity: {overall_sparsity:.2%}")
        
        return overall_sparsity
    
    def apply_masks(self):
        """Apply current masks to model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in self.masks:
                module.weight.data *= self.masks[name]
    
    def get_sparsity_stats(self):
        """Get detailed sparsity statistics"""
        stats = {}
        total_params = 0
        total_pruned = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in self.masks:
                mask = self.masks[name]
                pruned = torch.sum(mask == 0).item()
                total = mask.numel()
                
                stats[name] = {
                    'total_params': total,
                    'pruned_params': pruned,
                    'sparsity': pruned / total,
                    'remaining_params': total - pruned
                }
                
                total_params += total
                total_pruned += pruned
        
        stats['overall'] = {
            'total_params': total_params,
            'pruned_params': total_pruned,
            'sparsity': total_pruned / total_params,
            'remaining_params': total_params - total_pruned
        }
        
        return stats

# Example model for demonstration
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def pruning_comparison_example():
    """Compare different pruning strategies"""
    # Create model
    model = SimpleNet()
    
    # Initialize pruner
    pruner = MagnitudePruner(model)
    
    print("=== Initial Model Stats ===")
    initial_stats = pruner.get_sparsity_stats()
    print(f"Total parameters: {initial_stats['overall']['total_params']:,}")
    
    # Gradual pruning simulation
    pruning_steps = [10, 20, 30, 40, 50]  # Percentage to prune at each step
    
    for step in pruning_steps:
        print(f"\n=== Pruning Step: {step}% ===")
        sparsity = pruner.prune_by_percentage(step)
        
        # Simulate fine-tuning (in practice, you'd retrain the model)
        print("Fine-tuning model...")
        
        # Get current stats
        stats = pruner.get_sparsity_stats()
        print(f"Current overall sparsity: {stats['overall']['sparsity']:.2%}")
        print(f"Remaining parameters: {stats['overall']['remaining_params']:,}")
    
    return model, pruner

if __name__ == "__main__":
    model, pruner = pruning_comparison_example()
```

### Structured vs. Unstructured Pruning

**Unstructured Pruning**: Removes individual weights
- Higher compression ratios possible
- Requires sparse matrix operations for speed benefits
- More complex hardware implementation

**Structured Pruning**: Removes entire structures (channels, filters, neurons)
- Lower compression ratios but guaranteed speedup
- Compatible with standard dense operations
- Easier hardware implementation

```python
class StructuredPruning:
    """Structured pruning implementations"""
    
    @staticmethod
    def channel_pruning_conv2d(conv_layer, bn_layer, pruning_ratio=0.3):
        """Remove entire channels from Conv2d layer"""
        num_channels = conv_layer.out_channels
        num_to_prune = int(num_channels * pruning_ratio)
        
        # Calculate channel importance (L1 norm of filters)
        channel_importance = torch.sum(torch.abs(conv_layer.weight.data), dim=(1, 2, 3))
        
        # Get indices of least important channels
        _, pruned_indices = torch.topk(channel_importance, num_to_prune, largest=False)
        
        # Get indices of channels to keep
        all_indices = set(range(num_channels))
        keep_indices = list(all_indices - set(pruned_indices.tolist()))
        keep_indices = sorted(keep_indices)
        
        # Prune conv layer
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            len(keep_indices),
            conv_layer.kernel_size,
            conv_layer.stride,
            conv_layer.padding,
            conv_layer.dilation,
            conv_layer.groups,
            conv_layer.bias is not None
        )
        
        new_conv.weight.data = conv_layer.weight.data[keep_indices]
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[keep_indices]
        
        # Prune batch norm layer
        new_bn = nn.BatchNorm2d(len(keep_indices))
        new_bn.weight.data = bn_layer.weight.data[keep_indices]
        new_bn.bias.data = bn_layer.bias.data[keep_indices]
        new_bn.running_mean = bn_layer.running_mean[keep_indices]
        new_bn.running_var = bn_layer.running_var[keep_indices]
        
        print(f"Pruned {num_to_prune} channels from {num_channels} "
              f"({pruning_ratio:.1%} reduction)")
        
        return new_conv, new_bn, keep_indices
    
    @staticmethod
    def neuron_pruning_linear(linear_layer, pruning_ratio=0.3):
        """Remove entire neurons from Linear layer"""
        num_neurons = linear_layer.out_features
        num_to_prune = int(num_neurons * pruning_ratio)
        
        # Calculate neuron importance (L2 norm of weights)
        neuron_importance = torch.norm(linear_layer.weight.data, dim=1)
        
        # Get indices of least important neurons
        _, pruned_indices = torch.topk(neuron_importance, num_to_prune, largest=False)
        
        # Get indices of neurons to keep
        all_indices = set(range(num_neurons))
        keep_indices = list(all_indices - set(pruned_indices.tolist()))
        keep_indices = sorted(keep_indices)
        
        # Create new layer
        new_linear = nn.Linear(
            linear_layer.in_features,
            len(keep_indices),
            linear_layer.bias is not None
        )
        
        new_linear.weight.data = linear_layer.weight.data[keep_indices]
        if linear_layer.bias is not None:
            new_linear.bias.data = linear_layer.bias.data[keep_indices]
        
        print(f"Pruned {num_to_prune} neurons from {num_neurons} "
              f"({pruning_ratio:.1%} reduction)")
        
        return new_linear, keep_indices

class AutoStructuredPruner:
    """Automatically apply structured pruning to models"""
    
    def __init__(self, model):
        self.model = model
        self.original_structure = self._analyze_model_structure()
    
    def _analyze_model_structure(self):
        """Analyze model structure for pruning"""
        structure = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                structure[name] = {
                    'type': type(module).__name__,
                    'module': module
                }
        return structure
    
    def prune_model(self, target_reduction=0.5):
        """Apply structured pruning to entire model"""
        print(f"Applying structured pruning with {target_reduction:.1%} target reduction...")
        
        # Track changes for layer dependency handling
        layer_changes = {}
        
        # Process Conv2d + BatchNorm2d pairs
        conv_layers = [name for name, info in self.original_structure.items() 
                      if info['type'] == 'Conv2d']
        
        for conv_name in conv_layers:
            # Find corresponding BatchNorm
            bn_name = conv_name.replace('conv', 'bn')  # Simple naming convention
            
            if bn_name in self.original_structure:
                conv_layer = self.original_structure[conv_name]['module']
                bn_layer = self.original_structure[bn_name]['module']
                
                # Apply channel pruning
                new_conv, new_bn, keep_indices = StructuredPruning.channel_pruning_conv2d(
                    conv_layer, bn_layer, target_reduction
                )
                
                # Update model (this would require more sophisticated model modification)
                # For demonstration, we store the changes
                layer_changes[conv_name] = {
                    'new_layer': new_conv,
                    'keep_indices': keep_indices
                }
                layer_changes[bn_name] = {
                    'new_layer': new_bn,
                    'keep_indices': keep_indices
                }
        
        # Process Linear layers
        linear_layers = [name for name, info in self.original_structure.items() 
                        if info['type'] == 'Linear']
        
        for linear_name in linear_layers[:-1]:  # Don't prune final classification layer
            linear_layer = self.original_structure[linear_name]['module']
            
            new_linear, keep_indices = StructuredPruning.neuron_pruning_linear(
                linear_layer, target_reduction
            )
            
            layer_changes[linear_name] = {
                'new_layer': new_linear,
                'keep_indices': keep_indices
            }
        
        return layer_changes
    
    def calculate_model_size_reduction(self, layer_changes):
        """Calculate actual model size reduction"""
        original_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate new parameter count
        new_params = original_params
        for layer_name, change_info in layer_changes.items():
            if 'new_layer' in change_info:
                original_layer = self.original_structure[layer_name]['module']
                new_layer = change_info['new_layer']
                
                original_layer_params = sum(p.numel() for p in original_layer.parameters())
                new_layer_params = sum(p.numel() for p in new_layer.parameters())
                
                new_params = new_params - original_layer_params + new_layer_params
        
        reduction_ratio = (original_params - new_params) / original_params
        
        print(f"Original parameters: {original_params:,}")
        print(f"New parameters: {new_params:,}")
        print(f"Reduction: {reduction_ratio:.2%}")
        
        return reduction_ratio

# Example usage
def structured_pruning_example():
    """Demonstrate structured pruning"""
    model = SimpleNet()
    
    print("=== Original Model ===")
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {original_params:,}")
    
    # Apply structured pruning
    pruner = AutoStructuredPruner(model)
    layer_changes = pruner.prune_model(target_reduction=0.3)
    
    # Calculate size reduction
    reduction = pruner.calculate_model_size_reduction(layer_changes)
    
    return model, layer_changes

if __name__ == "__main__":
    structured_pruning_example()
```

### Iterative Pruning

Iterative pruning gradually removes weights over multiple training cycles, allowing the model to adapt to the reduced capacity.

```python
class IterativePruner:
    """Implement iterative magnitude pruning"""
    
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.9, 
                 pruning_frequency=100, recovery_epochs=10):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.recovery_epochs = recovery_epochs
        
        self.current_sparsity = initial_sparsity
        self.iteration = 0
        self.masks = self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize pruning masks"""
        masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                masks[name] = torch.ones_like(module.weight.data)
        return masks
    
    def _cubic_sparsity_schedule(self, iteration, total_iterations):
        """Cubic sparsity schedule as used in Lottery Ticket Hypothesis"""
        if iteration >= total_iterations:
            return self.final_sparsity
        
        progress = iteration / total_iterations
        sparsity = self.final_sparsity * (1 - (1 - progress) ** 3)
        return max(self.initial_sparsity, sparsity)
    
    def should_prune(self, iteration):
        """Check if pruning should be applied at this iteration"""
        return iteration % self.pruning_frequency == 0 and iteration > 0
    
    def prune_step(self, iteration, total_iterations):
        """Perform one pruning step"""
        if not self.should_prune(iteration):
            return False
        
        # Calculate target sparsity
        target_sparsity = self._cubic_sparsity_schedule(iteration, total_iterations)
        
        if target_sparsity <= self.current_sparsity:
            return False
        
        print(f"Iteration {iteration}: Pruning to {target_sparsity:.2%} sparsity")
        
        # Collect all weights that haven't been pruned
        all_weights = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data[self.masks[name] == 1]
                all_weights.append(weights.view(-1))
        
        # Calculate new threshold
        if len(all_weights) > 0:
            all_weights_tensor = torch.cat(all_weights)
            threshold = torch.quantile(torch.abs(all_weights_tensor), target_sparsity)
            
            # Update masks
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Only prune weights that haven't been pruned yet
                    new_mask = (torch.abs(module.weight.data) >= threshold).float()
                    self.masks[name] = self.masks[name] * new_mask
                    
                    # Apply mask
                    module.weight.data *= self.masks[name]
        
        self.current_sparsity = target_sparsity
        return True
    
    def apply_masks(self):
        """Apply current masks to model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in self.masks:
                module.weight.data *= self.masks[name]
    
    def get_sparsity_info(self):
        """Get current sparsity information"""
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in self.masks:
                mask = self.masks[name]
                total_params += mask.numel()
                pruned_params += torch.sum(mask == 0).item()
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        
        return {
            'target_sparsity': self.current_sparsity,
            'actual_sparsity': actual_sparsity,
            'total_params': total_params,
            'pruned_params': pruned_params,
            'remaining_params': total_params - pruned_params
        }

# Example training loop with iterative pruning
def train_with_iterative_pruning():
    """Example training loop with iterative pruning"""
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize pruner
    pruner = IterativePruner(
        model, 
        initial_sparsity=0.0,
        final_sparsity=0.9,
        pruning_frequency=100,
        recovery_epochs=10
    )
    
    total_iterations = 1000
    
    # Simulate training loop
    for iteration in range(total_iterations):
        # Check if pruning should be applied
        if pruner.prune_step(iteration, total_iterations):
            sparsity_info = pruner.get_sparsity_info()
            print(f"  Actual sparsity: {sparsity_info['actual_sparsity']:.2%}")
            print(f"  Remaining params: {sparsity_info['remaining_params']:,}")
        
        # Apply masks before forward pass
        pruner.apply_masks()
        
        # Simulate training step
        # In real training, you'd have actual data here
        dummy_input = torch.randn(32, 3, 32, 32)
        dummy_target = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        
        # Apply masks to gradients (important!)
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in pruner.masks:
                if module.weight.grad is not None:
                    module.weight.grad *= pruner.masks[name]
        
        optimizer.step()
        
        # Log progress
        if iteration % 100 == 0:
            sparsity_info = pruner.get_sparsity_info()
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}, "
                  f"Sparsity = {sparsity_info['actual_sparsity']:.2%}")
    
    # Final statistics
    final_sparsity_info = pruner.get_sparsity_info()
    print(f"\nFinal Results:")
    print(f"Target sparsity: {final_sparsity_info['target_sparsity']:.2%}")
    print(f"Actual sparsity: {final_sparsity_info['actual_sparsity']:.2%}")
    print(f"Parameters removed: {final_sparsity_info['pruned_params']:,}")
    print(f"Parameters remaining: {final_sparsity_info['remaining_params']:,}")
    
    return model, pruner

if __name__ == "__main__":
    model, pruner = train_with_iterative_pruning()
```

## Knowledge Distillation

Knowledge distillation transfers knowledge from a large, complex model (teacher) to a smaller, simpler model (student), enabling the student to achieve similar performance with fewer parameters.

### Understanding Knowledge Distillation

**Core Concept:**
- Teacher model: Large, pre-trained, high-accuracy model
- Student model: Smaller, faster model to be trained
- Knowledge transfer: Through soft targets (probability distributions) rather than hard labels
- Temperature scaling: Softens probability distributions to reveal more information

**Benefits:**
- Model compression without architectural constraints
- Better performance than training student from scratch
- Transfers learned representations and decision boundaries
- Can combine multiple teachers

### Teacher-Student Models Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TeacherModel(nn.Module):
    """Large teacher model (e.g., ResNet-50 style)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class StudentModel(nn.Module):
    """Smaller student model (e.g., MobileNet style)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Lightweight feature extraction
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Compare model sizes
teacher = TeacherModel()
student = StudentModel()

print(f"Teacher model parameters: {count_parameters(teacher):,}")
print(f"Student model parameters: {count_parameters(student):,}")
print(f"Compression ratio: {count_parameters(teacher)/count_parameters(student):.1f}x")
```

### Distillation Loss Implementation

```python
class DistillationLoss(nn.Module):
    """Knowledge distillation loss function"""
    
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss
        
        Args:
            student_logits: Raw outputs from student model
            teacher_logits: Raw outputs from teacher model  
            labels: Ground truth labels
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = self.kl_loss(student_log_probs, teacher_probs)
        
        # Standard classification loss
        classification_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = (
            self.alpha * (self.temperature ** 2) * distillation_loss +
            (1 - self.alpha) * classification_loss
        )
        
        return total_loss, distillation_loss, classification_loss

class KnowledgeDistiller:
    """Complete knowledge distillation training framework"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.distill_loss = DistillationLoss(temperature, alpha)
        
    def train_student(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train student model using knowledge distillation"""
        optimizer = optim.Adam(self.student.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.student.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                
                student_logits = self.student(data)
                
                # Compute loss
                total_loss, distill_loss, class_loss = self.distill_loss(
                    student_logits, teacher_logits, target
                )
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}: '
                          f'Total Loss: {total_loss.item():.4f}, '
                          f'Distill Loss: {distill_loss.item():.4f}, '
                          f'Class Loss: {class_loss.item():.4f}')
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_acc = self.evaluate_student(val_loader)
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Accuracy: {val_acc:.2%}')
        
        return train_losses, val_accuracies
    
    def evaluate_student(self, test_loader):
        """Evaluate student model accuracy"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def compare_models(self, test_loader):
        """Compare teacher and student performance"""
        # Evaluate teacher
        teacher_acc = self._evaluate_model(self.teacher, test_loader)
        
        # Evaluate student
        student_acc = self.evaluate_student(test_loader)
        
        # Model sizes
        teacher_params = count_parameters(self.teacher)
        student_params = count_parameters(self.student)
        
        print(f"\n=== Model Comparison ===")
        print(f"Teacher - Params: {teacher_params:,}, Accuracy: {teacher_acc:.2%}")
        print(f"Student - Params: {student_params:,}, Accuracy: {student_acc:.2%}")
        print(f"Compression ratio: {teacher_params/student_params:.1f}x")
        print(f"Accuracy retention: {student_acc/teacher_acc:.2%}")
        
        return teacher_acc, student_acc
    
    def _evaluate_model(self, model, test_loader):
        """Generic model evaluation"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total

# Example usage with dummy data
def distillation_example():
    """Complete knowledge distillation example"""
    # Create models
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)
    
    # Create dummy dataset
    train_data = torch.randn(1000, 3, 32, 32)
    train_labels = torch.randint(0, 10, (1000,))
    val_data = torch.randn(200, 3, 32, 32)
    val_labels = torch.randint(0, 10, (200,))
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Pre-train teacher (in practice, load pre-trained weights)
    print("Pre-training teacher model...")
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    teacher_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):  # Quick teacher training
        teacher.train()
        for data, target in train_loader:
            teacher_optimizer.zero_grad()
            output = teacher(data)
            loss = teacher_criterion(output, target)
            loss.backward()
            teacher_optimizer.step()
    
    # Knowledge distillation
    print("\nStarting knowledge distillation...")
    distiller = KnowledgeDistiller(teacher, student, temperature=4.0, alpha=0.8)
    
    train_losses, val_accuracies = distiller.train_student(
        train_loader, val_loader, epochs=20
    )
    
    # Compare final performance
    teacher_acc, student_acc = distiller.compare_models(val_loader)
    
    return teacher, student, distiller

if __name__ == "__main__":
    teacher, student, distiller = distillation_example()
```

### Feature Distillation

Feature distillation transfers intermediate representations, not just final outputs.

```python
class FeatureDistillationModel(nn.Module):
    """Student model with feature matching capabilities"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Feature adaptation layers
        self.feature_adapter1 = nn.Conv2d(32, 64, 1)  # Match teacher feature 1
        self.feature_adapter2 = nn.Conv2d(64, 128, 1)  # Match teacher feature 2
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, return_features=False):
        # Layer 1
        x1 = self.relu(self.bn1(self.conv1(x)))
        adapted_feat1 = self.feature_adapter1(x1)
        
        # Layer 2  
        x2 = self.relu(self.bn2(self.conv2(x1)))
        adapted_feat2 = self.feature_adapter2(x2)
        
        # Layer 3
        x3 = self.relu(self.bn3(self.conv3(x2)))
        
        # Classification
        x_pooled = self.pool(x3)
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        logits = self.classifier(x_flat)
        
        if return_features:
            return logits, [adapted_feat1, adapted_feat2]
        return logits

class FeatureDistillationLoss(nn.Module):
    """Loss function for feature distillation"""
    
    def __init__(self, temperature=3.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for feature distillation
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_logits, teacher_logits, student_features, 
                teacher_features, labels):
        """
        Compute feature distillation loss
        
        Args:
            student_logits: Student final outputs
            teacher_logits: Teacher final outputs
            student_features: List of student intermediate features
            teacher_features: List of teacher intermediate features
            labels: Ground truth labels
        """
        # Classification loss
        class_loss = self.ce_loss(student_logits, labels)
        
        # Knowledge distillation loss
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = self.kl_loss(student_log_probs, teacher_probs)
        
        # Feature distillation loss
        feature_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Normalize features
            s_feat_norm = F.normalize(s_feat, p=2, dim=1)
            t_feat_norm = F.normalize(t_feat, p=2, dim=1)
            
            # Feature matching loss
            feature_loss += self.mse_loss(s_feat_norm, t_feat_norm)
        
        # Combined loss
        total_loss = (
            self.alpha * class_loss +
            (1 - self.alpha - self.beta) * (self.temperature ** 2) * kd_loss +
            self.beta * feature_loss
        )
        
        return total_loss, class_loss, kd_loss, feature_loss

class AdvancedKnowledgeDistiller:
    """Advanced distiller with feature matching"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, 
                 alpha=0.5, beta=0.3):
        self.teacher = teacher_model
        self.student = student_model
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.distill_loss = FeatureDistillationLoss(temperature, alpha, beta)
    
    def extract_teacher_features(self, x):
        """Extract intermediate features from teacher"""
        features = []
        
        # This assumes teacher has specific layer structure
        # In practice, you'd register hooks or modify the teacher model
        with torch.no_grad():
            x1 = self.teacher.relu(self.teacher.bn1(self.teacher.conv1(x)))
            features.append(x1)
            
            x2 = self.teacher.relu(self.teacher.bn2(self.teacher.conv2(x1)))
            features.append(x2)
            
            x3 = self.teacher.relu(self.teacher.bn3(self.teacher.conv3(x2)))
            
            x_pooled = self.teacher.pool(x3)
            x_flat = x_pooled.view(x_pooled.size(0), -1)
            logits = self.teacher.classifier(x_flat)
        
        return logits, features
    
    def train_step(self, data, target, optimizer):
        """Single training step with feature distillation"""
        optimizer.zero_grad()
        
        # Get teacher outputs and features
        teacher_logits, teacher_features = self.extract_teacher_features(data)
        
        # Get student outputs and features
        student_logits, student_features = self.student(data, return_features=True)
        
        # Compute loss
        total_loss, class_loss, kd_loss, feat_loss = self.distill_loss(
            student_logits, teacher_logits, student_features, 
            teacher_features, target
        )
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'kd_loss': kd_loss.item(),
            'feature_loss': feat_loss.item()
        }

# Attention Transfer (another advanced technique)
class AttentionTransfer:
    """Implement attention transfer mechanism"""
    
    @staticmethod
    def attention_map(feature_map):
        """Compute attention map from feature map"""
        # Spatial attention: sum across channels
        attention = torch.mean(torch.abs(feature_map), dim=1, keepdim=True)
        
        # Normalize
        batch_size, _, h, w = attention.shape
        attention = attention.view(batch_size, -1)
        attention = F.softmax(attention, dim=1)
        attention = attention.view(batch_size, 1, h, w)
        
        return attention
    
    @staticmethod
    def attention_transfer_loss(student_features, teacher_features):
        """Compute attention transfer loss"""
        total_loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Compute attention maps
            s_attention = AttentionTransfer.attention_map(s_feat)
            t_attention = AttentionTransfer.attention_map(t_feat)
            
            # L2 loss between attention maps
            loss = F.mse_loss(s_attention, t_attention)
            total_loss += loss
        
        return total_loss

# Example usage
def feature_distillation_example():
    """Example of feature distillation"""
    # Create models (assuming modified teacher for feature extraction)
    teacher = TeacherModel(num_classes=10)
    student = FeatureDistillationModel(num_classes=10)
    
    # Create sample data
    data = torch.randn(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))
    
    # Initialize distiller
    distiller = AdvancedKnowledgeDistiller(teacher, student)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    # Single training step
    losses = distiller.train_step(data, target, optimizer)
    
    print("Feature Distillation Losses:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value:.4f}")
    
    return teacher, student, distiller

if __name__ == "__main__":
    feature_distillation_example()
```

### Advanced Implementation Techniques

```python
class MultiTeacherDistillation:
    """Distillation from multiple teacher models"""
    
    def __init__(self, teachers, student, weights=None):
        self.teachers = teachers
        self.student = student
        
        # Freeze all teachers
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Weights for combining teacher outputs
        self.weights = weights or [1.0 / len(teachers)] * len(teachers)
    
    def ensemble_teacher_output(self, x):
        """Combine outputs from multiple teachers"""
        teacher_outputs = []
        
        with torch.no_grad():
            for teacher in self.teachers:
                output = teacher(x)
                teacher_outputs.append(output)
        
        # Weighted combination
        ensemble_output = torch.zeros_like(teacher_outputs[0])
        for output, weight in zip(teacher_outputs, self.weights):
            ensemble_output += weight * output
        
        return ensemble_output
    
    def train_step(self, data, target, optimizer, temperature=3.0):
        """Training step with multiple teachers"""
        optimizer.zero_grad()
        
        # Get ensemble teacher output
        teacher_logits = self.ensemble_teacher_output(data)
        
        # Get student output
        student_logits = self.student(data)
        
        # Compute distillation loss
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        class_loss = F.cross_entropy(student_logits, target)
        
        total_loss = 0.7 * (temperature ** 2) * kd_loss + 0.3 * class_loss
        
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()

class OnlineDistillation:
    """Online knowledge distillation - students teach each other"""
    
    def __init__(self, students):
        self.students = students
        self.num_students = len(students)
    
    def train_step(self, data, target, optimizers, temperature=3.0):
        """One step of online distillation"""
        student_outputs = []
        
        # Forward pass for all students
        for student in self.students:
            output = student(data)
            student_outputs.append(output)
        
        # Compute ensemble output (exclude current student)
        losses = []
        
        for i, (student, optimizer) in enumerate(zip(self.students, optimizers)):
            optimizer.zero_grad()
            
            # Current student output
            student_logits = student_outputs[i]
            
            # Ensemble of other students
            other_outputs = [student_outputs[j] for j in range(self.num_students) if j != i]
            ensemble_logits = torch.mean(torch.stack(other_outputs), dim=0)
            
            # Distillation loss
            teacher_probs = F.softmax(ensemble_logits / temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            class_loss = F.cross_entropy(student_logits, target)
            
            total_loss = 0.5 * (temperature ** 2) * kd_loss + 0.5 * class_loss
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
        
        return losses

# Progressive Knowledge Distillation
class ProgressiveDistillation:
    """Progressive knowledge distillation with curriculum learning"""
    
    def __init__(self, teacher, student, num_stages=3):
        self.teacher = teacher
        self.student = student
        self.num_stages = num_stages
        self.current_stage = 0
        
        # Different temperatures for different stages
        self.temperatures = [1.0, 3.0, 5.0]
        # Different alpha values for different stages
        self.alphas = [0.3, 0.5, 0.7]
    
    def should_advance_stage(self, current_accuracy, target_accuracy):
        """Check if should advance to next distillation stage"""
        return current_accuracy >= target_accuracy and self.current_stage < self.num_stages - 1
    
    def get_current_hyperparams(self):
        """Get current stage hyperparameters"""
        stage_idx = min(self.current_stage, len(self.temperatures) - 1)
        return {
            'temperature': self.temperatures[stage_idx],
            'alpha': self.alphas[stage_idx]
        }
    
    def advance_stage(self):
        """Advance to next distillation stage"""
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            print(f"Advanced to distillation stage {self.current_stage + 1}")
            return True
        return False

# Example of complete distillation pipeline
def complete_distillation_pipeline():
    """Complete knowledge distillation pipeline with all techniques"""
    # Create models
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)
    
    # Method 1: Standard Knowledge Distillation
    print("=== Standard Knowledge Distillation ===")
    standard_distiller = KnowledgeDistiller(teacher, student)
    
    # Method 2: Multi-teacher Distillation
    print("\n=== Multi-Teacher Distillation ===")
    teachers = [TeacherModel(num_classes=10) for _ in range(3)]
    multi_distiller = MultiTeacherDistillation(teachers, student)
    
    # Method 3: Online Distillation
    print("\n=== Online Distillation ===") 
    students = [StudentModel(num_classes=10) for _ in range(3)]
    online_distiller = OnlineDistillation(students)
    
    # Method 4: Progressive Distillation
    print("\n=== Progressive Distillation ===")
    progressive_distiller = ProgressiveDistillation(teacher, student)
    
    print("All distillation methods initialized successfully!")
    
    return {
        'standard': standard_distiller,
        'multi_teacher': multi_distiller,
        'online': online_distiller,
        'progressive': progressive_distiller
    }

if __name__ == "__main__":
    distillers = complete_distillation_pipeline()
```

## Model Compression

Model compression encompasses various techniques to reduce model size, memory usage, and computational requirements while preserving performance.

### Understanding Model Compression

**Why Model Compression is Important:**
- Mobile and edge device deployment constraints
- Reduced inference latency and energy consumption
- Lower storage and bandwidth requirements
- Cost-effective cloud deployment

**Compression Techniques Overview:**
1. **Weight Sharing**: Multiple weights share the same value
2. **Low-rank Factorization**: Decompose weight matrices into smaller matrices
3. **Huffman Coding**: Compress weight storage using variable-length encoding
4. **Tensor Decomposition**: Decompose high-dimensional tensors

### Weight Sharing Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import pickle

class WeightSharing:
    """Implement weight sharing for model compression"""
    
    def __init__(self, num_clusters=256):
        self.num_clusters = num_clusters
        self.codebooks = {}
        self.indices = {}
    
    def kmeans_clustering(self, weights, k):
        """K-means clustering for weight quantization"""
        weights_flat = weights.flatten()
        
        # Initialize centroids
        centroids = np.linspace(weights_flat.min(), weights_flat.max(), k)
        
        for _ in range(10):  # 10 iterations
            # Assign points to clusters
            distances = np.abs(weights_flat[:, np.newaxis] - centroids)
            cluster_assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(k):
                mask = cluster_assignments == i
                if np.sum(mask) > 0:
                    centroids[i] = np.mean(weights_flat[mask])
        
        return centroids, cluster_assignments
    
    def compress_layer(self, layer_weights, layer_name):
        """Compress a single layer using weight sharing"""
        original_shape = layer_weights.shape
        
        # Apply k-means clustering
        centroids, assignments = self.kmeans_clustering(
            layer_weights.detach().numpy(), 
            self.num_clusters
        )
        
        # Store codebook and indices
        self.codebooks[layer_name] = centroids
        self.indices[layer_name] = assignments.reshape(original_shape)
        
        # Reconstruct weights
        reconstructed = centroids[assignments].reshape(original_shape)
        
        return torch.tensor(reconstructed, dtype=layer_weights.dtype)
    
    def compress_model(self, model):
        """Compress entire model using weight sharing"""
        compressed_model = type(model)()
        compressed_model.load_state_dict(model.state_dict())
        
        compression_stats = {}
        
        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                original_weights = module.weight.data
                original_size = original_weights.numel() * 4  # 4 bytes per float32
                
                # Compress weights
                compressed_weights = self.compress_layer(original_weights, name)
                module.weight.data = compressed_weights
                
                # Calculate compression ratio
                codebook_size = len(self.codebooks[name]) * 4  # 4 bytes per float32
                indices_size = len(self.indices[name].flatten()) * 1  # 1 byte per index (256 clusters)
                compressed_size = codebook_size + indices_size
                
                compression_stats[name] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': original_size / compressed_size,
                    'num_unique_weights': len(np.unique(compressed_weights.numpy()))
                }
        
        return compressed_model, compression_stats
    
    def save_compression_data(self, filepath):
        """Save codebooks and indices for deployment"""
        compression_data = {
            'codebooks': self.codebooks,
            'indices': self.indices,
            'num_clusters': self.num_clusters
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(compression_data, f)
    
    def load_compression_data(self, filepath):
        """Load compression data"""
        with open(filepath, 'rb') as f:
            compression_data = pickle.load(f)
        
        self.codebooks = compression_data['codebooks']
        self.indices = compression_data['indices']
        self.num_clusters = compression_data['num_clusters']

# Example usage
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def weight_sharing_example():
    """Demonstrate weight sharing compression"""
    # Create and initialize model
    model = SimpleModel()
    
    # Initialize weight sharing compressor
    compressor = WeightSharing(num_clusters=64)
    
    # Compress model
    compressed_model, stats = compressor.compress_model(model)
    
    # Print compression statistics
    print("=== Weight Sharing Compression Results ===")
    total_original = 0
    total_compressed = 0
    
    for layer_name, layer_stats in stats.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Original size: {layer_stats['original_size']:,} bytes")
        print(f"  Compressed size: {layer_stats['compressed_size']:,} bytes")
        print(f"  Compression ratio: {layer_stats['compression_ratio']:.2f}x")
        print(f"  Unique weights: {layer_stats['num_unique_weights']}")
        
        total_original += layer_stats['original_size']
        total_compressed += layer_stats['compressed_size']
    
    overall_ratio = total_original / total_compressed
    print(f"\nOverall compression ratio: {overall_ratio:.2f}x")
    print(f"Model size reduction: {(1 - total_compressed/total_original)*100:.1f}%")
    
    return model, compressed_model, compressor

if __name__ == "__main__":
    original, compressed, compressor = weight_sharing_example()
```

### Low-rank Factorization

Low-rank factorization decomposes weight matrices into products of smaller matrices, reducing the number of parameters.

```python
class LowRankFactorization:
    """Implement low-rank factorization for model compression"""
    
    @staticmethod
    def svd_decomposition(weight_matrix, rank_ratio=0.5):
        """Perform SVD decomposition with rank reduction"""
        # Convert to numpy for SVD
        W = weight_matrix.detach().numpy()
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Determine rank
        original_rank = min(W.shape)
        target_rank = max(1, int(original_rank * rank_ratio))
        
        # Truncate to target rank
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        Vt_truncated = Vt[:target_rank, :]
        
        # Create factor matrices
        A = U_truncated * np.sqrt(S_truncated)  # Shape: (out_features, rank)
        B = np.sqrt(S_truncated)[:, np.newaxis] * Vt_truncated  # Shape: (rank, in_features)
        
        return torch.tensor(A, dtype=weight_matrix.dtype), torch.tensor(B, dtype=weight_matrix.dtype)
    
    @staticmethod
    def factorize_linear_layer(linear_layer, rank_ratio=0.5):
        """Factorize a linear layer into two smaller layers"""
        original_weight = linear_layer.weight
        out_features, in_features = original_weight.shape
        
        # Perform SVD factorization
        A, B = LowRankFactorization.svd_decomposition(original_weight, rank_ratio)
        rank = A.shape[1]
        
        # Create two new linear layers
        layer1 = nn.Linear(in_features, rank, bias=False)
        layer2 = nn.Linear(rank, out_features, bias=(linear_layer.bias is not None))
        
        # Set weights
        layer1.weight.data = B
        layer2.weight.data = A
        
        if linear_layer.bias is not None:
            layer2.bias.data = linear_layer.bias.data
        
        # Calculate compression statistics
        original_params = out_features * in_features
        new_params = rank * (in_features + out_features)
        compression_ratio = original_params / new_params
        
        return nn.Sequential(layer1, layer2), compression_ratio
    
    @staticmethod
    def factorize_conv2d_layer(conv_layer, rank_ratio=0.5):
        """Factorize a conv2d layer using channel-wise decomposition"""
        weight = conv_layer.weight  # Shape: (out_channels, in_channels, h, w)
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        if kernel_h == 1 and kernel_w == 1:
            # 1x1 convolution - treat as linear layer
            weight_2d = weight.squeeze().squeeze()
            A, B = LowRankFactorization.svd_decomposition(weight_2d, rank_ratio)
            rank = A.shape[1]
            
            # Create two 1x1 conv layers
            conv1 = nn.Conv2d(in_channels, rank, 1, 
                             stride=conv_layer.stride,
                             padding=conv_layer.padding,
                             bias=False)
            conv2 = nn.Conv2d(rank, out_channels, 1,
                             bias=(conv_layer.bias is not None))
            
            conv1.weight.data = B.unsqueeze(2).unsqueeze(3)
            conv2.weight.data = A.unsqueeze(2).unsqueeze(3)
            
            if conv_layer.bias is not None:
                conv2.bias.data = conv_layer.bias.data
            
        else:
            # For larger kernels, use separable convolution approximation
            rank = max(1, int(min(in_channels, out_channels) * rank_ratio))
            
            # Pointwise reduction
            conv1 = nn.Conv2d(in_channels, rank, 1, bias=False)
            # Depthwise spatial convolution
            conv2 = nn.Conv2d(rank, rank, (kernel_h, kernel_w),
                             stride=conv_layer.stride,
                             padding=conv_layer.padding,
                             groups=rank, bias=False)
            # Pointwise expansion
            conv3 = nn.Conv2d(rank, out_channels, 1,
                             bias=(conv_layer.bias is not None))
            
            # Initialize weights (simplified initialization)
            nn.init.kaiming_normal_(conv1.weight)
            nn.init.kaiming_normal_(conv2.weight)
            nn.init.kaiming_normal_(conv3.weight)
            
            if conv_layer.bias is not None:
                conv3.bias.data = conv_layer.bias.data
            
            return nn.Sequential(conv1, conv2, conv3), None
        
        original_params = out_channels * in_channels * kernel_h * kernel_w
        new_params = rank * (in_channels + out_channels)
        compression_ratio = original_params / new_params
        
        return nn.Sequential(conv1, conv2), compression_ratio

class FactorizedModel(nn.Module):
    """Model with factorized layers"""
    
    def __init__(self, original_model, rank_ratio=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.compression_stats = {}
        
        # Factorize each layer
        for name, module in original_model.named_modules():
            if isinstance(module, nn.Linear):
                factorized_layer, ratio = LowRankFactorization.factorize_linear_layer(
                    module, rank_ratio
                )
                self.layers.append(factorized_layer)
                self.compression_stats[name] = ratio
                
            elif isinstance(module, nn.Conv2d):
                factorized_layer, ratio = LowRankFactorization.factorize_conv2d_layer(
                    module, rank_ratio
                )
                self.layers.append(factorized_layer)
                if ratio:
                    self.compression_stats[name] = ratio
                    
            elif isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.AdaptiveAvgPool2d)):
                self.layers.append(module)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def low_rank_factorization_example():
    """Demonstrate low-rank factorization"""
    # Create original model
    original_model = SimpleModel()
    
    # Create factorized model
    factorized_model = FactorizedModel(original_model, rank_ratio=0.3)
    
    # Compare model sizes
    original_params = sum(p.numel() for p in original_model.parameters())
    factorized_params = sum(p.numel() for p in factorized_model.parameters())
    
    print("=== Low-Rank Factorization Results ===")
    print(f"Original parameters: {original_params:,}")
    print(f"Factorized parameters: {factorized_params:,}")
    print(f"Overall compression ratio: {original_params/factorized_params:.2f}x")
    print(f"Parameter reduction: {(1-factorized_params/original_params)*100:.1f}%")
    
    # Test inference
    test_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        original_output = original_model(test_input)
        factorized_output = factorized_model(test_input)
    
    # Compare outputs (they will be different due to factorization)
    output_diff = torch.norm(original_output - factorized_output).item()
    print(f"Output difference (L2 norm): {output_diff:.4f}")
    
    return original_model, factorized_model

if __name__ == "__main__":
    original, factorized = low_rank_factorization_example()
```

### Huffman Coding for Weight Storage

```python
import heapq
from collections import Counter, defaultdict
import struct

class HuffmanCoding:
    """Implement Huffman coding for weight compression"""
    
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
        self.tree = None
    
    def _build_frequency_table(self, weights):
        """Build frequency table for weights"""
        # Quantize weights to reduce vocabulary size
        quantized_weights = np.round(weights * 1000).astype(int)  # 3 decimal precision
        return Counter(quantized_weights.flatten())
    
    def _build_huffman_tree(self, frequency_table):
        """Build Huffman tree from frequency table"""
        heap = [[freq, [[symbol, ""]]] for symbol, freq in frequency_table.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
                
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        return heap[0]
    
    def _extract_codes(self, huffman_tree):
        """Extract codes from Huffman tree"""
        if len(huffman_tree) == 2:  # Only one symbol
            symbol, code = huffman_tree[1][0]
            self.codes[symbol] = '0'
            self.reverse_codes['0'] = symbol
        else:
            for pair in huffman_tree[1:]:
                symbol, code = pair[0], pair[1]
                self.codes[symbol] = code
                self.reverse_codes[code] = symbol
    
    def build_codes(self, weights):
        """Build Huffman codes for weights"""
        frequency_table = self._build_frequency_table(weights)
        
        if len(frequency_table) == 1:
            # Special case: only one unique value
            symbol = list(frequency_table.keys())[0]
            self.codes[symbol] = '0'
            self.reverse_codes['0'] = symbol
        else:
            huffman_tree = self._build_huffman_tree(frequency_table)
            self._extract_codes(huffman_tree)
    
    def encode_weights(self, weights):
        """Encode weights using Huffman coding"""
        quantized_weights = np.round(weights * 1000).astype(int)
        
        encoded_bits = []
        for weight in quantized_weights.flatten():
            if weight in self.codes:
                encoded_bits.append(self.codes[weight])
            else:
                # Handle unseen weights (should not happen with proper training)
                encoded_bits.append('0')
        
        return ''.join(encoded_bits)
    
    def decode_weights(self, encoded_string, original_shape):
        """Decode weights from Huffman-encoded string"""
        decoded_weights = []
        i = 0
        
        while i < len(encoded_string):
            for length in range(1, len(encoded_string) - i + 1):
                code = encoded_string[i:i+length]
                if code in self.reverse_codes:
                    decoded_weights.append(self.reverse_codes[code])
                    i += length
                    break
            else:
                # If no valid code found, skip
                i += 1
        
        # Convert back to float and reshape
        decoded_array = np.array(decoded_weights) / 1000.0
        
        # Pad or truncate to match original shape
        total_elements = np.prod(original_shape)
        if len(decoded_array) < total_elements:
            # Pad with zeros
            decoded_array = np.pad(decoded_array, (0, total_elements - len(decoded_array)))
        elif len(decoded_array) > total_elements:
            # Truncate
            decoded_array = decoded_array[:total_elements]
        
        return decoded_array.reshape(original_shape)
    
    def compress_model_weights(self, model):
        """Compress all model weights using Huffman coding"""
        compressed_data = {}
        compression_stats = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data.numpy()
                original_shape = weights.shape
                
                # Build Huffman codes for this layer
                layer_huffman = HuffmanCoding()
                layer_huffman.build_codes(weights)
                
                # Encode weights
                encoded_string = layer_huffman.encode_weights(weights)
                
                # Calculate compression ratio
                original_bits = weights.size * 32  # 32 bits per float32
                compressed_bits = len(encoded_string)
                compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 1
                
                compressed_data[name] = {
                    'encoded_weights': encoded_string,
                    'codes': layer_huffman.codes,
                    'reverse_codes': layer_huffman.reverse_codes,
                    'original_shape': original_shape
                }
                
                compression_stats[name] = {
                    'original_bits': original_bits,
                    'compressed_bits': compressed_bits,
                    'compression_ratio': compression_ratio,
                    'unique_weights': len(layer_huffman.codes)
                }
        
        return compressed_data, compression_stats
    
    def decompress_model_weights(self, compressed_data, model):
        """Decompress weights and load into model"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and name in compressed_data:
                data = compressed_data[name]
                
                # Create decoder
                decoder = HuffmanCoding()
                decoder.reverse_codes = data['reverse_codes']
                
                # Decode weights
                decoded_weights = decoder.decode_weights(
                    data['encoded_weights'], 
                    data['original_shape']
                )
                
                # Load into model
                module.weight.data = torch.tensor(decoded_weights, dtype=torch.float32)

def huffman_compression_example():
    """Demonstrate Huffman coding compression"""
    # Create model
    model = SimpleModel()
    
    # Initialize Huffman compressor
    huffman = HuffmanCoding()
    
    # Compress model
    compressed_data, stats = huffman.compress_model_weights(model)
    
    # Print compression statistics
    print("=== Huffman Coding Compression Results ===")
    total_original_bits = 0
    total_compressed_bits = 0
    
    for layer_name, layer_stats in stats.items():
        print(f"\nLayer: {layer_name}")
        print(f"  Original bits: {layer_stats['original_bits']:,}")
        print(f"  Compressed bits: {layer_stats['compressed_bits']:,}")
        print(f"  Compression ratio: {layer_stats['compression_ratio']:.2f}x")
        print(f"  Unique weights: {layer_stats['unique_weights']}")
        
        total_original_bits += layer_stats['original_bits']
        total_compressed_bits += layer_stats['compressed_bits']
    
    overall_ratio = total_original_bits / total_compressed_bits
    print(f"\nOverall compression ratio: {overall_ratio:.2f}x")
    print(f"Storage reduction: {(1 - total_compressed_bits/total_original_bits)*100:.1f}%")
    
    # Test decompression
    decompressed_model = SimpleModel()
    huffman.decompress_model_weights(compressed_data, decompressed_model)
    
    # Test that decompressed model works
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model(test_input)
        decompressed_output = decompressed_model(test_input)
    
    reconstruction_error = torch.norm(original_output - decompressed_output).item()
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    return model, decompressed_model, compressed_data

if __name__ == "__main__":
    original, decompressed, compressed_data = huffman_compression_example()
```

### Tensor Decomposition

```python
import tensorly as tl
from tensorly.decomposition import tucker, parafac

class TensorDecomposition:
    """Implement tensor decomposition for model compression"""
    
    @staticmethod
    def tucker_decomposition_conv(conv_layer, ranks=None):
        """Apply Tucker decomposition to conv layer"""
        weight = conv_layer.weight.data
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Default ranks (50% of original dimensions)
        if ranks is None:
            ranks = [out_channels//2, in_channels//2, kernel_h, kernel_w]
        
        # Perform Tucker decomposition
        tl.set_backend('pytorch')
        core, factors = tucker(weight, rank=ranks)
        
        # Create new layers
        # First layer: 1x1 conv for input channel reduction
        layer1 = nn.Conv2d(in_channels, ranks[1], 1, bias=False)
        layer1.weight.data = factors[1].t().unsqueeze(-1).unsqueeze(-1)
        
        # Second layer: spatial convolution with core tensor
        layer2 = nn.Conv2d(ranks[1], ranks[0], (kernel_h, kernel_w),
                          stride=conv_layer.stride,
                          padding=conv_layer.padding, bias=False)
        
        # Reshape core for conv2d weights
        core_reshaped = core.permute(0, 1, 2, 3)
        layer2.weight.data = core_reshaped
        
        # Third layer: 1x1 conv for output channel expansion
        layer3 = nn.Conv2d(ranks[0], out_channels, 1,
                          bias=(conv_layer.bias is not None))
        layer3.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)
        
        if conv_layer.bias is not None:
            layer3.bias.data = conv_layer.bias.data
        
        # Calculate compression ratio
        original_params = out_channels * in_channels * kernel_h * kernel_w
        new_params = (
            ranks[1] * in_channels +  # layer1
            ranks[0] * ranks[1] * kernel_h * kernel_w +  # layer2
            out_channels * ranks[0]  # layer3
        )
        compression_ratio = original_params / new_params
        
        return nn.Sequential(layer1, layer2, layer3), compression_ratio
    
    @staticmethod
    def cp_decomposition_conv(conv_layer, rank):
        """Apply CP (CANDECOMP/PARAFAC) decomposition to conv layer"""
        weight = conv_layer.weight.data
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        
        # Perform CP decomposition
        tl.set_backend('pytorch')
        cp_tensor = parafac(weight, rank=rank)
        factors = cp_tensor[1]  # Get factor matrices
        
        # Create separable convolution layers
        # Point-wise convolution
        layer1 = nn.Conv2d(in_channels, rank, 1, bias=False)
        layer1.weight.data = factors[1].t().unsqueeze(-1).unsqueeze(-1)
        
        # Depth-wise convolution (approximation)
        layer2 = nn.Conv2d(rank, rank, (kernel_h, kernel_w),
                          stride=conv_layer.stride,
                          padding=conv_layer.padding,
                          groups=rank, bias=False)
        
        # Initialize depth-wise weights from spatial factors
        for i in range(rank):
            layer2.weight.data[i, 0, :, :] = torch.outer(factors[2][:, i], factors[3][:, i])
        
        # Point-wise convolution for output
        layer3 = nn.Conv2d(rank, out_channels, 1,
                          bias=(conv_layer.bias is not None))
        layer3.weight.data = factors[0].unsqueeze(-1).unsqueeze(-1)
        
        if conv_layer.bias is not None:
            layer3.bias.data = conv_layer.bias.data
        
        # Calculate compression ratio
        original_params = out_channels * in_channels * kernel_h * kernel_w
        new_params = (
            rank * in_channels +  # layer1
            rank * kernel_h * kernel_w +  # layer2 (approximation)
            out_channels * rank  # layer3
        )
        compression_ratio = original_params / new_params
        
        return nn.Sequential(layer1, layer2, layer3), compression_ratio

class TensorDecomposedModel(nn.Module):
    """Model with tensor-decomposed layers"""
    
    def __init__(self, original_model, decomposition_type='tucker', ranks=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.compression_stats = {}
        self.decomposition_type = decomposition_type
        
        for name, module in original_model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
                if decomposition_type == 'tucker':
                    decomposed_layer, ratio = TensorDecomposition.tucker_decomposition_conv(
                        module, ranks
                    )
                elif decomposition_type == 'cp':
                    rank = ranks[0] if ranks else min(module.out_channels, module.in_channels) // 2
                    decomposed_layer, ratio = TensorDecomposition.cp_decomposition_conv(
                        module, rank
                    )
                
                self.layers.append(decomposed_layer)
                self.compression_stats[name] = ratio
                
            else:
                # Keep other layers unchanged
                self.layers.append(module)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def tensor_decomposition_example():
    """Demonstrate tensor decomposition"""
    try:
        import tensorly as tl
        
        # Create original model
        original_model = SimpleModel()
        
        # Apply Tucker decomposition
        tucker_model = TensorDecomposedModel(
            original_model, 
            decomposition_type='tucker',
            ranks=[32, 32, 3, 3]  # Reduced ranks
        )
        
        # Apply CP decomposition
        cp_model = TensorDecomposedModel(
            original_model,
            decomposition_type='cp', 
            ranks=[16]  # CP rank
        )
        
        # Compare model sizes
        original_params = sum(p.numel() for p in original_model.parameters())
        tucker_params = sum(p.numel() for p in tucker_model.parameters())
        cp_params = sum(p.numel() for p in cp_model.parameters())
        
        print("=== Tensor Decomposition Results ===")
        print(f"Original parameters: {original_params:,}")
        print(f"Tucker decomposed parameters: {tucker_params:,}")
        print(f"CP decomposed parameters: {cp_params:,}")
        print(f"Tucker compression ratio: {original_params/tucker_params:.2f}x")
        print(f"CP compression ratio: {original_params/cp_params:.2f}x")
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        
        with torch.no_grad():
            original_output = original_model(test_input)
            tucker_output = tucker_model(test_input)
            cp_output = cp_model(test_input)
        
        tucker_error = torch.norm(original_output - tucker_output).item()
        cp_error = torch.norm(original_output - cp_output).item()
        
        print(f"Tucker reconstruction error: {tucker_error:.4f}")
        print(f"CP reconstruction error: {cp_error:.4f}")
        
        return original_model, tucker_model, cp_model
        
    except ImportError:
        print("TensorLy not installed. Install with: pip install tensorly")
        return None, None, None

if __name__ == "__main__":
    original, tucker, cp = tensor_decomposition_example()
```

### Combined Compression Pipeline

```python
class ComprehensiveModelCompressor:
    """Combine multiple compression techniques"""
    
    def __init__(self, model):
        self.original_model = model
        self.compression_pipeline = []
        self.compression_stats = {}
    
    def add_quantization(self, bits=8):
        """Add quantization to pipeline"""
        self.compression_pipeline.append(('quantization', {'bits': bits}))
        return self
    
    def add_pruning(self, sparsity=0.5):
        """Add pruning to pipeline"""
        self.compression_pipeline.append(('pruning', {'sparsity': sparsity}))
        return self
    
    def add_weight_sharing(self, clusters=256):
        """Add weight sharing to pipeline"""
        self.compression_pipeline.append(('weight_sharing', {'clusters': clusters}))
        return self
    
    def add_low_rank(self, rank_ratio=0.5):
        """Add low-rank factorization to pipeline"""
        self.compression_pipeline.append(('low_rank', {'rank_ratio': rank_ratio}))
        return self
    
    def compress(self):
        """Apply all compression techniques in sequence"""
        current_model = self.original_model
        
        for technique, params in self.compression_pipeline:
            print(f"Applying {technique} compression...")
            
            if technique == 'quantization':
                # Apply quantization (simplified)
                current_model = self._apply_quantization(current_model, params)
                
            elif technique == 'pruning':
                # Apply pruning
                current_model = self._apply_pruning(current_model, params)
                
            elif technique == 'weight_sharing':
                # Apply weight sharing
                current_model = self._apply_weight_sharing(current_model, params)
                
            elif technique == 'low_rank':
                # Apply low-rank factorization
                current_model = self._apply_low_rank(current_model, params)
        
        return current_model
    
    def _apply_quantization(self, model, params):
        """Apply quantization compression"""
        # Simplified quantization
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Quantize weights to 8-bit
                weights = module.weight.data
                min_val, max_val = weights.min(), weights.max()
                scale = (max_val - min_val) / 255
                zero_point = -min_val / scale
                
                quantized = torch.round(weights / scale + zero_point)
                quantized = torch.clamp(quantized, 0, 255)
                
                # Dequantize (in practice, would keep quantized)
                dequantized = scale * (quantized - zero_point)
                module.weight.data = dequantized
        
        return model
    
    def _apply_pruning(self, model, params):
        """Apply pruning compression"""
        pruner = MagnitudePruner(model)
        pruner.prune_by_percentage(params['sparsity'] * 100)
        return model
    
    def _apply_weight_sharing(self, model, params):
        """Apply weight sharing compression"""
        compressor = WeightSharing(num_clusters=params['clusters'])
        compressed_model, _ = compressor.compress_model(model)
        return compressed_model
    
    def _apply_low_rank(self, model, params):
        """Apply low-rank factorization"""
        factorized_model = FactorizedModel(model, params['rank_ratio'])
        return factorized_model
    
    def analyze_compression(self, original_model, compressed_model):
        """Analyze compression results"""
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        compression_ratio = original_params / compressed_params
        parameter_reduction = (1 - compressed_params / original_params) * 100
        
        # Test accuracy preservation (simplified)
        test_input = torch.randn(10, 3, 32, 32)
        
        with torch.no_grad():
            original_output = original_model(test_input)
            compressed_output = compressed_model(test_input)
        
        output_mse = torch.mean((original_output - compressed_output) ** 2).item()
        
        results = {
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compression_ratio,
            'parameter_reduction_percent': parameter_reduction,
            'output_mse': output_mse,
            'compression_pipeline': self.compression_pipeline
        }
        
        return results

def comprehensive_compression_example():
    """Demonstrate comprehensive model compression"""
    # Create original model
    original_model = SimpleModel()
    
    # Create compressor with multiple techniques
    compressor = ComprehensiveModelCompressor(original_model)
    
    # Add compression techniques
    compressed_model = (compressor
                       .add_pruning(sparsity=0.3)
                       .add_quantization(bits=8)
                       .add_weight_sharing(clusters=128)
                       .compress())
    
    # Analyze results
    results = compressor.analyze_compression(original_model, compressed_model)
    
    print("=== Comprehensive Compression Results ===")
    print(f"Original parameters: {results['original_parameters']:,}")
    print(f"Compressed parameters: {results['compressed_parameters']:,}")
    print(f"Compression ratio: {results['compression_ratio']:.2f}x")
    print(f"Parameter reduction: {results['parameter_reduction_percent']:.1f}%")
    print(f"Output MSE: {results['output_mse']:.6f}")
    print(f"Compression pipeline: {[step[0] for step in results['compression_pipeline']]}")
    
    return original_model, compressed_model, results

if __name__ == "__main__":
    original, compressed, results = comprehensive_compression_example()
```

## Learning Objectives

By the end of this section, you should be able to:

### Core Understanding
- **Explain the fundamental principles** behind each optimization technique and when to apply them
- **Identify bottlenecks** in model deployment scenarios (memory, latency, accuracy trade-offs)
- **Compare and contrast** different optimization approaches based on use case requirements
- **Understand the mathematical foundations** of quantization, pruning, distillation, and compression

### Practical Implementation
- **Implement quantization** using both post-training and quantization-aware training approaches
- **Apply structured and unstructured pruning** with proper mask management and iterative schedules
- **Set up knowledge distillation** pipelines with appropriate loss functions and temperature scaling
- **Deploy model compression techniques** including weight sharing, low-rank factorization, and tensor decomposition

### Advanced Applications
- **Design optimization pipelines** that combine multiple techniques effectively
- **Evaluate optimization trade-offs** between model size, speed, and accuracy
- **Optimize models for specific deployment targets** (mobile, edge devices, cloud inference)
- **Debug and troubleshoot** optimization-related issues in production environments

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Implement post-training quantization and measure the accuracy-speed trade-off  
□ Set up quantization-aware training with proper calibration datasets  
□ Apply magnitude-based pruning with iterative sparsity schedules  
□ Distinguish between structured and unstructured pruning trade-offs  
□ Implement teacher-student knowledge distillation with custom loss functions  
□ Apply feature distillation and attention transfer techniques  
□ Compress models using weight sharing and evaluate compression ratios  
□ Perform low-rank factorization of linear and convolutional layers  
□ Use tensor decomposition (Tucker/CP) for model compression  
□ Combine multiple optimization techniques in a single pipeline  
□ Benchmark optimized models against original models  
□ Deploy optimized models to target hardware platforms  

### Practical Exercises

**Exercise 1: Quantization Pipeline**
```python
# TODO: Implement a complete quantization pipeline
class QuantizationPipeline:
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
    
    def post_training_quantization(self):
        # Your implementation here
        pass
    
    def quantization_aware_training(self, train_loader, epochs=10):
        # Your implementation here
        pass
    
    def benchmark_quantized_model(self):
        # Compare FP32 vs INT8 performance
        pass

# Test your implementation
model = torchvision.models.resnet18(pretrained=True)
# pipeline = QuantizationPipeline(model, calibration_data)
# quantized_model = pipeline.post_training_quantization()
```

**Exercise 2: Structured Pruning Implementation**
```python
# TODO: Implement channel pruning for ResNet blocks
class ChannelPruner:
    def __init__(self, model, target_flops_reduction=0.5):
        self.model = model
        self.target_flops_reduction = target_flops_reduction
    
    def analyze_layer_importance(self):
        # Compute importance scores for each layer
        pass
    
    def prune_channels(self, layer, num_channels_to_prune):
        # Remove least important channels
        pass
    
    def update_dependent_layers(self, pruned_layer, next_layer):
        # Update input dimensions of subsequent layers
        pass

# Test: Prune 30% of channels from each conv layer
```

**Exercise 3: Multi-Teacher Distillation**
```python
# TODO: Implement ensemble knowledge distillation
class EnsembleDistillation:
    def __init__(self, teachers, student, teacher_weights=None):
        self.teachers = teachers
        self.student = student
        self.teacher_weights = teacher_weights
    
    def compute_ensemble_logits(self, x):
        # Combine outputs from multiple teachers
        pass
    
    def distillation_loss(self, student_logits, ensemble_logits, targets):
        # Implement combined loss function
        pass
    
    def train_student(self, train_loader, epochs=50):
        # Training loop with ensemble distillation
        pass

# Test with 3 different teacher architectures
```

**Exercise 4: Compression Pipeline**
```python
# TODO: Build a comprehensive compression pipeline
class ModelCompressor:
    def __init__(self, model, target_compression_ratio=5.0):
        self.model = model
        self.target_ratio = target_compression_ratio
    
    def apply_pruning(self, sparsity=0.5):
        # Apply magnitude-based pruning
        pass
    
    def apply_quantization(self, bits=8):
        # Apply weight quantization
        pass
    
    def apply_knowledge_distillation(self, teacher, epochs=20):
        # Train compressed model with distillation
        pass
    
    def evaluate_compression(self):
        # Measure size, speed, and accuracy
        pass

# Test: Achieve 5x compression with <2% accuracy loss
```

## Study Materials

### Essential Reading
- **Primary:** "Efficient Deep Learning" by Gaurav Menghani (Chapters 4-7)
- **Technical:** "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding" (Han et al.)
- **Quantization:** "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al.)
- **Distillation:** "Distilling the Knowledge in a Neural Network" (Hinton et al.)
- **Pruning:** "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin)

### Video Resources
- **MIT 6.S965**: TinyML and Efficient Deep Learning Computing
- **Stanford CS230**: Deep Learning - Model Optimization Lectures
- **NVIDIA Deep Learning Institute**: Model Optimization Courses
- **PyTorch Tutorials**: Quantization and Pruning Workshops

### Hands-on Labs

**Lab 1: Mobile Model Optimization**
- Optimize a computer vision model for mobile deployment
- Apply quantization, pruning, and knowledge distillation
- Measure inference time on actual mobile devices
- Compare accuracy vs. speed trade-offs

**Lab 2: Edge Device Deployment**
- Compress a speech recognition model for edge deployment
- Use structured pruning and low-rank factorization
- Deploy to Raspberry Pi or NVIDIA Jetson
- Benchmark real-world performance

**Lab 3: Cloud Inference Optimization**
- Optimize large language models for cloud inference
- Apply advanced quantization and tensor decomposition
- Implement dynamic batching and caching strategies
- Measure throughput and cost effectiveness

### Development Environment Setup

**Required Libraries:**
```bash
# Core ML frameworks
pip install torch torchvision torchaudio
pip install tensorflow tensorflow-model-optimization

# Quantization tools
pip install intel-neural-compressor  # Intel Neural Compressor
pip install brevitas  # Quantization-aware training

# Pruning libraries
pip install torch-pruning
pip install neural-structured-learning

# Compression tools
pip install tensorly  # Tensor decomposition
pip install numpy scipy scikit-learn

# Deployment tools
pip install onnx onnxruntime
pip install tensorrt  # NVIDIA TensorRT (if available)
pip install openvino  # Intel OpenVINO

# Benchmarking
pip install memory-profiler
pip install line-profiler
pip install torchinfo  # Model analysis
```

**Hardware Requirements:**
- **GPU**: NVIDIA GPU with CUDA for training (RTX 3060 or better recommended)
- **CPU**: Multi-core processor for deployment testing
- **Memory**: 16GB+ RAM for large model optimization
- **Storage**: SSD for fast data loading during training

**Benchmarking Setup:**
```python
# Model analysis template
import torchinfo
from memory_profiler import profile
import time

def benchmark_model(model, input_shape, device='cpu'):
    """Comprehensive model benchmarking"""
    # Model summary
    summary = torchinfo.summary(model, input_shape)
    
    # Memory usage
    @profile
    def inference_test():
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape).to(device)
            output = model(dummy_input)
        return output
    
    # Timing test
    model.to(device)
    times = []
    for _ in range(100):
        start = time.time()
        output = inference_test()
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    return {
        'model_summary': summary,
        'avg_inference_time': avg_time,
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    }
```

### Industry Case Studies

**Case Study 1: MobileNet Optimization**
- Google's approach to mobile-first CNN architectures
- Depthwise separable convolutions and width multipliers
- Quantization strategies for mobile deployment

**Case Study 2: BERT Compression**
- DistilBERT: Knowledge distillation for BERT
- Structured pruning in transformer architectures
- Task-specific optimization strategies

**Case Study 3: Production Model Serving**
- Netflix's model optimization pipeline
- A/B testing optimized vs. original models
- Cost-performance trade-offs in production

### Assessment Questions

**Conceptual Questions:**
1. When would you choose quantization-aware training over post-training quantization?
2. Compare the trade-offs between structured and unstructured pruning for mobile deployment.
3. How does temperature scaling affect knowledge distillation effectiveness?
4. What are the computational complexity implications of different tensor decomposition methods?

**Technical Implementation:**
5. Implement symmetric vs. asymmetric quantization and compare their effectiveness on a given model.
6. Design a pruning schedule that achieves 90% sparsity while maintaining accuracy within 1% of the original.
7. Set up feature-level knowledge distillation between ResNet-50 (teacher) and MobileNet (student).
8. Combine quantization and pruning in a way that maximizes compression while minimizing accuracy loss.

**System Design:**
9. Design an optimization pipeline for deploying a computer vision model to mobile devices with memory constraints.
10. Create a benchmarking framework that evaluates optimization techniques across different hardware platforms.
11. Design an A/B testing setup to validate optimized models in production environments.
12. Plan a gradual rollout strategy for deploying optimized models to minimize risk.

### Debugging and Troubleshooting Guide

**Common Issues and Solutions:**

**Quantization Problems:**
- **Issue**: Accuracy drops significantly after quantization
- **Solution**: Use quantization-aware training, ensure proper calibration dataset, check for outlier weights

**Pruning Issues:**
- **Issue**: Model fails to recover accuracy after pruning
- **Solution**: Use gradual pruning schedule, ensure proper mask application during training, verify gradient flow

**Distillation Problems:**
- **Issue**: Student model not learning from teacher
- **Solution**: Adjust temperature scaling, balance distillation vs. classification loss, verify teacher model quality

**Deployment Issues:**
- **Issue**: Optimized model runs slower than expected
- **Solution**: Profile hardware utilization, check for sparse operations support, verify memory access patterns

### Next Steps and Advanced Topics

After mastering these techniques, explore:
- **Neural Architecture Search (NAS)** for automated optimization
- **AutoML optimization** pipelines
- **Hardware-aware optimization** for specific accelerators
- **Dynamic optimization** during inference
- **Federated learning optimization** techniques

## Recommended Projects

**Beginner Project**: Optimize a pre-trained image classifier for mobile deployment
**Intermediate Project**: Implement and compare multiple pruning strategies on a custom model
**Advanced Project**: Build an end-to-end optimization pipeline with automated hyperparameter tuning
