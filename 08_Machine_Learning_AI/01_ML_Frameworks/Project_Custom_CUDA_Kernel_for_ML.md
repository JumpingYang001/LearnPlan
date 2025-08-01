# Project: Custom CUDA Kernel for Machine Learning

*Duration: 2-3 weeks*  
*Difficulty: Advanced*  
*Prerequisites: CUDA Programming, PyTorch/TensorFlow basics, C++*

## Project Overview

This project will teach you how to implement high-performance custom operations for machine learning by writing CUDA kernels and integrating them with popular ML frameworks. You'll learn to optimize GPU computations, profile performance, and understand when custom kernels provide benefits over standard library implementations.

## Learning Objectives

By completing this project, you will:
- **Understand CUDA architecture** and GPU programming fundamentals
- **Implement custom CUDA kernels** for ML operations
- **Integrate CUDA kernels** with PyTorch and TensorFlow
- **Profile and benchmark** GPU performance
- **Optimize memory access patterns** for maximum throughput
- **Handle edge cases** and error checking in GPU code
- **Compare performance** against cuDNN and other optimized libraries

## Project Structure

```
custom_cuda_ml_project/
├── src/
│   ├── kernels/
│   │   ├── matrix_ops.cu          # Custom matrix operations
│   │   ├── activation_funcs.cu    # Custom activation functions
│   │   └── convolution.cu         # Custom convolution kernel
│   ├── python_bindings/
│   │   ├── pytorch_extension.cpp  # PyTorch C++ extension
│   │   └── tensorflow_op.cpp      # TensorFlow custom op
│   └── utils/
│       ├── cuda_utils.cuh         # CUDA utility functions
│       └── benchmark.py           # Performance benchmarking
├── tests/
│   ├── test_kernels.py           # Unit tests for kernels
│   └── test_integration.py       # Integration tests
├── benchmarks/
│   ├── performance_comparison.py
│   └── memory_profiling.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_optimization.py
└── setup.py                      # Build configuration
```

## Phase 1: CUDA Fundamentals and Basic Kernels

### Understanding CUDA Architecture

Before writing kernels, let's understand the GPU architecture:

```
GPU Architecture (Simplified)
┌─────────────────────────────────────────────────┐
│                    GPU                          │
│  ┌─────────────┐  ┌─────────────┐              │
│  │     SM 0    │  │     SM 1    │   ...        │
│  │ ┌─────────┐ │  │ ┌─────────┐ │              │
│  │ │ Core 0  │ │  │ │ Core 0  │ │              │
│  │ │ Core 1  │ │  │ │ Core 1  │ │              │
│  │ │   ...   │ │  │ │   ...   │ │              │
│  │ │ Core 127│ │  │ │ Core 127│ │              │
│  │ └─────────┘ │  │ └─────────┘ │              │
│  │ Shared Mem  │  │ Shared Mem  │              │
│  └─────────────┘  └─────────────┘              │
│                Global Memory                    │
└─────────────────────────────────────────────────┘
```

### Basic CUDA Kernel Implementation

**File: `src/kernels/matrix_ops.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Basic element-wise multiplication kernel
__global__ void elementwise_multiply_basic(const float* a, const float* b, 
                                         float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Optimized version with better memory coalescing
__global__ void elementwise_multiply_optimized(const float* __restrict__ a, 
                                             const float* __restrict__ b,
                                             float* __restrict__ c, 
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void elementwise_multiply_vectorized(const float4* a, 
                                               const float4* b,
                                               float4* c, 
                                               int n_vec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vec) {
        float4 a_vec = a[idx];
        float4 b_vec = b[idx];
        float4 c_vec;
        
        c_vec.x = a_vec.x * b_vec.x;
        c_vec.y = a_vec.y * b_vec.y;
        c_vec.z = a_vec.z * b_vec.z;
        c_vec.w = a_vec.w * b_vec.w;
        
        c[idx] = c_vec;
    }
}

// Matrix multiplication with shared memory optimization
__global__ void matrix_multiply_shared(const float* A, const float* B, 
                                     float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            tile_A[threadIdx.y][threadIdx.x] = 
                A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            tile_B[threadIdx.y][threadIdx.x] = 
                B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host wrapper functions
extern "C" {
    void launch_elementwise_multiply(const float* a, const float* b, float* c, 
                                   int n, cudaStream_t stream = 0) {
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        
        elementwise_multiply_optimized<<<grid_size, block_size, 0, stream>>>(
            a, b, c, n);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_matrix_multiply(const float* A, const float* B, float* C,
                              int M, int N, int K, cudaStream_t stream = 0) {
        const int TILE_SIZE = 16;
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, 
                      (M + TILE_SIZE - 1) / TILE_SIZE);
        
        matrix_multiply_shared<<<grid_size, block_size, 0, stream>>>(
            A, B, C, M, N, K);
        
        CUDA_CHECK(cudaGetLastError());
    }
}
```

### Custom Activation Functions

**File: `src/kernels/activation_funcs.cu`**

```cpp
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Swish activation function: f(x) = x * sigmoid(x)
__device__ float swish_activation(float x) {
    return x / (1.0f + expf(-x));
}

// Swish derivative: f'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
__device__ float swish_derivative(float x) {
    float swish_val = swish_activation(x);
    float sigmoid_val = 1.0f / (1.0f + expf(-x));
    return swish_val + sigmoid_val * (1.0f - swish_val);
}

// GELU activation function (approximation)
__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Mish activation function: f(x) = x * tanh(softplus(x))
__device__ float mish_activation(float x) {
    float softplus = logf(1.0f + expf(x));
    return x * tanhf(softplus);
}

// Forward pass kernel for custom activation functions
__global__ void custom_activation_forward(const float* input, float* output,
                                        int n, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        
        switch (activation_type) {
            case 0: // Swish
                output[idx] = swish_activation(x);
                break;
            case 1: // GELU
                output[idx] = gelu_activation(x);
                break;
            case 2: // Mish
                output[idx] = mish_activation(x);
                break;
            default:
                output[idx] = x; // Identity
        }
    }
}

// Backward pass kernel for custom activation functions
__global__ void custom_activation_backward(const float* grad_output,
                                         const float* input,
                                         float* grad_input,
                                         int n, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float grad_out = grad_output[idx];
        
        switch (activation_type) {
            case 0: // Swish
                grad_input[idx] = grad_out * swish_derivative(x);
                break;
            case 1: // GELU (numerical approximation of derivative)
                // Simplified derivative approximation
                grad_input[idx] = grad_out * 0.5f * (1.0f + tanhf(0.7978845608f * x));
                break;
            case 2: // Mish
                // Complex derivative - simplified for this example
                grad_input[idx] = grad_out * tanhf(logf(1.0f + expf(x)));
                break;
            default:
                grad_input[idx] = grad_out;
        }
    }
}

// Host wrapper functions
extern "C" {
    void launch_custom_activation_forward(const float* input, float* output,
                                        int n, int activation_type,
                                        cudaStream_t stream = 0) {
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        
        custom_activation_forward<<<grid_size, block_size, 0, stream>>>(
            input, output, n, activation_type);
    }
    
    void launch_custom_activation_backward(const float* grad_output,
                                         const float* input,
                                         float* grad_input,
                                         int n, int activation_type,
                                         cudaStream_t stream = 0) {
        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;
        
        custom_activation_backward<<<grid_size, block_size, 0, stream>>>(
            grad_output, input, grad_input, n, activation_type);
    }
}
```

## Phase 2: PyTorch Integration

### Creating PyTorch C++ Extensions

**File: `src/python_bindings/pytorch_extension.cpp`**

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations for CUDA kernels
void launch_elementwise_multiply(const float* a, const float* b, float* c, 
                               int n, cudaStream_t stream);
void launch_custom_activation_forward(const float* input, float* output,
                                    int n, int activation_type, 
                                    cudaStream_t stream);
void launch_custom_activation_backward(const float* grad_output,
                                     const float* input, float* grad_input,
                                     int n, int activation_type,
                                     cudaStream_t stream);

// PyTorch wrapper for element-wise multiplication
torch::Tensor elementwise_multiply_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "Input tensor a must be on CUDA device");
    TORCH_CHECK(b.is_cuda(), "Input tensor b must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 tensors supported");
    
    // Create output tensor
    auto output = torch::empty_like(a);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_elementwise_multiply(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        a.numel(),
        stream
    );
    
    return output;
}

// Custom activation function class for PyTorch autograd
class CustomActivationFunction : public torch::autograd::Function<CustomActivationFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input,
                               int activation_type) {
        // Save input for backward pass
        ctx->save_for_backward({input});
        ctx->saved_data["activation_type"] = activation_type;
        
        auto output = torch::empty_like(input);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        launch_custom_activation_forward(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel(),
            activation_type,
            stream
        );
        
        return output;
    }
    
    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        int activation_type = ctx->saved_data["activation_type"].toInt();
        
        auto grad_input = torch::empty_like(input);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        launch_custom_activation_backward(
            grad_outputs[0].data_ptr<float>(),
            input.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            input.numel(),
            activation_type,
            stream
        );
        
        return {grad_input, torch::Tensor(), torch::Tensor()};
    }
};

// Python-callable wrapper
torch::Tensor custom_activation(torch::Tensor input, int activation_type) {
    return CustomActivationFunction::apply(input, activation_type);
}

// Module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_multiply", &elementwise_multiply_cuda, 
          "Element-wise multiplication (CUDA)");
    m.def("custom_activation", &custom_activation, 
          "Custom activation function (CUDA)");
}
```

### PyTorch Python Interface

**File: `src/python_bindings/pytorch_ops.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import warnings

# Load the C++ extension
def load_cuda_extension():
    """Dynamically load the CUDA extension."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        cuda_ops = load(
            name="cuda_ops",
            sources=[
                os.path.join(current_dir, "pytorch_extension.cpp"),
                os.path.join(current_dir, "..", "kernels", "matrix_ops.cu"),
                os.path.join(current_dir, "..", "kernels", "activation_funcs.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True
        )
        return cuda_ops
    except Exception as e:
        warnings.warn(f"Failed to load CUDA extension: {e}")
        return None

# Global extension instance
_cuda_ops = None

def get_cuda_ops():
    """Get the CUDA operations module."""
    global _cuda_ops
    if _cuda_ops is None:
        _cuda_ops = load_cuda_extension()
    return _cuda_ops

class CustomElementwiseMultiply(nn.Module):
    """Custom element-wise multiplication layer."""
    
    def __init__(self):
        super().__init__()
        self.cuda_ops = get_cuda_ops()
    
    def forward(self, a, b):
        if self.cuda_ops and a.is_cuda and b.is_cuda:
            return self.cuda_ops.elementwise_multiply(a, b)
        else:
            # Fallback to PyTorch implementation
            return a * b

class SwishActivation(nn.Module):
    """Custom Swish activation function."""
    
    def __init__(self):
        super().__init__()
        self.cuda_ops = get_cuda_ops()
    
    def forward(self, x):
        if self.cuda_ops and x.is_cuda:
            return self.cuda_ops.custom_activation(x, 0)  # 0 = Swish
        else:
            # Fallback to PyTorch implementation
            return x * torch.sigmoid(x)

class GELUActivation(nn.Module):
    """Custom GELU activation function."""
    
    def __init__(self):
        super().__init__()
        self.cuda_ops = get_cuda_ops()
    
    def forward(self, x):
        if self.cuda_ops and x.is_cuda:
            return self.cuda_ops.custom_activation(x, 1)  # 1 = GELU
        else:
            # Fallback to PyTorch implementation
            return F.gelu(x)

class MishActivation(nn.Module):
    """Custom Mish activation function."""
    
    def __init__(self):
        super().__init__()
        self.cuda_ops = get_cuda_ops()
    
    def forward(self, x):
        if self.cuda_ops and x.is_cuda:
            return self.cuda_ops.custom_activation(x, 2)  # 2 = Mish
        else:
            # Fallback to PyTorch implementation
            return x * torch.tanh(F.softplus(x))

# Example neural network using custom operations
class CustomNet(nn.Module):
    """Example network using custom CUDA operations."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = SwishActivation()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.custom_activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# Utility functions for testing
def test_custom_operations():
    """Test custom CUDA operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test element-wise multiplication
    a = torch.randn(1000, 1000, device=device, requires_grad=True)
    b = torch.randn(1000, 1000, device=device, requires_grad=True)
    
    # Custom implementation
    multiply_layer = CustomElementwiseMultiply()
    result_custom = multiply_layer(a, b)
    
    # PyTorch implementation
    result_pytorch = a * b
    
    # Check correctness
    assert torch.allclose(result_custom, result_pytorch, atol=1e-6), \
        "Custom multiplication does not match PyTorch implementation"
    
    print("✅ Element-wise multiplication test passed")
    
    # Test custom activation
    x = torch.randn(1000, 1000, device=device, requires_grad=True)
    
    swish_layer = SwishActivation()
    result_custom = swish_layer(x)
    result_pytorch = x * torch.sigmoid(x)
    
    assert torch.allclose(result_custom, result_pytorch, atol=1e-5), \
        "Custom Swish does not match PyTorch implementation"
    
    print("✅ Custom Swish activation test passed")
    
    # Test gradient computation
    loss_custom = result_custom.sum()
    loss_pytorch = result_pytorch.sum()
    
    loss_custom.backward()
    grad_custom = x.grad.clone()
    
    x.grad.zero_()
    loss_pytorch.backward()
    grad_pytorch = x.grad.clone()
    
    assert torch.allclose(grad_custom, grad_pytorch, atol=1e-4), \
        "Custom gradient does not match PyTorch gradient"
    
    print("✅ Gradient computation test passed")

if __name__ == "__main__":
    test_custom_operations()
```

## Phase 3: Performance Benchmarking and Optimization

### Comprehensive Benchmarking Suite

**File: `benchmarks/performance_comparison.py`**

```python
import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import contextmanager
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.python_bindings.pytorch_ops import (
    CustomElementwiseMultiply, SwishActivation, GELUActivation, MishActivation
)

@contextmanager
def cuda_timer():
    """Context manager for precise CUDA timing."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    yield
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"CUDA Execution time: {elapsed_time:.3f} ms")

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, device="cuda", warmup_iterations=10, benchmark_iterations=100):
        self.device = torch.device(device)
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = {}
    
    def benchmark_elementwise_operations(self, sizes):
        """Benchmark element-wise operations."""
        print("🚀 Benchmarking Element-wise Operations...")
        
        results = {
            'size': [],
            'custom_time': [],
            'pytorch_time': [],
            'speedup': []
        }
        
        custom_multiply = CustomElementwiseMultiply()
        
        for size in sizes:
            print(f"  Testing size: {size}x{size}")
            
            # Generate test data
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = custom_multiply(a, b)
                _ = a * b
            
            torch.cuda.synchronize()
            
            # Benchmark custom implementation
            start_time = time.perf_counter()
            for _ in range(self.benchmark_iterations):
                result_custom = custom_multiply(a, b)
            torch.cuda.synchronize()
            custom_time = (time.perf_counter() - start_time) * 1000 / self.benchmark_iterations
            
            # Benchmark PyTorch implementation
            start_time = time.perf_counter()
            for _ in range(self.benchmark_iterations):
                result_pytorch = a * b
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start_time) * 1000 / self.benchmark_iterations
            
            speedup = pytorch_time / custom_time
            
            results['size'].append(size)
            results['custom_time'].append(custom_time)
            results['pytorch_time'].append(pytorch_time)
            results['speedup'].append(speedup)
            
            print(f"    Custom: {custom_time:.3f}ms, PyTorch: {pytorch_time:.3f}ms, "
                  f"Speedup: {speedup:.2f}x")
        
        self.results['elementwise'] = results
        return results
    
    def plot_results(self):
        """Plot benchmark results."""
        if 'elementwise' not in self.results:
            print("No results to plot. Run benchmarks first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CUDA Kernel Performance Benchmarks', fontsize=16)
        
        # Element-wise multiplication
        data = self.results['elementwise']
        axes[0, 0].plot(data['size'], data['custom_time'], 'b-o', label='Custom CUDA')
        axes[0, 0].plot(data['size'], data['pytorch_time'], 'r-s', label='PyTorch')
        axes[0, 0].set_xlabel('Matrix Size')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_title('Element-wise Multiplication')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig('cuda_performance_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main benchmarking function."""
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    print(f"🔍 GPU: {torch.cuda.get_device_name()}")
    print(f"🔍 CUDA Version: {torch.version.cuda}")
    print(f"🔍 PyTorch Version: {torch.__version__}")
    print()
    
    # Test sizes
    test_sizes = [512, 1024, 2048, 4096]
    
    benchmark = PerformanceBenchmark()
    benchmark.benchmark_elementwise_operations(test_sizes)
    benchmark.plot_results()

if __name__ == "__main__":
    main()
```

## Phase 4: Advanced Optimizations

### Memory Access Optimization

**File: `src/kernels/optimized_kernels.cu`**

```cpp
// Optimized convolution kernel with shared memory
__global__ void optimized_conv2d(const float* input, const float* weight,
                                float* output, int N, int C, int H, int W,
                                int K, int R, int S, int pad_h, int pad_w) {
    // Shared memory for input tiles
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + blockDim.x * blockDim.y;
    
    int batch = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_h = blockIdx.z / ((W + blockDim.x - 1) / blockDim.x);
    int out_w = blockIdx.z % ((W + blockDim.x - 1) / blockDim.x);
    
    int tid_h = threadIdx.y;
    int tid_w = threadIdx.x;
    
    // Calculate output position
    int out_row = out_h * blockDim.y + tid_h;
    int out_col = out_w * blockDim.x + tid_w;
    
    float result = 0.0f;
    
    // Iterate over input channels
    for (int c = 0; c < C; c++) {
        // Load weight into shared memory
        if (tid_h < R && tid_w < S) {
            shared_weight[tid_h * S + tid_w] = 
                weight[out_channel * C * R * S + c * R * S + tid_h * S + tid_w];
        }
        
        __syncthreads();
        
        // Convolution computation
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int in_row = out_row - pad_h + r;
                int in_col = out_col - pad_w + s;
                
                if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                    float input_val = input[batch * C * H * W + c * H * W + 
                                          in_row * W + in_col];
                    float weight_val = shared_weight[r * S + s];
                    result += input_val * weight_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (out_row < H && out_col < W) {
        output[batch * K * H * W + out_channel * H * W + out_row * W + out_col] = result;
    }
}
```

### Memory Profiling Tools

**File: `benchmarks/memory_profiling.py`**

```python
import torch
import psutil
import gc
from typing import Dict, Any

class MemoryProfiler:
    """GPU and CPU memory profiling utility."""
    
    def __init__(self):
        self.baseline_gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.baseline_cpu = psutil.virtual_memory().used
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**2    # MB
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        stats['cpu_used'] = psutil.virtual_memory().used / 1024**2  # MB
        stats['cpu_percent'] = psutil.virtual_memory().percent
        
        return stats
    
    def profile_operation(self, operation_func, *args, **kwargs):
        """Profile memory usage of an operation."""
        # Clear cache and collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Get initial memory
        initial_stats = self.get_memory_stats()
        
        # Run operation
        result = operation_func(*args, **kwargs)
        
        # Get final memory
        final_stats = self.get_memory_stats()
        
        # Calculate memory usage
        memory_usage = {}
        for key in initial_stats:
            memory_usage[f"delta_{key}"] = final_stats[key] - initial_stats[key]
        
        return result, memory_usage

# Example usage
def profile_custom_operations():
    """Profile memory usage of custom operations."""
    profiler = MemoryProfiler()
    
    # Test different tensor sizes
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        print(f"\n📊 Profiling size {size}x{size}:")
        
        # Create test tensors
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Profile custom operation
        from src.python_bindings.pytorch_ops import CustomElementwiseMultiply
        custom_op = CustomElementwiseMultiply()
        
        result, memory_stats = profiler.profile_operation(custom_op, a, b)
        
        print(f"  GPU Memory Delta: {memory_stats.get('delta_gpu_allocated', 0):.2f} MB")
        print(f"  CPU Memory Delta: {memory_stats.get('delta_cpu_used', 0):.2f} MB")
```

## Phase 5: Project Setup and Build System

### Setup Configuration

**File: `setup.py`**

```python
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from torch.utils import cpp_extension
import torch
import os

# CUDA extension
cuda_extension = cpp_extension.CUDAExtension(
    name='cuda_ml_ops',
    sources=[
        'src/python_bindings/pytorch_extension.cpp',
        'src/kernels/matrix_ops.cu',
        'src/kernels/activation_funcs.cu',
    ],
    extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_70,code=sm_70',  # V100
            '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
            '-gencode=arch=compute_80,code=sm_80',  # A100
            '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        ]
    },
    include_dirs=[
        'src/',
        torch.utils.cpp_extension.include_paths(),
    ]
)

setup(
    name='custom_cuda_ml',
    version='0.1.0',
    description='Custom CUDA kernels for machine learning',
    author='Your Name',
    author_email='your.email@example.com',
    packages=['src', 'src.python_bindings', 'benchmarks'],
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'pandas>=1.1.0',
        'psutil>=5.7.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.8',
        ]
    },
    zip_safe=False,
)
```

### Development Environment Setup

**File: `environment.yml`**

```yaml
name: cuda_ml_dev
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.9
  - pytorch>=1.12.0
  - torchvision
  - torchaudio
  - cudatoolkit=11.6
  - numpy
  - matplotlib
  - pandas
  - jupyter
  - pytest
  - pip
  - pip:
    - pybind11
    - psutil
    - black
    - flake8
    - mypy
```

### Build Instructions

**File: `BUILD.md`**

```markdown
# Build Instructions

## Prerequisites

1. **CUDA Toolkit**: Install CUDA 11.6+ from NVIDIA
2. **PyTorch**: Install PyTorch with CUDA support
3. **C++ Compiler**: GCC 7+ or MSVC 2017+

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd custom_cuda_ml_project

# Create conda environment
conda env create -f environment.yml
conda activate cuda_ml_dev

# Build and install
pip install -e .

# Run tests
python -m pytest tests/

# Run benchmarks
python benchmarks/performance_comparison.py
```

## Development Build

```bash
# Development installation with debugging symbols
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" pip install -e . --verbose

# Run specific tests
python src/python_bindings/pytorch_ops.py
```

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA is in PATH
2. **Compilation errors**: Check GCC/MSVC compatibility
3. **Runtime errors**: Verify GPU compute capability
```

## Project Milestones and Assessment

### Milestone 1: CUDA Fundamentals (Week 1)
**Deliverables:**
- [ ] Implement basic element-wise operations kernel
- [ ] Create memory-optimized matrix multiplication
- [ ] Write comprehensive error checking
- [ ] Document kernel performance characteristics

**Assessment Criteria:**
- Correctness of CUDA kernel implementation
- Understanding of memory coalescing patterns
- Proper error handling and edge case management
- Performance analysis and optimization rationale

### Milestone 2: Framework Integration (Week 2)
**Deliverables:**
- [ ] PyTorch C++ extension with autograd support
- [ ] Python wrapper classes with fallback implementations
- [ ] Unit tests for correctness and gradient checking
- [ ] Integration with existing PyTorch workflows

**Assessment Criteria:**
- Proper integration with PyTorch autograd system
- Comprehensive testing of forward and backward passes
- Clean API design and error handling
- Documentation of usage patterns

### Milestone 3: Performance Optimization (Week 3)
**Deliverables:**
- [ ] Comprehensive benchmarking suite
- [ ] Memory profiling and optimization
- [ ] Performance comparison with standard implementations
- [ ] Optimization recommendations and analysis

**Assessment Criteria:**
- Thorough performance analysis methodology
- Clear identification of optimization opportunities
- Quantitative comparison with baseline implementations
- Understanding of GPU architecture implications

### Final Project Assessment

**Technical Report Requirements:**
1. **Implementation Overview** (25%)
   - CUDA kernel design and optimization strategies
   - Framework integration approach
   - Code architecture and design decisions

2. **Performance Analysis** (35%)
   - Benchmarking methodology and results
   - Memory usage analysis
   - Comparison with existing implementations
   - Scalability analysis across different problem sizes

3. **Code Quality** (25%)
   - Clean, well-documented code
   - Comprehensive testing suite
   - Error handling and edge cases
   - Following CUDA and PyTorch best practices

4. **Innovation and Insights** (15%)
   - Novel optimization techniques
   - Insights into GPU programming
   - Future improvement suggestions
   - Understanding of trade-offs and limitations

### Learning Outcomes Verification

Upon completion, you should demonstrate:

✅ **CUDA Programming Proficiency**
- Write efficient CUDA kernels with proper memory management
- Optimize for GPU architecture (memory coalescing, occupancy)
- Handle synchronization and error checking

✅ **Framework Integration Skills**
- Create PyTorch extensions with autograd support
- Design clean APIs for ML operations
- Handle tensor operations and memory management

✅ **Performance Engineering**
- Profile and benchmark GPU operations
- Identify and resolve performance bottlenecks
- Compare and evaluate different optimization strategies

✅ **Software Engineering Practices**
- Write maintainable, tested code
- Create comprehensive documentation
- Follow best practices for GPU development

### Resources for Further Learning

**Advanced Topics to Explore:**
- Custom CUDA streams and multi-GPU programming
- Mixed precision training and Tensor Core utilization
- Integration with TensorFlow and other frameworks
- Distributed training with custom operations
- CUDA graphs for performance optimization

**Recommended Next Projects:**
- Implement custom attention mechanisms for transformers
- Create optimized computer vision operators
- Build distributed training primitives
- Develop specialized scientific computing kernels

This project provides a comprehensive foundation for understanding and implementing high-performance GPU computing for machine learning applications.
