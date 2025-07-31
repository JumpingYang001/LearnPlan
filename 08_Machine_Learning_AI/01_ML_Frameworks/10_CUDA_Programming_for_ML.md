# CUDA Programming for Machine Learning

*Duration: 2-3 weeks*

Machine Learning workloads are inherently parallel and can benefit tremendously from GPU acceleration. CUDA (Compute Unified Device Architecture) enables developers to harness the power of NVIDIA GPUs for ML applications, providing significant speedups over CPU-only implementations.

## CUDA Basics for ML

### GPU Architecture for ML

#### Why GPUs Excel at Machine Learning

GPUs are designed with thousands of cores optimized for parallel computation, making them ideal for ML workloads:

```
CPU vs GPU Architecture Comparison:

CPU (Optimized for latency)          GPU (Optimized for throughput)
┌─────────────────────┐             ┌─────────────────────────────────┐
│ Core 1 │ Core 2     │             │ SM1  │ SM2  │ SM3  │ ... │ SM80 │
│ ┌────┐ │ ┌────┐     │             │┌───┐ │┌───┐ │┌───┐ │     │┌───┐│
│ │ALU │ │ │ALU │     │             ││32 ││ ││32 ││ ││32 ││ ... ││32 ││
│ └────┘ │ └────┘     │             ││cores│││cores│││cores││     ││cores││
│ Large  │ Large      │             │└───┘ │└───┘ │└───┘ │     │└───┘│
│ Cache  │ Cache      │             └─────────────────────────────────┘
└─────────────────────┘             Thousands of lightweight cores
```

#### NVIDIA GPU Architecture Overview

**Streaming Multiprocessors (SMs):**
- Each SM contains 32-128 cores (depending on architecture)
- Shared memory and registers
- Warp schedulers for thread execution

**Memory Hierarchy:**
```
┌─────────────────────────────────────────┐
│              Global Memory               │ ← Largest, slowest (GB)
│              (DRAM/HBM)                  │
├─────────────────────────────────────────┤
│              L2 Cache                    │ ← Shared across SMs
├─────────────────────────────────────────┤
│        L1 Cache/Shared Memory            │ ← Per SM (KB)
├─────────────────────────────────────────┤
│              Registers                   │ ← Per thread (fastest)
└─────────────────────────────────────────┘
```

#### GPU Compute Capabilities for ML

```cpp
#include <cuda_runtime.h>
#include <iostream>

void queryGPUProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Cores per MP: " << getCoresPerMP(prop.major, prop.minor) << std::endl;
        std::cout << "  Total Cores: " << prop.multiProcessorCount * getCoresPerMP(prop.major, prop.minor) << std::endl;
        std::cout << "  Memory Bandwidth: " << (prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8.0) / 1e6 << " GB/s" << std::endl;
    }
}

// Helper function to get cores per multiprocessor
int getCoresPerMP(int major, int minor) {
    // Simplified mapping - actual numbers vary by specific architecture
    if (major == 7) return 64;   // Volta/Turing
    if (major == 8) return 64;   // Ampere
    if (major == 9) return 128;  // Hopper
    return 32; // Default/older architectures
}
```

### CUDA Programming Model

#### Thread Hierarchy

CUDA organizes threads in a hierarchical structure optimized for parallel execution:

```
Grid (entire kernel launch)
├── Block 0
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   └── Thread (0,2)
├── Block 1
│   ├── Thread (1,0)
│   ├── Thread (1,1)
│   └── Thread (1,2)
└── Block N
    └── ...

Example: Matrix Addition
Grid: 256 x 256 elements
Blocks: 16 x 16 threads each
Total blocks needed: (256/16) x (256/16) = 16 x 16 = 256 blocks
```

#### Basic CUDA Kernel Structure

```cpp
// CUDA kernel for element-wise vector addition
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host code to launch kernel
void launchVectorAdd() {
    const int N = 1024 * 1024; // 1M elements
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}
```

### Memory Hierarchy and Optimization

#### Understanding Memory Types

```cpp
// Global Memory - largest but slowest
__device__ float global_array[1024];

// Shared Memory - fast, per-block storage
__global__ void sharedMemoryExample() {
    __shared__ float shared_data[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load data into shared memory
    shared_data[tid] = global_array[bid * blockDim.x + tid];
    
    // Synchronize threads in block
    __syncthreads();
    
    // Now all threads can access shared_data efficiently
    float sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        sum += shared_data[i];
    }
}

// Constant Memory - cached, read-only
__constant__ float filter_kernel[9]; // 3x3 convolution kernel

__global__ void convolution2D(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum += input[(y+i)*width + (x+j)] * filter_kernel[(i+1)*3 + (j+1)];
            }
        }
        output[y*width + x] = sum;
    }
}
```

#### Memory Access Patterns for ML

**Coalesced Memory Access Example:**
```cpp
// GOOD: Coalesced access - adjacent threads access adjacent memory
__global__ void coalescedAccess(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f; // Adjacent threads access adjacent elements
    }
}

// BAD: Strided access - poor memory throughput
__global__ void stridedAccess(float* data, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < N) {
        data[idx * stride] = data[idx * stride] * 2.0f; // Non-coalesced
    }
}
```

### Kernel Optimization Techniques

#### Occupancy Optimization

```cpp
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

// Function to find optimal block size
int findOptimalBlockSize(void (*kernel)(float*, int), int minGridSize) {
    int blockSize, gridSize;
    
    // Calculate occupancy
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, kernel, 0, 0);
    
    printf("Suggested block size: %d\n", blockSize);
    printf("Minimum grid size for max occupancy: %d\n", gridSize);
    
    return blockSize;
}

// Kernel with register usage optimization
__global__ void optimizedKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use local variables to reduce register pressure
    if (idx < N) {
        float temp = data[idx];
        temp = temp * temp + 1.0f;
        temp = sqrtf(temp);
        data[idx] = temp;
    }
}
```

#### Matrix Multiplication Optimization Example

```cpp
// Naive matrix multiplication
__global__ void matMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized with shared memory tiling
#define TILE_SIZE 16

__global__ void matMulTiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < N && tile * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && tile * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

## cuDNN Library

### Deep Learning Primitives

cuDNN (CUDA Deep Neural Network library) is NVIDIA's GPU-accelerated library of primitives for deep neural networks. It provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

#### Key Features of cuDNN
- **Optimized Implementations**: Hand-tuned kernels for maximum performance
- **Multiple Algorithms**: Different algorithms for different use cases
- **Tensor Descriptors**: Flexible tensor format support
- **Automatic Algorithm Selection**: Runtime selection of optimal algorithms

#### Setting Up cuDNN

```cpp
#include <cudnn.h>
#include <cuda_runtime.h>

class CuDNNManager {
private:
    cudnnHandle_t cudnn_handle;
    
public:
    CuDNNManager() {
        // Initialize cuDNN
        cudnnCreate(&cudnn_handle);
        
        // Set math mode for Tensor Cores (if available)
        cudnnSetMathType(cudnn_handle, CUDNN_TENSOR_OP_MATH);
    }
    
    ~CuDNNManager() {
        cudnnDestroy(cudnn_handle);
    }
    
    cudnnHandle_t getHandle() { return cudnn_handle; }
};

// Check cuDNN version
void checkCuDNNVersion() {
    size_t version = cudnnGetVersion();
    printf("cuDNN version: %zu\n", version);
    
    size_t cudart_version;
    cudnnGetProperty(MAJOR_VERSION, &cudart_version);
    printf("cuDNN major version: %zu\n", cudart_version);
}
```

### Convolution Algorithms

cuDNN provides multiple convolution algorithms optimized for different scenarios:

#### Algorithm Selection and Benchmarking

```cpp
#include <cudnn.h>
#include <vector>
#include <chrono>

class ConvolutionBenchmark {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    
public:
    ConvolutionBenchmark(int batch_size, int channels, int height, int width,
                        int num_filters, int filter_size, int stride, int padding) {
        cudnnCreate(&cudnn);
        
        // Create tensor descriptors
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnCreateActivationDescriptor(&activation_desc);
        
        // Set descriptors
        cudnnSetTensorNdDescriptor(input_desc, CUDNN_FLOAT, 4, 
                                  new int[]{batch_size, channels, height, width},
                                  new int[]{channels*height*width, height*width, width, 1});
        
        cudnnSetFilterNdDescriptor(filter_desc, CUDNN_FLOAT, CUDNN_TENSOR_NCHW, 4,
                                  new int[]{num_filters, channels, filter_size, filter_size});
        
        cudnnSetConvolutionNdDescriptor(conv_desc, 2,
                                       new int[]{padding, padding},
                                       new int[]{stride, stride},
                                       new int[]{1, 1},
                                       CUDNN_CROSS_CORRELATION, CUDNN_FLOAT);
        
        // Calculate output dimensions
        int out_n, out_c, out_h, out_w;
        cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_desc, filter_desc,
                                             4, new int[]{&out_n, &out_c, &out_h, &out_w});
        
        cudnnSetTensorNdDescriptor(output_desc, CUDNN_FLOAT, 4,
                                  new int[]{out_n, out_c, out_h, out_w},
                                  new int[]{out_c*out_h*out_w, out_h*out_w, out_w, 1});
        
        cudnnSetTensorNdDescriptor(bias_desc, CUDNN_FLOAT, 4,
                                  new int[]{1, out_c, 1, 1},
                                  new int[]{out_c, 1, 1, 1});
        
        // Set activation descriptor (ReLU)
        cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU,
                                    CUDNN_NOT_PROPAGATE_NAN, 0.0);
    }
    
    void benchmarkAlgorithms(float* input, float* filter, float* bias, float* output) {
        // Find available algorithms
        int requested_count = 10;
        int returned_count;
        cudnnConvolutionFwdAlgoPerf_t perf_results[10];
        
        cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                            requested_count, &returned_count, perf_results);
        
        printf("Available Convolution Algorithms:\n");
        for (int i = 0; i < returned_count; i++) {
            printf("Algorithm %d: %s, Time: %.3f ms, Memory: %zu bytes\n",
                   i, getAlgorithmName(perf_results[i].algo),
                   perf_results[i].time, perf_results[i].memory);
        }
        
        // Use the fastest algorithm
        cudnnConvolutionFwdAlgo_t best_algo = perf_results[0].algo;
        
        // Get workspace size
        size_t workspace_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                               best_algo, &workspace_size);
        
        void* workspace;
        cudaMalloc(&workspace, workspace_size);
        
        // Perform convolution with bias and activation (fused operation)
        float alpha = 1.0f, beta = 0.0f;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Convolution + Bias + Activation in one call
        cudnnConvolutionBiasActivationForward(cudnn,
                                             &alpha, input_desc, input,
                                             filter_desc, filter,
                                             conv_desc, best_algo, workspace, workspace_size,
                                             &beta, output_desc, output,
                                             bias_desc, bias,
                                             activation_desc, output_desc, output);
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("Fused Conv+Bias+ReLU execution time: %.3f ms\n", duration.count() / 1000.0f);
        
        cudaFree(workspace);
    }
    
private:
    const char* getAlgorithmName(cudnnConvolutionFwdAlgo_t algo) {
        switch (algo) {
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "IMPLICIT_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "IMPLICIT_PRECOMP_GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "GEMM";
            case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "DIRECT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "FFT";
            case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "FFT_TILING";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "WINOGRAD";
            case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "WINOGRAD_NONFUSED";
            default: return "UNKNOWN";
        }
    }
};
```

### Tensor Operations

#### Tensor Descriptor Management

```cpp
class TensorManager {
private:
    cudnnTensorDescriptor_t tensor_desc;
    float* data;
    size_t size_bytes;
    
public:
    TensorManager(int n, int c, int h, int w) {
        cudnnCreateTensorDescriptor(&tensor_desc);
        
        // Set tensor format (NCHW - batch, channels, height, width)
        cudnnSetTensorNdDescriptor(tensor_desc, CUDNN_FLOAT, 4,
                                  new int[]{n, c, h, w},
                                  new int[]{c*h*w, h*w, w, 1});
        
        size_bytes = n * c * h * w * sizeof(float);
        cudaMalloc(&data, size_bytes);
    }
    
    ~TensorManager() {
        cudaFree(data);
        cudnnDestroyTensorDescriptor(tensor_desc);
    }
    
    // Tensor operations
    void scale(cudnnHandle_t cudnn, float alpha) {
        cudnnScaleTensor(cudnn, tensor_desc, data, &alpha);
    }
    
    void add(cudnnHandle_t cudnn, TensorManager& other, float alpha = 1.0f, float beta = 1.0f) {
        cudnnAddTensor(cudnn, &alpha, other.tensor_desc, other.data,
                      &beta, tensor_desc, data);
    }
    
    void transform(cudnnHandle_t cudnn, TensorManager& src, float alpha = 1.0f, float beta = 0.0f) {
        cudnnTransformTensor(cudnn, &alpha, src.tensor_desc, src.data,
                           &beta, tensor_desc, data);
    }
    
    float* getData() { return data; }
    cudnnTensorDescriptor_t getDescriptor() { return tensor_desc; }
};
```

### Integration with Frameworks

#### PyTorch Integration Example

```python
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# Enable cuDNN benchmarking for optimal algorithm selection
cudnn.benchmark = True
cudnn.deterministic = False  # For reproducibility, set to True

class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(OptimizedCNN, self).__init__()
        
        # Use channels_last memory format for better performance
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Convert to channels_last for optimal cuDNN performance
        x = x.to(memory_format=torch.channels_last)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training loop with cuDNN optimizations
def train_with_cudnn_optimizations():
    model = OptimizedCNN().cuda()
    
    # Convert model to channels_last
    model = model.to(memory_format=torch.channels_last)
    
    # Use automatic mixed precision for Tensor Core utilization
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            target = target.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

# Check cuDNN information
def check_cudnn_info():
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
```

#### TensorFlow Integration

```python
import tensorflow as tf

# Configure GPU memory growth and cuDNN
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for Tensor Core utilization
            tf.config.optimizer.set_jit(True)  # Enable XLA
            
            # Check cuDNN
            print(f"Built with cuDNN: {tf.test.is_built_with_cuda()}")
            print(f"GPU devices: {len(gpus)}")
            
        except RuntimeError as e:
            print(e)

# Model with cuDNN optimized layers
def create_optimized_model():
    model = tf.keras.Sequential([
        # Use cuDNN optimized LSTM
        tf.keras.layers.LSTM(128, return_sequences=True, 
                           activation='tanh', recurrent_activation='sigmoid'),
        
        # Use cuDNN optimized Conv2D
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Use fused batch norm and activation
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

# Mixed precision training
def train_with_mixed_precision():
    # Set global mixed precision policy
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model = create_optimized_model()
    
    # Compile with loss scaling for mixed precision
    optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
```

## NCCL (NVIDIA Collective Communications Library)

### Multi-GPU Communication

NCCL is optimized for multi-GPU and multi-node communication, essential for distributed deep learning. It provides efficient implementations of collective operations like all-reduce, broadcast, reduce, and all-gather.

#### NCCL Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NCCL Communication                        │
│                                                             │
│  GPU 0 ←→ GPU 1 ←→ GPU 2 ←→ GPU 3  (NVLink/PCIe)          │
│    ↕       ↕       ↕       ↕                               │
│  GPU 4 ←→ GPU 5 ←→ GPU 6 ←→ GPU 7                          │
│                                                             │
│  Node 0 ←────── InfiniBand/Ethernet ──────→ Node 1         │
└─────────────────────────────────────────────────────────────┘
```

#### Basic NCCL Setup and Initialization

```cpp
#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>

class NCCLManager {
private:
    ncclComm_t nccl_comm;
    int rank, size;
    cudaStream_t stream;
    
public:
    NCCLManager() {
        // Initialize MPI for multi-node coordination
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Get unique ID for NCCL communicator
        ncclUniqueId nccl_id;
        if (rank == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        
        // Broadcast the unique ID to all processes
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Set CUDA device for this process
        cudaSetDevice(rank % getLocalGPUCount());
        
        // Initialize NCCL communicator
        ncclCommInitRank(&nccl_comm, size, nccl_id, rank);
        
        // Create CUDA stream for async operations
        cudaStreamCreate(&stream);
        
        printf("NCCL initialized: Rank %d/%d\n", rank, size);
    }
    
    ~NCCLManager() {
        ncclCommDestroy(nccl_comm);
        cudaStreamDestroy(stream);
        MPI_Finalize();
    }
    
    int getLocalGPUCount() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        return device_count;
    }
    
    ncclComm_t getComm() { return nccl_comm; }
    cudaStream_t getStream() { return stream; }
    int getRank() { return rank; }
    int getSize() { return size; }
};
```

### All-reduce Operations

All-reduce is the most common operation in distributed training, where gradients from all GPUs are summed and the result is distributed back to all GPUs.

#### Efficient All-reduce Implementation

```cpp
class DistributedTraining {
private:
    NCCLManager nccl_mgr;
    float* gradients;
    size_t gradient_size;
    
public:
    DistributedTraining(size_t grad_size) : gradient_size(grad_size) {
        cudaMalloc(&gradients, gradient_size * sizeof(float));
    }
    
    ~DistributedTraining() {
        cudaFree(gradients);
    }
    
    // Perform all-reduce on gradients
    void allReduceGradients() {
        // All-reduce: sum gradients across all GPUs
        ncclAllReduce(gradients, gradients, gradient_size, ncclFloat, ncclSum,
                     nccl_mgr.getComm(), nccl_mgr.getStream());
        
        // Wait for completion
        cudaStreamSynchronize(nccl_mgr.getStream());
        
        // Scale by number of processes to get average
        float scale = 1.0f / nccl_mgr.getSize();
        scaleArray<<<(gradient_size + 255) / 256, 256, 0, nccl_mgr.getStream()>>>(
            gradients, gradient_size, scale);
    }
    
    // Broadcast model parameters from rank 0 to all other ranks
    void broadcastParameters(float* parameters, size_t param_size) {
        ncclBroadcast(parameters, parameters, param_size, ncclFloat, 0,
                     nccl_mgr.getComm(), nccl_mgr.getStream());
        cudaStreamSynchronize(nccl_mgr.getStream());
    }
    
    // All-gather: collect data from all GPUs
    void allGatherData(float* local_data, float* gathered_data, size_t local_size) {
        ncclAllGather(local_data, gathered_data, local_size, ncclFloat,
                     nccl_mgr.getComm(), nccl_mgr.getStream());
        cudaStreamSynchronize(nccl_mgr.getStream());
    }
    
    // Reduce-scatter: distribute reduction results
    void reduceScatter(float* input_data, float* output_data, size_t output_size) {
        ncclReduceScatter(input_data, output_data, output_size, ncclFloat, ncclSum,
                         nccl_mgr.getComm(), nccl_mgr.getStream());
        cudaStreamSynchronize(nccl_mgr.getStream());
    }
};

// CUDA kernel for scaling arrays
__global__ void scaleArray(float* data, size_t size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}
```

### Topology Awareness

NCCL automatically detects and optimizes for different GPU topologies to maximize communication bandwidth.

#### Topology Detection and Optimization

```cpp
class TopologyOptimizer {
public:
    static void analyzeTopology() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        printf("=== GPU Topology Analysis ===\n");
        printf("Total GPUs: %d\n\n", device_count);
        
        // Check peer-to-peer access capabilities
        for (int i = 0; i < device_count; i++) {
            for (int j = 0; j < device_count; j++) {
                if (i != j) {
                    int can_access;
                    cudaDeviceCanAccessPeer(&can_access, i, j);
                    
                    if (can_access) {
                        printf("GPU %d can access GPU %d directly\n", i, j);
                        
                        // Enable peer access
                        cudaSetDevice(i);
                        cudaDeviceEnablePeerAccess(j, 0);
                    }
                }
            }
        }
        
        // Analyze bandwidth between GPUs
        benchmarkP2PBandwidth(device_count);
    }
    
private:
    static void benchmarkP2PBandwidth(int device_count) {
        const size_t data_size = 64 * 1024 * 1024; // 64 MB
        const int num_iterations = 10;
        
        printf("\n=== P2P Bandwidth Matrix (GB/s) ===\n");
        printf("     ");
        for (int j = 0; j < device_count; j++) {
            printf("GPU%d  ", j);
        }
        printf("\n");
        
        for (int i = 0; i < device_count; i++) {
            printf("GPU%d ", i);
            
            for (int j = 0; j < device_count; j++) {
                if (i == j) {
                    printf(" N/A  ");
                    continue;
                }
                
                float bandwidth = measureP2PBandwidth(i, j, data_size, num_iterations);
                printf("%5.1f ", bandwidth);
            }
            printf("\n");
        }
    }
    
    static float measureP2PBandwidth(int src_gpu, int dst_gpu, size_t size, int iterations) {
        float *d_src, *d_dst;
        
        // Allocate memory on source GPU
        cudaSetDevice(src_gpu);
        cudaMalloc(&d_src, size);
        
        // Allocate memory on destination GPU
        cudaSetDevice(dst_gpu);
        cudaMalloc(&d_dst, size);
        
        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm up
        for (int i = 0; i < 3; i++) {
            cudaMemcpyPeer(d_dst, dst_gpu, d_src, src_gpu, size);
        }
        
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpyPeer(d_dst, dst_gpu, d_src, src_gpu, size);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        float bandwidth_gb_s = (size * iterations * 1e-9) / (elapsed_ms * 1e-3);
        
        // Cleanup
        cudaSetDevice(src_gpu);
        cudaFree(d_src);
        cudaSetDevice(dst_gpu);
        cudaFree(d_dst);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return bandwidth_gb_s;
    }
};
```

### Distributed Training Integration

#### PyTorch Distributed Training with NCCL

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize NCCL backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Move model to GPU and wrap with DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        
        # Set up optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            
            # Backward pass - DDP automatically handles all-reduce
            loss.backward()
            
            # Optional: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0 and self.rank == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
    
    def all_reduce_metrics(self, metric_tensor):
        """All-reduce metrics across all processes"""
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        metric_tensor /= self.world_size
        return metric_tensor

def distributed_training_main(rank, world_size):
    """Main training function for each process"""
    setup_distributed(rank, world_size)
    
    # Create model and trainer
    model = create_model()  # Your model creation function
    trainer = DistributedTrainer(model, rank, world_size)
    
    # Create distributed sampler for data loading
    train_dataset = create_dataset()  # Your dataset creation function
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, 
        num_workers=4, pin_memory=True)
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        avg_loss = trainer.train_epoch(train_loader)
        
        if rank == 0:
            print(f'Epoch {epoch}, Average Loss: {avg_loss:.6f}')
    
    cleanup_distributed()

# Launch distributed training
def launch_distributed_training():
    world_size = torch.cuda.device_count()
    mp.spawn(distributed_training_main, args=(world_size,), nprocs=world_size, join=True)
```

#### Advanced NCCL Communication Patterns

```python
import torch
import torch.distributed as dist

class AdvancedNCCLOps:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
    
    def hierarchical_all_reduce(self, tensor):
        """
        Hierarchical all-reduce: first reduce within nodes, 
        then across nodes, then broadcast back
        """
        # Assume 8 GPUs per node
        gpus_per_node = 8
        node_id = self.rank // gpus_per_node
        local_rank = self.rank % gpus_per_node
        
        # Step 1: All-reduce within the node
        node_group = dist.new_group(list(range(node_id * gpus_per_node, 
                                              (node_id + 1) * gpus_per_node)))
        dist.all_reduce(tensor, group=node_group)
        
        # Step 2: All-reduce across nodes (only local rank 0 participates)
        if local_rank == 0:
            inter_node_ranks = list(range(0, self.world_size, gpus_per_node))
            inter_node_group = dist.new_group(inter_node_ranks)
            dist.all_reduce(tensor, group=inter_node_group)
        
        # Step 3: Broadcast within node
        dist.broadcast(tensor, src=node_id * gpus_per_node, group=node_group)
    
    def ring_all_reduce(self, tensor):
        """Custom ring all-reduce implementation"""
        chunk_size = tensor.numel() // self.world_size
        chunks = tensor.split(chunk_size)
        
        # Reduce-scatter phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank - step) % self.world_size
            recv_rank = (self.rank - step - 1) % self.world_size
            
            send_chunk = chunks[send_rank]
            recv_buffer = torch.zeros_like(send_chunk)
            
            # Send and receive
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1) % self.world_size
            
            req_send = dist.isend(send_chunk, next_rank)
            req_recv = dist.irecv(recv_buffer, prev_rank)
            
            req_send.wait()
            req_recv.wait()
            
            # Accumulate
            chunks[recv_rank] += recv_buffer
        
        # All-gather phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank - step + 1) % self.world_size
            recv_rank = (self.rank - step) % self.world_size
            
            send_chunk = chunks[send_rank]
            
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1) % self.world_size
            
            req_send = dist.isend(send_chunk, next_rank)
            req_recv = dist.irecv(chunks[recv_rank], prev_rank)
            
            req_send.wait()
            req_recv.wait()
        
        return torch.cat(chunks)
    
    def bandwidth_optimal_broadcast(self, tensor, src_rank):
        """Bandwidth-optimal binary tree broadcast"""
        if self.rank == src_rank:
            # Root sends to its children
            for step in range(int(math.log2(self.world_size))):
                for child in self._get_children(self.rank, step):
                    if child < self.world_size:
                        dist.send(tensor, child)
        else:
            # Receive from parent
            parent = self._get_parent(self.rank)
            if parent is not None:
                dist.recv(tensor, parent)
            
            # Forward to children
            for step in range(int(math.log2(self.world_size))):
                for child in self._get_children(self.rank, step):
                    if child < self.world_size:
                        dist.send(tensor, child)
    
    def _get_parent(self, rank):
        if rank == 0:
            return None
        return (rank - 1) // 2
    
    def _get_children(self, rank, step):
        base = 2 ** (step + 1)
        return [rank + base // 2, rank + base]
```

## Custom CUDA Kernels

### Kernel Development for ML

Writing custom CUDA kernels allows you to implement ML operations that are not available in standard libraries or to optimize specific operations for your use case.

#### ML-Specific Kernel Patterns

**1. Element-wise Operations**
```cpp
// Activation functions
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_activation(float x) {
    float exp_2x = expf(2.0f * x);
    return (exp_2x - 1.0f) / (exp_2x + 1.0f);
}

__global__ void activationKernel(float* input, float* output, int n, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        switch(activation_type) {
            case 0: output[idx] = relu(x); break;
            case 1: output[idx] = sigmoid(x); break;
            case 2: output[idx] = tanh_activation(x); break;
            default: output[idx] = x; break;
        }
    }
}
```

**2. Reduction Operations**
```cpp
// Optimized softmax kernel
__global__ void softmaxKernel(float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float shared_data[256];
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (batch_idx < batch_size) {
        float* batch_input = input + batch_idx * num_classes;
        float* batch_output = output + batch_idx * num_classes;
        
        // Find maximum (for numerical stability)
        float local_max = -INFINITY;
        for (int i = tid; i < num_classes; i += blockDim.x) {
            local_max = fmaxf(local_max, batch_input[i]);
        }
        
        shared_data[tid] = local_max;
        __syncthreads();
        
        // Reduce to find global max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            max_val = shared_data[0];
        }
        __syncthreads();
        
        // Compute sum of exponentials
        float local_sum = 0.0f;
        for (int i = tid; i < num_classes; i += blockDim.x) {
            local_sum += expf(batch_input[i] - max_val);
        }
        
        shared_data[tid] = local_sum;
        __syncthreads();
        
        // Reduce to find global sum
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            sum_exp = shared_data[0];
        }
        __syncthreads();
        
        // Compute final softmax values
        for (int i = tid; i < num_classes; i += blockDim.x) {
            batch_output[i] = expf(batch_input[i] - max_val) / sum_exp;
        }
    }
}
```

**3. Custom Convolution Implementation**
```cpp
// Simplified 2D convolution kernel
__global__ void conv2dKernel(
    float* input,    // [batch, in_channels, height, width]
    float* kernel,   // [out_channels, in_channels, kernel_h, kernel_w]
    float* output,   // [batch, out_channels, out_height, out_width]
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int output_h, int output_w, int stride, int padding
) {
    // Thread indices
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = threadIdx.y;
    
    if (batch_idx < batch_size && out_ch < out_channels && 
        out_x < output_w && out_y < output_h) {
        
        float sum = 0.0f;
        
        // Convolution computation
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_y = out_y * stride - padding + kh;
                    int in_x = out_x * stride - padding + kw;
                    
                    // Boundary check
                    if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                        int input_idx = ((batch_idx * in_channels + in_ch) * input_h + in_y) * input_w + in_x;
                        int kernel_idx = ((out_ch * in_channels + in_ch) * kernel_h + kh) * kernel_w + kw;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * out_channels + out_ch) * output_h + out_y) * output_w + out_x;
        output[output_idx] = sum;
    }
}
```

### Operation Fusion

Fusing multiple operations into a single kernel reduces memory bandwidth requirements and improves performance.

#### Fused Operations Examples

**1. Fused Convolution + Batch Norm + ReLU**
```cpp
__global__ void fusedConvBnReluKernel(
    float* input, float* weight, float* bias,
    float* bn_scale, float* bn_bias, float* bn_mean, float* bn_var,
    float* output, float epsilon,
    int batch_size, int channels, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        // Extract indices
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        int b = idx / (width * height * channels);
        
        // Convolution (simplified for demo)
        float conv_result = input[idx] * weight[c] + bias[c];
        
        // Batch Normalization
        float normalized = (conv_result - bn_mean[c]) / sqrtf(bn_var[c] + epsilon);
        float bn_result = normalized * bn_scale[c] + bn_bias[c];
        
        // ReLU activation
        output[idx] = fmaxf(0.0f, bn_result);
    }
}
```

**2. Fused GEMM + Bias + Activation**
```cpp
template<int BLOCK_SIZE, int ACTIVATION_TYPE>
__global__ void fusedGemmBiasActivation(
    float* A, float* B, float* C, float* bias,
    int M, int N, int K, float alpha, float beta
) {
    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Matrix multiplication with tiling
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load tiles
        if (row < M && tile * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && tile * BLOCK_SIZE + ty < K)
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        // Add bias
        float result = alpha * sum + beta * C[row * N + col] + bias[col];
        
        // Apply activation
        if (ACTIVATION_TYPE == 0) {
            result = fmaxf(0.0f, result);  // ReLU
        } else if (ACTIVATION_TYPE == 1) {
            result = 1.0f / (1.0f + expf(-result));  // Sigmoid
        }
        
        C[row * N + col] = result;
    }
}
```

### Memory Access Patterns

#### Optimized Memory Access Strategies

**1. Coalesced Global Memory Access**
```cpp
// GOOD: Coalesced access pattern
__global__ void coalescedTranspose(float* input, float* output, int width, int height) {
    __shared__ float tile[32][33]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Read coalesced from input
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    // Compute transposed coordinates
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    // Write coalesced to output
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

**2. Shared Memory Optimization**
```cpp
// Optimized reduction with shared memory
template<int BLOCK_SIZE>
__global__ void optimizedReduction(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    
    // Load and perform first level of reduction
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    if (idx + BLOCK_SIZE < n) {
        sdata[tid] += input[idx + BLOCK_SIZE];
    }
    
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll last warp
    if (tid < 32) {
        warpReduce<BLOCK_SIZE>(sdata, tid);
    }
    
    // Write result
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<int BLOCK_SIZE>
__device__ void warpReduce(volatile float* sdata, int tid) {
    if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
    if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
    if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
    if (BLOCK_SIZE >= 8)  sdata[tid] += sdata[tid + 4];
    if (BLOCK_SIZE >= 4)  sdata[tid] += sdata[tid + 2];
    if (BLOCK_SIZE >= 2)  sdata[tid] += sdata[tid + 1];
}
```

### Performance Profiling

#### CUDA Profiling Tools and Techniques

**1. Using NVIDIA Nsight Compute**
```cpp
#include <cuda_profiler_api.h>

class CUDAProfiler {
public:
    static void startProfiling() {
        cudaProfilerStart();
    }
    
    static void stopProfiling() {
        cudaProfilerStop();
    }
    
    static void profileKernel(const char* kernel_name, 
                             void (*kernel_func)(),
                             int iterations = 100) {
        // Warm up
        for (int i = 0; i < 10; i++) {
            kernel_func();
        }
        cudaDeviceSynchronize();
        
        // Profile
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            kernel_func();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        printf("Kernel %s: %.3f ms (avg over %d iterations)\n", 
               kernel_name, elapsed_ms / iterations, iterations);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};
```

**2. Memory Bandwidth Analysis**
```cpp
class BandwidthBenchmark {
public:
    static void measureGlobalMemoryBandwidth() {
        const size_t size = 64 * 1024 * 1024; // 64 MB
        const int iterations = 100;
        
        float *d_a, *d_b;
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        
        // Test different access patterns
        printf("=== Global Memory Bandwidth Test ===\n");
        
        // Sequential access
        float bw_sequential = benchmarkCopy(d_a, d_b, size, iterations, "sequential");
        
        // Strided access
        float bw_strided = benchmarkStridedCopy(d_a, d_b, size, iterations, 32);
        
        printf("Sequential bandwidth: %.2f GB/s\n", bw_sequential);
        printf("Strided bandwidth: %.2f GB/s\n", bw_strided);
        printf("Efficiency: %.1f%%\n", (bw_strided / bw_sequential) * 100);
        
        cudaFree(d_a);
        cudaFree(d_b);
    }
    
private:
    static float benchmarkCopy(float* src, float* dst, size_t size, int iterations, 
                              const char* test_name) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm up
        for (int i = 0; i < 5; i++) {
            cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        
        float bandwidth_gb_s = (2.0f * size * sizeof(float) * iterations * 1e-9) / 
                              (elapsed_ms * 1e-3);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return bandwidth_gb_s;
    }
    
    static float benchmarkStridedCopy(float* src, float* dst, size_t size, 
                                     int iterations, int stride) {
        // Implementation for strided memory access benchmark
        // ... (similar to above but with strided access pattern)
        return 0.0f; // Placeholder
    }
};
```

#### PyTorch Custom CUDA Extension

```python
# setup.py for PyTorch CUDA extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_ops',
            sources=[
                'custom_ops.cpp',
                'custom_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

```cpp
// custom_ops.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations
torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

torch::Tensor fused_linear_relu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA");
    
    return fused_linear_relu_cuda(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_relu", &fused_linear_relu, "Fused Linear + ReLU");
}
```

```cuda
// custom_kernels.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int input_size, int output_size
) {
    int batch_idx = blockIdx.x;
    int out_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx < batch_size && out_idx < output_size) {
        float sum = bias[out_idx];
        
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weight[out_idx * input_size + i];
        }
        
        // ReLU activation
        output[batch_idx * output_size + out_idx] = fmaxf(0.0f, sum);
    }
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int output_size = weight.size(0);
    
    auto output = torch::zeros({batch_size, output_size}, input.options());
    
    dim3 grid(batch_size, (output_size + 255) / 256);
    dim3 block(256);
    
    fused_linear_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, input_size, output_size
    );
    
    return output;
}
```


## Learning Objectives

By the end of this section, you should be able to:

- **Understand GPU architecture** and how it benefits ML workloads
- **Write efficient CUDA kernels** for common ML operations
- **Optimize memory access patterns** for maximum bandwidth utilization
- **Use cuDNN effectively** for deep learning primitives
- **Implement multi-GPU training** using NCCL
- **Profile and optimize** CUDA code for ML applications
- **Integrate custom CUDA kernels** with PyTorch and TensorFlow
- **Debug GPU memory issues** and optimize GPU utilization

### Self-Assessment Checklist

Before proceeding to advanced topics, ensure you can:

□ Write a basic CUDA kernel for element-wise operations  
□ Explain the difference between global, shared, and constant memory  
□ Implement efficient matrix multiplication using shared memory  
□ Use cuDNN for convolution operations  
□ Set up multi-GPU training with NCCL  
□ Profile CUDA kernels using NVIDIA tools  
□ Identify and fix memory access inefficiencies  
□ Create PyTorch CUDA extensions  

### Practical Projects

**Project 1: Custom Activation Functions**
```cpp
// TODO: Implement and benchmark custom activation functions
// - GELU, Swish, Mish activations
// - Compare performance with cuDNN implementations
// - Implement fused activation + batch norm
```

**Project 2: Optimized Attention Mechanism**
```cpp
// TODO: Implement scaled dot-product attention
// - Handle variable sequence lengths
// - Optimize for different head dimensions
// - Add causal masking support
```

**Project 3: Multi-GPU Data Pipeline**
```python
# TODO: Create efficient multi-GPU data loading
# - Implement data parallelism with NCCL
# - Handle gradient synchronization
# - Optimize communication overlap with computation
```

## Study Materials

### Essential Reading
- **Primary:** "Professional CUDA C Programming" by John Cheng, Max Grossman, Ty McKercher
- **CUDA Programming Guide:** [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **cuDNN Developer Guide:** [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- **NCCL Documentation:** [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

### Video Resources
- "CUDA Programming Masterclass" - Udemy
- "GPU Programming Fundamentals" - NVIDIA Deep Learning Institute
- "Optimizing CUDA Applications" - GTC Sessions
- "Multi-GPU Programming with NCCL" - NVIDIA Developer Channel

### Hands-on Labs
- **Lab 1:** Implement and optimize common ML kernels (ReLU, Softmax, CrossEntropy)
- **Lab 2:** Create a custom convolution implementation with different optimizations
- **Lab 3:** Build a multi-GPU training pipeline for image classification
- **Lab 4:** Profile and optimize a real ML workload

### Practice Questions

**Conceptual Questions:**
1. Why are GPUs more suitable for ML workloads compared to CPUs?
2. What is the difference between CUDA cores and Tensor cores?
3. How does memory coalescing affect performance in ML kernels?
4. When should you use shared memory vs global memory?
5. What are the trade-offs between different cuDNN convolution algorithms?

**Technical Questions:**
6. How do you calculate occupancy for a CUDA kernel?
7. What causes bank conflicts in shared memory and how to avoid them?
8. How does NCCL optimize all-reduce operations for different topologies?
9. What is the purpose of CUDA streams in ML applications?
10. How do you handle variable-length sequences efficiently on GPU?

**Performance Questions:**
11. How do you measure and optimize memory bandwidth utilization?
12. What profiling tools would you use to identify bottlenecks?
13. How do you overlap computation with communication in multi-GPU training?
14. What are the performance implications of different data layouts (NCHW vs NHWC)?
15. How do you optimize kernel launch configurations?

### Development Environment Setup

**Required Software:**
```bash
# CUDA Toolkit installation (Linux)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# cuDNN installation
tar -xzvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
sudo cp cudnn-*/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*/lib/libcudnn* /usr/local/cuda/lib64 

# NCCL installation
sudo apt install libnccl2 libnccl-dev

# Development tools
sudo apt install nvidia-cuda-toolkit
sudo apt install nsight-compute nsight-systems
```

**Environment Variables:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=/usr/local/cuda
export NCCL_ROOT=/usr/local/cuda
```

**Compilation Examples:**
```bash
# Basic CUDA compilation
nvcc -o program program.cu

# Optimized compilation
nvcc -O3 -use_fast_math -Xcompiler -O3 -o program program.cu

# With cuDNN and NCCL
nvcc -I$CUDNN_PATH/include -I$NCCL_ROOT/include \
     -L$CUDNN_PATH/lib64 -L$NCCL_ROOT/lib \
     -lcudnn -lnccl -o program program.cu

# Debug build
nvcc -g -G -O0 -o program_debug program.cu

# With specific architecture
nvcc -arch=sm_80 -o program program.cu  # For A100
nvcc -arch=sm_86 -o program program.cu  # For RTX 30 series
```

**Profiling Commands:**
```bash
# Nsight Compute profiling
ncu --set full -o profile_report ./program

# Nsight Systems profiling
nsys profile -o profile_timeline ./program

# Basic timing
nvprof ./program

# Memory profiling
cuda-memcheck ./program

# Check GPU utilization
nvidia-smi -l 1
```

### Performance Optimization Checklist

**Memory Optimization:**
- [ ] Use coalesced memory access patterns
- [ ] Minimize global memory transactions
- [ ] Utilize shared memory for data reuse
- [ ] Consider memory layout (AoS vs SoA)
- [ ] Use constant memory for read-only data

**Kernel Optimization:**
- [ ] Maximize occupancy
- [ ] Minimize register usage
- [ ] Avoid divergent branches
- [ ] Use appropriate block and grid sizes
- [ ] Consider loop unrolling

**Multi-GPU Optimization:**
- [ ] Overlap communication with computation
- [ ] Use optimal communication algorithms
- [ ] Balance workload across GPUs
- [ ] Minimize host-device transfers
- [ ] Utilize high-bandwidth interconnects

**Framework Integration:**
- [ ] Use framework-specific optimizations
- [ ] Enable mixed precision training
- [ ] Optimize data loading pipelines
- [ ] Use compiled models when possible
- [ ] Monitor GPU utilization and memory usage

## Advanced Topics Preview

### Next Steps After Mastering CUDA for ML:
1. **Tensor Core Programming** - Leveraging specialized AI hardware
2. **Multi-Instance GPU (MIG)** - Resource partitioning for inference
3. **CUDA Graphs** - Reducing kernel launch overhead
4. **Triton Programming** - High-level GPU kernel development
5. **Custom Autograd Functions** - Framework-specific optimizations

### Real-World Applications:
- Large Language Model (LLM) inference optimization
- Real-time computer vision pipelines
- Distributed training of foundation models
- Custom operators for novel architectures
- High-performance inference serving
