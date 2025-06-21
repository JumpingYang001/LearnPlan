# CUDA Programming Fundamentals

*Duration: 2 weeks*

## Overview

This section covers the essential building blocks of CUDA programming, from setting up the development environment to writing your first CUDA kernels. You'll learn the CUDA programming model, memory management, and implement basic CUDA programs.

## Learning Objectives

By the end of this section, you will be able to:
- Set up a complete CUDA development environment
- Understand the CUDA programming model and execution hierarchy
- Manage GPU memory effectively
- Write and execute basic CUDA kernels
- Handle errors and debug CUDA programs

## 1. CUDA Development Environment

### CUDA Toolkit Installation

#### Windows Installation
```powershell
# Download CUDA Toolkit from NVIDIA
# Install Visual Studio (2019 or later) first
# Run CUDA installer and follow setup wizard

# Verify installation
nvcc --version
nvidia-smi
```

#### Environment Setup
```cpp
// Check CUDA installation
#include <cuda_runtime.h>
#include <iostream>

void checkCudaInstallation() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    }
}
```

### NVIDIA Drivers and Compatibility

```cpp
// Check driver version compatibility
void checkDriverCompatibility() {
    int runtimeVersion, driverVersion;
    
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." 
              << (runtimeVersion % 100) / 10 << std::endl;
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." 
              << (driverVersion % 100) / 10 << std::endl;
    
    if (driverVersion < runtimeVersion) {
        std::cout << "Warning: Driver version is older than runtime version" << std::endl;
    }
}
```

### Development Tools

#### NVCC Compiler
```bash
# Basic compilation
nvcc hello_cuda.cu -o hello_cuda

# With debugging information
nvcc -g -G hello_cuda.cu -o hello_cuda_debug

# With optimization
nvcc -O3 hello_cuda.cu -o hello_cuda_optimized

# Specify compute capability
nvcc -arch=sm_75 hello_cuda.cu -o hello_cuda

# Generate PTX assembly
nvcc -ptx hello_cuda.cu
```

#### Nsight Development Environment
```cpp
// Code with profiling markers for Nsight
#include <nvtx3/nvToolsExt.h>

void profiledFunction() {
    nvtxRangePush("Custom Function");
    
    // Your CUDA code here
    
    nvtxRangePop();
}
```

### Compilation Workflow

```makefile
# Makefile example
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_75
CPP_FLAGS = -std=c++11

SOURCES = main.cu kernel.cu utils.cpp
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cpp=.o)

TARGET = cuda_program

$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

%.o: %.cpp
	g++ $(CPP_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean
```

## 2. CUDA Programming Model

### Host and Device Code

```cpp
#include <cuda_runtime.h>
#include <iostream>

// Device function (runs on GPU)
__device__ float device_multiply(float a, float b) {
    return a * b;
}

// Global function (kernel, called from host)
__global__ void vector_multiply(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = device_multiply(a[idx], b[idx]);
    }
}

// Host function (runs on CPU)
void host_vector_multiply() {
    const int N = 1024;
    const int size = N * sizeof(float);
    
    // Host memory allocation
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    
    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Device memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_multiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
}
```

### Kernel Functions

```cpp
// Simple kernel example
__global__ void simple_kernel() {
    printf("Hello from thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);
}

// Kernel with parameters
__global__ void parameterized_kernel(int* data, int multiplier, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= multiplier;
    }
}

// Kernel with shared memory
__global__ void shared_memory_kernel(int* input, int* output, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0;
    }
    
    __syncthreads();
    
    // Process data in shared memory
    if (tid > 0 && tid < blockDim.x - 1) {
        shared_data[tid] = (shared_data[tid-1] + shared_data[tid] + shared_data[tid+1]) / 3;
    }
    
    __syncthreads();
    
    // Write back to global memory
    if (idx < n) {
        output[idx] = shared_data[tid];
    }
}
```

### Thread Hierarchy

```cpp
// Demonstrate thread hierarchy
__global__ void thread_hierarchy_demo(int* output) {
    // Grid dimensions
    int gridDimX = gridDim.x;
    int gridDimY = gridDim.y;
    int gridDimZ = gridDim.z;
    
    // Block dimensions
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int blockDimZ = blockDim.z;
    
    // Block indices
    int blockIdxX = blockIdx.x;
    int blockIdxY = blockIdx.y;
    int blockIdxZ = blockIdx.z;
    
    // Thread indices within block
    int threadIdxX = threadIdx.x;
    int threadIdxY = threadIdx.y;
    int threadIdxZ = threadIdx.z;
    
    // Global thread index (1D case)
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Global thread index (2D case)
    int globalIdx2D = (blockIdx.y * gridDim.x + blockIdx.x) * 
                      (blockDim.x * blockDim.y) + 
                      (threadIdx.y * blockDim.x + threadIdx.x);
    
    if (globalIdx == 0) {
        printf("Grid: (%d, %d, %d)\n", gridDimX, gridDimY, gridDimZ);
        printf("Block: (%d, %d, %d)\n", blockDimX, blockDimY, blockDimZ);
    }
    
    output[globalIdx] = globalIdx;
}

// Launch with different configurations
void launch_configurations() {
    int* d_output;
    const int N = 1024;
    cudaMalloc(&d_output, N * sizeof(int));
    
    // 1D configuration
    dim3 blockSize1D(256);
    dim3 gridSize1D((N + blockSize1D.x - 1) / blockSize1D.x);
    thread_hierarchy_demo<<<gridSize1D, blockSize1D>>>(d_output);
    
    // 2D configuration
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((32 + blockSize2D.x - 1) / blockSize2D.x,
                    (32 + blockSize2D.y - 1) / blockSize2D.y);
    thread_hierarchy_demo<<<gridSize2D, blockSize2D>>>(d_output);
    
    cudaFree(d_output);
}
```

### Thread Indexing

```cpp
// Various indexing patterns
__global__ void indexing_patterns(float* data, int width, int height) {
    // 1D indexing
    int idx1D = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2D indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx2D = row * width + col;
    
    // 3D indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx3D = z * (width * height) + y * width + x;
    
    // Strided indexing (for grid-stride loops)
    int stride = blockDim.x * gridDim.x;
    for (int i = idx1D; i < width * height; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}

// Matrix operations with proper indexing
__global__ void matrix_add(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Transposed matrix access pattern
__global__ void matrix_transpose(float* input, float* output, 
                               int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;
        output[output_idx] = input[input_idx];
    }
}
```

### Execution Configuration

```cpp
// Optimal block size selection
int getOptimalBlockSize(void (*kernel)(float*, int), int n) {
    int minGridSize, blockSize;
    
    // CUDA occupancy API
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                      kernel, 0, 0);
    
    return blockSize;
}

// Dynamic configuration based on problem size
void configureExecution(int n, dim3& gridSize, dim3& blockSize) {
    // Start with a reasonable block size
    blockSize.x = 256;
    
    // Calculate grid size
    gridSize.x = (n + blockSize.x - 1) / blockSize.x;
    
    // Adjust for very small problems
    if (gridSize.x == 1 && n < blockSize.x) {
        blockSize.x = n;
    }
    
    // Ensure we don't exceed device limits
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (blockSize.x > prop.maxThreadsPerBlock) {
        blockSize.x = prop.maxThreadsPerBlock;
        gridSize.x = (n + blockSize.x - 1) / blockSize.x;
    }
}

// Multiple kernel configurations
void multipleConfigurations() {
    const int N = 100000;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Configuration 1: 1D layout
    dim3 block1D(256);
    dim3 grid1D((N + block1D.x - 1) / block1D.x);
    
    // Configuration 2: 2D layout for matrix operations
    int side = sqrt(N);
    dim3 block2D(16, 16);
    dim3 grid2D((side + block2D.x - 1) / block2D.x,
                (side + block2D.y - 1) / block2D.y);
    
    // Configuration 3: Using occupancy calculator
    int optimalBlockSize = getOptimalBlockSize(simple_kernel, N);
    dim3 blockOpt(optimalBlockSize);
    dim3 gridOpt((N + blockOpt.x - 1) / blockOpt.x);
    
    cudaFree(d_data);
}
```

## 3. Memory Management

### Basic Memory Operations

```cpp
// Complete memory management example
class CudaMemoryManager {
private:
    void* d_ptr;
    size_t size;
    
public:
    CudaMemoryManager(size_t sz) : size(sz) {
        cudaError_t error = cudaMalloc(&d_ptr, size);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA malloc failed: " + 
                                   std::string(cudaGetErrorString(error)));
        }
    }
    
    ~CudaMemoryManager() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }
    
    void* get() const { return d_ptr; }
    
    void copyFromHost(const void* host_ptr) {
        cudaError_t error = cudaMemcpy(d_ptr, host_ptr, size, 
                                     cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy H2D failed");
        }
    }
    
    void copyToHost(void* host_ptr) {
        cudaError_t error = cudaMemcpy(host_ptr, d_ptr, size, 
                                     cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy D2H failed");
        }
    }
};
```

### Unified Memory

```cpp
// Unified Memory example
void unifiedMemoryExample() {
    const int N = 1000000;
    float* data;
    
    // Allocate unified memory
    cudaMallocManaged(&data, N * sizeof(float));
    
    // Initialize on CPU
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }
    
    // Process on GPU
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    vector_square<<<gridSize, blockSize>>>(data, N);
    cudaDeviceSynchronize();
    
    // Access result on CPU
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
    
    cudaFree(data);
}

__global__ void vector_square(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}
```

### Pinned Memory

```cpp
// Pinned memory for faster transfers
void pinnedMemoryExample() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    // Allocate pinned host memory
    float* h_pinned;
    cudaMallocHost(&h_pinned, size);
    
    // Allocate pageable host memory for comparison
    float* h_pageable = (float*)malloc(size);
    
    float* d_data;
    cudaMalloc(&d_data, size);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_pinned[i] = h_pageable[i] = i;
    }
    
    // Time transfers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Pinned memory transfer
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pinnedTime;
    cudaEventElapsedTime(&pinnedTime, start, stop);
    
    // Pageable memory transfer
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pageableTime;
    cudaEventElapsedTime(&pageableTime, start, stop);
    
    std::cout << "Pinned memory transfer: " << pinnedTime << " ms" << std::endl;
    std::cout << "Pageable memory transfer: " << pageableTime << " ms" << std::endl;
    
    // Cleanup
    cudaFreeHost(h_pinned);
    free(h_pageable);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

### Zero-Copy Memory

```cpp
// Zero-copy memory example
void zeroCopyExample() {
    const int N = 1000;
    float* h_data;
    
    // Allocate zero-copy memory
    cudaHostAlloc(&h_data, N * sizeof(float), cudaHostAllocMapped);
    
    // Get device pointer
    float* d_data;
    cudaHostGetDevicePointer(&d_data, h_data, 0);
    
    // Initialize on host
    for (int i = 0; i < N; i++) {
        h_data[i] = i;
    }
    
    // Process on device using mapped memory
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_square<<<gridSize, blockSize>>>(d_data, N);
    cudaDeviceSynchronize();
    
    // Access result directly from host pointer
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    cudaFreeHost(h_data);
}
```

## 4. First CUDA Programs

### Vector Addition

```cpp
// Complete vector addition example
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void runVectorAddition() {
    const int N = 1000000;
    const size_t size = N * sizeof(float);
    
    // Host vectors
    std::vector<float> h_A(N), h_B(N), h_C(N);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy input vectors to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    auto start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU execution time: " << duration.count() << " microseconds" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

### Matrix Operations

```cpp
// Matrix multiplication example
__global__ void matrixMultiply(const float* A, const float* B, float* C,
                              int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Optimized matrix multiplication with shared memory
__global__ void matrixMultiplyShared(const float* A, const float* B, float* C,
                                    int rowsA, int colsA, int colsB) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int m = 0; m < (colsA + 15) / 16; m++) {
        // Load tiles into shared memory
        if (row < rowsA && m * 16 + threadIdx.x < colsA) {
            As[threadIdx.y][threadIdx.x] = A[row * colsA + m * 16 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < colsB && m * 16 + threadIdx.y < colsA) {
            Bs[threadIdx.y][threadIdx.x] = B[(m * 16 + threadIdx.y) * colsB + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < 16; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}
```

### Simple Image Processing

```cpp
// Grayscale conversion kernel
__global__ void rgbToGrayscale(const unsigned char* rgb, unsigned char* gray,
                              int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        int rgbIdx = idx * 3;
        
        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        
        // Luminance formula
        gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Gaussian blur kernel
__global__ void gaussianBlur(const unsigned char* input, unsigned char* output,
                            int width, int height) {
    const float kernel[5][5] = {
        {1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f},
        {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
        {7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f},
        {4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f},
        {1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f}
    };
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= 2 && col < width - 2 && row >= 2 && row < height - 2) {
        float sum = 0.0f;
        
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int idx = (row + dy) * width + (col + dx);
                sum += input[idx] * kernel[dy + 2][dx + 2];
            }
        }
        
        output[row * width + col] = (unsigned char)sum;
    }
}
```

### Error Handling

```cpp
// Comprehensive error handling
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

class CudaErrorChecker {
public:
    static void checkLastError(const char* msg) {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error (" << msg << "): " 
                      << cudaGetErrorString(error) << std::endl;
            exit(1);
        }
    }
    
    static void checkKernelExecution(const char* kernelName) {
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Kernel execution error (" << kernelName << "): " 
                      << cudaGetErrorString(error) << std::endl;
            exit(1);
        }
    }
};

// Safe kernel launch
template<typename... Args>
void safeLaunchKernel(void(*kernel)(Args...), dim3 grid, dim3 block, 
                     size_t sharedMem, cudaStream_t stream, Args... args) {
    kernel<<<grid, block, sharedMem, stream>>>(args...);
    CudaErrorChecker::checkKernelExecution("safeLaunchKernel");
}

// Example usage with error checking
void errorHandlingExample() {
    const int N = 1000;
    float *d_data;
    
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Launch kernel with error checking
    vectorSquare<<<grid, block>>>(d_data, N);
    CudaErrorChecker::checkKernelExecution("vectorSquare");
    
    CUDA_CHECK(cudaFree(d_data));
}
```

## Practical Exercises

1. **Development Environment Setup**
   - Install CUDA Toolkit and verify installation
   - Compile and run sample programs
   - Set up debugging with CUDA-GDB

2. **Basic Kernel Development**
   - Implement vector operations (add, multiply, dot product)
   - Create matrix transpose kernel
   - Write element-wise image processing kernels

3. **Memory Management Practice**
   - Compare performance of different memory types
   - Implement memory pool allocator
   - Benchmark memory transfer speeds

4. **Thread Indexing Mastery**
   - Create kernels for 1D, 2D, and 3D problems
   - Implement different access patterns
   - Handle boundary conditions correctly

## Key Takeaways

- CUDA programming follows a host-device model with distinct execution spaces
- Proper thread indexing is crucial for correctness and performance
- Memory management requires careful attention to transfer overhead
- Error handling is essential for robust CUDA applications
- Block and grid configuration affects performance significantly

## Next Steps

After mastering CUDA fundamentals, proceed to:
- CUDA Optimization Techniques
- Advanced memory management strategies
- Performance profiling and analysis
- Parallel algorithm patterns
