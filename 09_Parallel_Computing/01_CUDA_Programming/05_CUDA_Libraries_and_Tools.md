# CUDA Libraries and Tools

*Duration: 2 weeks*

## Overview

CUDA provides a rich ecosystem of libraries and tools that accelerate development and optimize performance. This section covers essential CUDA libraries, the Thrust parallel algorithms library, and profiling tools.

## CUDA Math Libraries

### cuBLAS (Basic Linear Algebra Subroutines)

cuBLAS provides GPU-accelerated BLAS functionality:

```cpp
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Example: Matrix-vector multiplication using cuBLAS
void cublas_gemv_example() {
    const int m = 1024, n = 1024;
    float *h_A, *h_x, *h_y;
    float *d_A, *d_x, *d_y;
    
    // Allocate host memory
    h_A = (float*)malloc(m * n * sizeof(float));
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(m * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < m * n; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < n; i++) h_x[i] = rand() / (float)RAND_MAX;
    
    // Allocate device memory
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform matrix-vector multiplication: y = A * x
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
    
    // Copy result back to host
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y);
}

// Matrix multiplication example
void cublas_gemm_example() {
    const int m = 512, n = 512, k = 512;
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    // Initialize matrices (assume they're filled)
    // ...
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // C = A * B
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

### cuSPARSE (Sparse Matrix Operations)

```cpp
#include <cusparse.h>

// Sparse matrix-vector multiplication
void cusparse_spmv_example() {
    // Example with CSR format
    const int m = 1000, nnz = 5000;
    
    // Host data
    int *h_csrRowPtr = (int*)malloc((m + 1) * sizeof(int));
    int *h_csrColInd = (int*)malloc(nnz * sizeof(int));
    float *h_csrVal = (float*)malloc(nnz * sizeof(float));
    float *h_x = (float*)malloc(m * sizeof(float));
    float *h_y = (float*)malloc(m * sizeof(float));
    
    // Initialize sparse matrix and vector
    // ...
    
    // Device data
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y;
    
    cudaMalloc(&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, nnz * sizeof(int));
    cudaMalloc(&d_csrVal, nnz * sizeof(float));
    cudaMalloc(&d_x, m * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, m * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    // Perform SpMV: y = A * x
    const float alpha = 1.0f, beta = 0.0f;
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                   m, m, nnz, &alpha, descr, d_csrVal, d_csrRowPtr, d_csrColInd,
                   d_x, &beta, d_y);
    
    // Copy result back
    cudaMemcpy(h_y, d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    // Free memory...
}
```

### cuRAND (Random Number Generation)

```cpp
#include <curand.h>

// Generate random numbers on GPU
void curand_example() {
    const int n = 1000000;
    float *d_random;
    
    cudaMalloc(&d_random, n * sizeof(float));
    
    // Create generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    // Generate uniform random numbers [0,1)
    curandGenerateUniform(gen, d_random, n);
    
    // Generate normal distribution (mean=0, std=1)
    curandGenerateNormal(gen, d_random, n, 0.0f, 1.0f);
    
    curandDestroyGenerator(gen);
    cudaFree(d_random);
}

// Device API example
__global__ void curand_device_example(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(1234, idx, 0, &state);
        output[idx] = curand_uniform(&state);
    }
}
```

### cuFFT (Fast Fourier Transform)

```cpp
#include <cufft.h>

// 1D FFT example
void cufft_1d_example() {
    const int nx = 1024;
    cufftComplex *h_data, *d_data;
    
    h_data = (cufftComplex*)malloc(nx * sizeof(cufftComplex));
    cudaMalloc(&d_data, nx * sizeof(cufftComplex));
    
    // Initialize input data
    for (int i = 0; i < nx; i++) {
        h_data[i].x = cos(2 * M_PI * i / nx);  // Real part
        h_data[i].y = 0.0f;                    // Imaginary part
    }
    
    cudaMemcpy(d_data, h_data, nx * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, nx, CUFFT_C2C, 1);
    
    // Execute forward FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    // Execute inverse FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    
    // Copy result back
    cudaMemcpy(h_data, d_data, nx * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    // Normalize after inverse FFT
    for (int i = 0; i < nx; i++) {
        h_data[i].x /= nx;
        h_data[i].y /= nx;
    }
    
    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_data);
}

// 2D FFT example
void cufft_2d_example() {
    const int nx = 512, ny = 512;
    cufftComplex *d_data;
    
    cudaMalloc(&d_data, nx * ny * sizeof(cufftComplex));
    
    // Create 2D FFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, nx, ny, CUFFT_C2C);
    
    // Execute FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    
    cufftDestroy(plan);
    cudaFree(d_data);
}
```

## Thrust Library

Thrust provides high-level parallel algorithms:

```cpp
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

// Basic Thrust operations
void thrust_basic_example() {
    // Create host vector
    thrust::host_vector<int> h_vec(1000000);
    
    // Initialize with sequence
    thrust::sequence(h_vec.begin(), h_vec.end());
    
    // Copy to device
    thrust::device_vector<int> d_vec = h_vec;
    
    // Transform: square each element
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                     thrust::placeholders::_1 * thrust::placeholders::_1);
    
    // Reduce: sum all elements
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0);
    
    // Sort the vector
    thrust::sort(d_vec.begin(), d_vec.end());
    
    printf("Sum: %d\n", sum);
}

// Custom functors
struct saxpy_functor {
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    
    __host__ __device__
    float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

void thrust_custom_functor() {
    thrust::device_vector<float> x(1000000, 1.0f);
    thrust::device_vector<float> y(1000000, 2.0f);
    thrust::device_vector<float> z(1000000);
    
    // z = 3.0 * x + y
    thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                     saxpy_functor(3.0f));
}

// Reduction with custom operator
struct variance_data {
    float mean, M2;
    size_t n;
    
    __host__ __device__
    variance_data() : mean(0), M2(0), n(0) {}
    
    __host__ __device__
    variance_data(float x) : mean(x), M2(0), n(1) {}
};

struct variance_op {
    __host__ __device__
    variance_data operator()(const variance_data& x, const variance_data& y) const {
        variance_data result;
        result.n = x.n + y.n;
        
        if (result.n == 0) return result;
        
        float delta = y.mean - x.mean;
        result.mean = x.mean + delta * y.n / result.n;
        result.M2 = x.M2 + y.M2 + delta * delta * x.n * y.n / result.n;
        
        return result;
    }
};

void thrust_variance_example() {
    thrust::device_vector<float> data(1000000);
    thrust::sequence(data.begin(), data.end());
    
    // Transform to variance_data
    thrust::device_vector<variance_data> var_data(data.size());
    thrust::transform(data.begin(), data.end(), var_data.begin(),
                     [] __device__ (float x) { return variance_data(x); });
    
    // Reduce to compute variance
    variance_data result = thrust::reduce(var_data.begin(), var_data.end(),
                                        variance_data(), variance_op());
    
    float variance = result.M2 / (result.n - 1);
    printf("Variance: %f\n", variance);
}
```

## CUDA Profiling Tools

### Nsight Compute

```cpp
// Profiling annotations
#include <nvtx3/nvToolsExt.h>

__global__ void profiled_kernel(float* data, int n) {
    // Add profiling range
    nvtxRangePush("Kernel computation");
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate compute-intensive operation
        float result = data[idx];
        for (int i = 0; i < 100; i++) {
            result = sqrtf(result + 1.0f);
        }
        data[idx] = result;
    }
    
    nvtxRangePop();
}

void profiling_example() {
    const int n = 1000000;
    float *d_data;
    
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Mark regions for profiling
    nvtxRangePush("Memory initialization");
    cudaMemset(d_data, 0, n * sizeof(float));
    nvtxRangePop();
    
    nvtxRangePush("Kernel execution");
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    profiled_kernel<<<grid, block>>>(d_data, n);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    cudaFree(d_data);
}
```

### Performance Metrics Collection

```cpp
#include <cuda_profiler_api.h>

void performance_metrics_example() {
    // Enable profiler
    cudaProfilerStart();
    
    const int n = 1000000;
    float *d_a, *d_b, *d_c;
    
    // Allocate memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Initialize data
    cudaMemset(d_a, 1, n * sizeof(float));
    cudaMemset(d_b, 2, n * sizeof(float));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("Throughput: %f GB/s\n", 
           3 * n * sizeof(float) / (milliseconds * 1e6));
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    cudaProfilerStop();
}

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

## Occupancy Analysis

```cpp
// Occupancy calculator example
__global__ void occupancy_test_kernel(float* data, int n) {
    __shared__ float shared_mem[1024];  // Uses shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Use registers
    float reg1 = 0, reg2 = 0, reg3 = 0;
    
    if (idx < n) {
        shared_mem[tid] = data[idx];
        __syncthreads();
        
        // Compute something
        reg1 = shared_mem[tid] * 2.0f;
        reg2 = reg1 + shared_mem[(tid + 1) % blockDim.x];
        reg3 = reg2 * reg1;
        
        data[idx] = reg3;
    }
}

void occupancy_analysis() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Get occupancy for different block sizes
    for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
        int minGridSize, actualBlockSize;
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &actualBlockSize,
                                         occupancy_test_kernel, 1024, 0);
        
        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                    occupancy_test_kernel,
                                                    blockSize, 1024);
        
        float occupancy = (maxActiveBlocks * blockSize / (float)prop.maxThreadsPerMultiProcessor) * 100;
        
        printf("Block size: %d, Occupancy: %.2f%%\n", blockSize, occupancy);
    }
}
```

## Error Checking Utilities

```cpp
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// cuBLAS error checking
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Usage example
void error_checking_example() {
    float *d_data;
    
    CUDA_CHECK(cudaMalloc(&d_data, 1000 * sizeof(float)));
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Use the handle...
    
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_data));
}
```

## Exercises

1. **Library Integration**: Create a program that uses multiple CUDA libraries (cuBLAS, cuFFT, cuRAND) to implement a signal processing pipeline.

2. **Thrust Algorithms**: Implement parallel sorting, reduction, and transformation algorithms using Thrust, comparing performance with custom kernels.

3. **Profiling Analysis**: Use Nsight Compute to profile a complex kernel and identify optimization opportunities.

4. **Occupancy Optimization**: Analyze and optimize kernel occupancy for different problem sizes and architectures.

## Key Takeaways

- CUDA libraries provide optimized implementations for common operations
- Thrust enables high-level parallel programming with STL-like syntax
- Profiling tools are essential for performance optimization
- Understanding occupancy is crucial for maximizing GPU utilization
- Proper error handling improves development productivity

## Next Steps

Proceed to [CUDA and OpenCV](06_CUDA_and_OpenCV.md) to learn about integrating CUDA with computer vision applications.
