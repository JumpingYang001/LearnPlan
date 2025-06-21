# CUDA and OpenCV

*Duration: 2 weeks*

## Overview

OpenCV (Open Source Computer Vision Library) provides excellent CUDA support for accelerating computer vision algorithms. This section covers CUDA-accelerated OpenCV modules, custom CUDA kernels for image processing, and integration patterns.

## CUDA-Accelerated OpenCV

### Setting up OpenCV with CUDA

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>

// Check CUDA device count
void check_cuda_devices() {
    int device_count = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices: " << device_count << std::endl;
    
    if (device_count > 0) {
        cv::cuda::DeviceInfo device_info(0);
        std::cout << "Device name: " << device_info.name() << std::endl;
        std::cout << "Compute capability: " << device_info.majorVersion() 
                  << "." << device_info.minorVersion() << std::endl;
    }
}
```

### GPU Mat Operations

```cpp
// Basic GPU Mat operations
void gpu_mat_example() {
    // Load image
    cv::Mat h_img = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (h_img.empty()) {
        std::cerr << "Could not load image!" << std::endl;
        return;
    }
    
    // Upload to GPU
    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);
    
    // Create output GPU matrices
    cv::cuda::GpuMat d_gray, d_blur, d_edges;
    
    // Convert to grayscale on GPU
    cv::cuda::cvtColor(d_img, d_gray, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur on GPU
    cv::cuda::GaussianBlur(d_gray, d_blur, cv::Size(15, 15), 2.0);
    
    // Detect edges on GPU
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = 
        cv::cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_blur, d_edges);
    
    // Download results
    cv::Mat h_gray, h_blur, h_edges;
    d_gray.download(h_gray);
    d_blur.download(h_blur);
    d_edges.download(h_edges);
    
    // Save results
    cv::imwrite("output_gray.jpg", h_gray);
    cv::imwrite("output_blur.jpg", h_blur);
    cv::imwrite("output_edges.jpg", h_edges);
}

// Performance comparison
void performance_comparison() {
    cv::Mat h_img = cv::imread("large_image.jpg", cv::IMREAD_COLOR);
    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);
    
    // CPU processing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::Mat h_result;
    cv::GaussianBlur(h_img, h_result, cv::Size(21, 21), 5.0);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    // GPU processing
    auto start_gpu = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat d_result;
    cv::cuda::GaussianBlur(d_img, d_result, cv::Size(21, 21), 5.0);
    cv::cuda::Stream stream;
    stream.waitForCompletion();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu);
    
    std::cout << "CPU time: " << cpu_time.count() << "ms" << std::endl;
    std::cout << "GPU time: " << gpu_time.count() << "ms" << std::endl;
    std::cout << "Speedup: " << (float)cpu_time.count() / gpu_time.count() << "x" << std::endl;
}
```

## Image Processing with CUDA

### Custom CUDA Kernels for Image Processing

```cpp
// Gaussian blur kernel
__global__ void gaussian_blur_kernel(cv::cuda::PtrStepSz<uchar3> src,
                                   cv::cuda::PtrStepSz<uchar3> dst,
                                   int kernel_size, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= src.cols || y >= src.rows) return;
    
    int half_kernel = kernel_size / 2;
    float sum_r = 0, sum_g = 0, sum_b = 0;
    float weight_sum = 0;
    
    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int nx = min(max(x + kx, 0), src.cols - 1);
            int ny = min(max(y + ky, 0), src.rows - 1);
            
            float weight = expf(-(kx*kx + ky*ky) / (2 * sigma * sigma));
            
            uchar3 pixel = src(ny, nx);
            sum_r += pixel.x * weight;
            sum_g += pixel.y * weight;
            sum_b += pixel.z * weight;
            weight_sum += weight;
        }
    }
    
    dst(y, x) = make_uchar3(sum_r / weight_sum, 
                           sum_g / weight_sum, 
                           sum_b / weight_sum);
}

// Host function to call the kernel
void apply_gaussian_blur(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, 
                        int kernel_size, float sigma) {
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    gaussian_blur_kernel<<<grid, block>>>(src, dst, kernel_size, sigma);
    cudaDeviceSynchronize();
}

// Sobel edge detection kernel
__global__ void sobel_edge_kernel(cv::cuda::PtrStepSz<uchar> src,
                                cv::cuda::PtrStepSz<uchar> dst) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= src.cols - 1 || y >= src.rows - 1 || x < 1 || y < 1) return;
    
    // Sobel X kernel
    int gx = -1 * src(y-1, x-1) + 1 * src(y-1, x+1) +
             -2 * src(y, x-1)   + 2 * src(y, x+1) +
             -1 * src(y+1, x-1) + 1 * src(y+1, x+1);
    
    // Sobel Y kernel
    int gy = -1 * src(y-1, x-1) + -2 * src(y-1, x) + -1 * src(y-1, x+1) +
              1 * src(y+1, x-1) +  2 * src(y+1, x) +  1 * src(y+1, x+1);
    
    // Magnitude
    int magnitude = sqrtf(gx * gx + gy * gy);
    dst(y, x) = min(magnitude, 255);
}

// Histogram calculation kernel
__global__ void histogram_kernel(cv::cuda::PtrStepSz<uchar> src, int* histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= src.cols || y >= src.rows) return;
    
    uchar pixel = src(y, x);
    atomicAdd(&histogram[pixel], 1);
}

void calculate_histogram(cv::cuda::GpuMat& src, std::vector<int>& hist) {
    int* d_histogram;
    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemset(d_histogram, 0, 256 * sizeof(int));
    
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    histogram_kernel<<<grid, block>>>(src, d_histogram);
    
    hist.resize(256);
    cudaMemcpy(hist.data(), d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_histogram);
}
```

## Computer Vision Algorithms

### Feature Detection and Matching

```cpp
// SURF feature detection with CUDA
void cuda_surf_example() {
    cv::Mat h_img1 = cv::imread("image1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat h_img2 = cv::imread("image2.jpg", cv::IMREAD_GRAYSCALE);
    
    cv::cuda::GpuMat d_img1, d_img2;
    d_img1.upload(h_img1);
    d_img2.upload(h_img2);
    
    // Create SURF detector
    cv::Ptr<cv::cuda::SURF_CUDA> surf = cv::cuda::SURF_CUDA::create(400);
    
    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::cuda::GpuMat descriptors1, descriptors2;
    
    surf->detectWithDescriptors(d_img1, cv::cuda::GpuMat(), keypoints1, descriptors1);
    surf->detectWithDescriptors(d_img2, cv::cuda::GpuMat(), keypoints2, descriptors2);
    
    // Match features
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = 
        cv::cuda::DescriptorMatcher::createBFMatcher();
    
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);
    
    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(h_img1, keypoints1, h_img2, keypoints2, matches, img_matches);
    cv::imwrite("matches.jpg", img_matches);
    
    std::cout << "Found " << matches.size() << " matches" << std::endl;
}

// Optical flow with CUDA
void cuda_optical_flow_example() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return;
    }
    
    cv::Mat frame, prev_frame;
    cv::cuda::GpuMat d_frame, d_prev_frame, d_gray, d_prev_gray;
    cv::cuda::GpuMat d_flow;
    
    // Create optical flow calculator
    cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1 = 
        cv::cuda::OpticalFlowDual_TVL1::create();
    
    cap >> prev_frame;
    d_prev_frame.upload(prev_frame);
    cv::cuda::cvtColor(d_prev_frame, d_prev_gray, cv::COLOR_BGR2GRAY);
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        d_frame.upload(frame);
        cv::cuda::cvtColor(d_frame, d_gray, cv::COLOR_BGR2GRAY);
        
        // Calculate optical flow
        tvl1->calc(d_prev_gray, d_gray, d_flow);
        
        // Visualize flow
        cv::cuda::GpuMat d_flow_vis;
        cv::cuda::split(d_flow, std::vector<cv::cuda::GpuMat>(2));
        // ... visualization code ...
        
        d_gray.copyTo(d_prev_gray);
        
        if (cv::waitKey(30) >= 0) break;
    }
}
```

### Object Detection

```cpp
// Template matching with CUDA
void cuda_template_matching() {
    cv::Mat h_img = cv::imread("scene.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat h_template = cv::imread("template.jpg", cv::IMREAD_GRAYSCALE);
    
    cv::cuda::GpuMat d_img, d_template, d_result;
    d_img.upload(h_img);
    d_template.upload(h_template);
    
    // Perform template matching
    cv::cuda::matchTemplate(d_img, d_template, d_result, cv::TM_CCOEFF_NORMED);
    
    // Find best match
    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::cuda::minMaxLoc(d_result, &min_val, &max_val, &min_loc, &max_loc);
    
    // Draw rectangle around best match
    cv::Mat h_result;
    d_img.download(h_result);
    cv::cvtColor(h_result, h_result, cv::COLOR_GRAY2BGR);
    
    cv::rectangle(h_result, max_loc, 
                 cv::Point(max_loc.x + h_template.cols, max_loc.y + h_template.rows),
                 cv::Scalar(0, 255, 0), 2);
    
    cv::imwrite("detection_result.jpg", h_result);
    std::cout << "Best match confidence: " << max_val << std::endl;
}

// HOG pedestrian detection
void cuda_hog_detection() {
    cv::Mat h_img = cv::imread("pedestrians.jpg");
    cv::cuda::GpuMat d_img;
    d_img.upload(h_img);
    
    // Create HOG descriptor
    cv::Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create();
    hog->setSVMDetector(cv::cuda::HOG::getDefaultPeopleDetector());
    
    // Detect pedestrians
    std::vector<cv::Rect> detections;
    hog->detectMultiScale(d_img, detections);
    
    // Draw detections
    for (const auto& rect : detections) {
        cv::rectangle(h_img, rect, cv::Scalar(0, 255, 0), 2);
    }
    
    cv::imwrite("hog_detections.jpg", h_img);
    std::cout << "Detected " << detections.size() << " pedestrians" << std::endl;
}
```

## Custom OpenCV CUDA Extensions

### Creating Custom CUDA Modules

```cpp
// Custom morphological operations
__global__ void dilate_kernel(cv::cuda::PtrStepSz<uchar> src,
                            cv::cuda::PtrStepSz<uchar> dst,
                            int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= src.cols || y >= src.rows) return;
    
    int half_kernel = kernel_size / 2;
    uchar max_val = 0;
    
    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int nx = min(max(x + kx, 0), src.cols - 1);
            int ny = min(max(y + ky, 0), src.rows - 1);
            
            max_val = max(max_val, src(ny, nx));
        }
    }
    
    dst(y, x) = max_val;
}

__global__ void erode_kernel(cv::cuda::PtrStepSz<uchar> src,
                           cv::cuda::PtrStepSz<uchar> dst,
                           int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= src.cols || y >= src.rows) return;
    
    int half_kernel = kernel_size / 2;
    uchar min_val = 255;
    
    for (int ky = -half_kernel; ky <= half_kernel; ky++) {
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int nx = min(max(x + kx, 0), src.cols - 1);
            int ny = min(max(y + ky, 0), src.rows - 1);
            
            min_val = min(min_val, src(ny, nx));
        }
    }
    
    dst(y, x) = min_val;
}

// Wrapper functions
void cuda_dilate(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size) {
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    dilate_kernel<<<grid, block>>>(src, dst, kernel_size);
    cudaDeviceSynchronize();
}

void cuda_erode(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size) {
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x,
              (src.rows + block.y - 1) / block.y);
    
    erode_kernel<<<grid, block>>>(src, dst, kernel_size);
    cudaDeviceSynchronize();
}
```

### Streaming and Asynchronous Processing

```cpp
// Video processing pipeline with streams
class VideoProcessor {
private:
    cv::cuda::Stream stream1, stream2;
    cv::cuda::GpuMat d_frame1, d_frame2;
    cv::cuda::GpuMat d_gray1, d_gray2;
    cv::cuda::GpuMat d_blur1, d_blur2;
    
public:
    void process_video_async(const std::string& input_path, 
                           const std::string& output_path) {
        cv::VideoCapture cap(input_path);
        cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M','J','P','G'), 
                              30, cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                         cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        
        cv::Mat frame1, frame2;
        bool use_stream1 = true;
        
        // Read first frame
        cap >> frame1;
        if (frame1.empty()) return;
        
        while (true) {
            if (use_stream1) {
                // Upload to stream 1
                d_frame1.upload(frame1, stream1);
                
                // Process on stream 1
                cv::cuda::cvtColor(d_frame1, d_gray1, cv::COLOR_BGR2GRAY, 0, stream1);
                cv::cuda::GaussianBlur(d_gray1, d_blur1, cv::Size(15, 15), 2.0, 0, stream1);
                cv::cuda::cvtColor(d_blur1, d_frame1, cv::COLOR_GRAY2BGR, 0, stream1);
                
                // Read next frame while processing
                cap >> frame2;
                if (frame2.empty()) break;
                
                // Download result from stream 1
                d_frame1.download(frame1, stream1);
                stream1.waitForCompletion();
                writer.write(frame1);
                
                frame1 = frame2.clone();
                use_stream1 = false;
            } else {
                // Similar processing for stream 2
                d_frame2.upload(frame1, stream2);
                cv::cuda::cvtColor(d_frame2, d_gray2, cv::COLOR_BGR2GRAY, 0, stream2);
                cv::cuda::GaussianBlur(d_gray2, d_blur2, cv::Size(15, 15), 2.0, 0, stream2);
                cv::cuda::cvtColor(d_blur2, d_frame2, cv::COLOR_GRAY2BGR, 0, stream2);
                
                cap >> frame2;
                if (frame2.empty()) break;
                
                d_frame2.download(frame1, stream2);
                stream2.waitForCompletion();
                writer.write(frame1);
                
                frame1 = frame2.clone();
                use_stream1 = true;
            }
        }
    }
};
```

## Memory Management Optimization

```cpp
// Memory pool for efficient allocation
class GpuMatPool {
private:
    std::vector<cv::cuda::GpuMat> pool;
    std::vector<bool> available;
    cv::Size mat_size;
    int mat_type;
    
public:
    GpuMatPool(cv::Size size, int type, int pool_size) 
        : mat_size(size), mat_type(type) {
        pool.resize(pool_size);
        available.resize(pool_size, true);
        
        for (int i = 0; i < pool_size; i++) {
            pool[i].create(size, type);
        }
    }
    
    cv::cuda::GpuMat* acquire() {
        for (int i = 0; i < pool.size(); i++) {
            if (available[i]) {
                available[i] = false;
                return &pool[i];
            }
        }
        return nullptr; // Pool exhausted
    }
    
    void release(cv::cuda::GpuMat* mat) {
        for (int i = 0; i < pool.size(); i++) {
            if (&pool[i] == mat) {
                available[i] = true;
                break;
            }
        }
    }
};

// Usage example
void memory_pool_example() {
    GpuMatPool pool(cv::Size(1920, 1080), CV_8UC3, 5);
    
    // Acquire matrices from pool
    cv::cuda::GpuMat* mat1 = pool.acquire();
    cv::cuda::GpuMat* mat2 = pool.acquire();
    
    // Use matrices...
    
    // Release back to pool
    pool.release(mat1);
    pool.release(mat2);
}
```

## Performance Benchmarking

```cpp
// Comprehensive benchmark suite
class OpenCVCudaBenchmark {
public:
    void run_all_benchmarks() {
        benchmark_basic_operations();
        benchmark_filtering();
        benchmark_feature_detection();
        benchmark_color_conversion();
    }
    
private:
    void benchmark_basic_operations() {
        cv::Mat h_img = cv::imread("test_image.jpg");
        cv::cuda::GpuMat d_img;
        d_img.upload(h_img);
        
        // Benchmark upload/download
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            d_img.upload(h_img);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto upload_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Upload time (100 iterations): " << upload_time.count() << " μs" << std::endl;
        
        // Benchmark memory allocation
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cv::cuda::GpuMat temp(h_img.rows, h_img.cols, h_img.type());
        }
        end = std::chrono::high_resolution_clock::now();
        
        auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Allocation time (100 iterations): " << alloc_time.count() << " μs" << std::endl;
    }
    
    void benchmark_filtering() {
        cv::Mat h_img = cv::imread("test_image.jpg", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat d_img, d_result;
        d_img.upload(h_img);
        
        // Benchmark Gaussian blur
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cv::cuda::GaussianBlur(d_img, d_result, cv::Size(15, 15), 2.0);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto blur_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Gaussian blur time (100 iterations): " << blur_time.count() << " μs" << std::endl;
        
        // Compare with CPU
        cv::Mat h_result;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cv::GaussianBlur(h_img, h_result, cv::Size(15, 15), 2.0);
        }
        end = std::chrono::high_resolution_clock::now();
        
        auto cpu_blur_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CPU Gaussian blur time (100 iterations): " << cpu_blur_time.count() << " μs" << std::endl;
        std::cout << "GPU speedup: " << (float)cpu_blur_time.count() / blur_time.count() << "x" << std::endl;
    }
};
```

## Exercises

1. **Image Processing Pipeline**: Create a complete image processing pipeline using CUDA-accelerated OpenCV functions.

2. **Custom Kernel Integration**: Implement custom CUDA kernels and integrate them with OpenCV GPU matrices.

3. **Real-time Video Processing**: Build a real-time video processing application using CUDA streams for optimal performance.

4. **Feature Matching System**: Implement a robust feature matching system using CUDA-accelerated SURF/ORB features.

5. **Performance Analysis**: Compare CPU vs GPU performance for various OpenCV operations and analyze the results.

## Key Takeaways

- OpenCV's CUDA module provides significant acceleration for computer vision tasks
- Custom CUDA kernels can be seamlessly integrated with OpenCV GPU matrices
- Streaming enables overlap of computation and memory transfers
- Memory management is crucial for optimal performance
- Not all operations benefit equally from GPU acceleration

## Next Steps

Proceed to [Deep Learning with CUDA](07_Deep_Learning_with_CUDA.md) to explore CUDA acceleration for deep learning applications.
