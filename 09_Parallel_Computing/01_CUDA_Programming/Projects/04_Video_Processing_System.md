# Project 4: Video Processing System

## Overview

Build a comprehensive real-time video processing system that demonstrates advanced CUDA programming techniques including streaming, memory management optimization, and multiple algorithm implementations.

## Project Goals

- Implement a real-time video processing pipeline
- Develop motion detection and tracking algorithms
- Optimize for low latency and high throughput
- Create a flexible framework for different processing algorithms

## System Architecture

```cpp
// Core video processing framework
class VideoProcessingSystem {
private:
    struct ProcessingPipeline {
        cv::cuda::Stream stream;
        cv::cuda::GpuMat input_frame;
        cv::cuda::GpuMat processed_frame;
        cv::cuda::GpuMat temp_buffers[4];
        bool is_busy;
        
        ProcessingPipeline() : is_busy(false) {
            // Pre-allocate buffers
            for (int i = 0; i < 4; i++) {
                temp_buffers[i].create(1080, 1920, CV_8UC3);
            }
        }
    };
    
    std::vector<ProcessingPipeline> pipelines;
    std::queue<cv::Mat> input_queue;
    std::queue<cv::Mat> output_queue;
    std::mutex queue_mutex;
    std::condition_variable cv_input, cv_output;
    
    bool running;
    std::thread processing_thread;
    
public:
    VideoProcessingSystem(int num_pipelines = 2);
    ~VideoProcessingSystem();
    
    void start();
    void stop();
    void process_frame(const cv::Mat& frame);
    bool get_processed_frame(cv::Mat& frame);
    
private:
    void processing_loop();
    void process_pipeline(ProcessingPipeline& pipeline, const cv::Mat& input);
};
```

## Core Implementation

### 1. Multi-Stream Processing Framework

```cpp
VideoProcessingSystem::VideoProcessingSystem(int num_pipelines) 
    : running(false) {
    pipelines.resize(num_pipelines);
    
    // Initialize CUDA context
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        throw std::runtime_error("No CUDA devices found");
    }
    
    cudaSetDevice(0);
    
    // Warm up GPU
    cv::cuda::GpuMat warm_up(100, 100, CV_8UC3);
    cv::cuda::GpuMat temp;
    cv::cuda::cvtColor(warm_up, temp, cv::COLOR_BGR2GRAY);
    cudaDeviceSynchronize();
}

void VideoProcessingSystem::start() {
    running = true;
    processing_thread = std::thread(&VideoProcessingSystem::processing_loop, this);
}

void VideoProcessingSystem::stop() {
    running = false;
    cv_input.notify_all();
    if (processing_thread.joinable()) {
        processing_thread.join();
    }
}

void VideoProcessingSystem::process_frame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    input_queue.push(frame.clone());
    cv_input.notify_one();
}

bool VideoProcessingSystem::get_processed_frame(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (!output_queue.empty()) {
        frame = output_queue.front();
        output_queue.pop();
        return true;
    }
    return false;
}

void VideoProcessingSystem::processing_loop() {
    while (running) {
        cv::Mat input_frame;
        
        // Get input frame
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv_input.wait(lock, [this] { return !input_queue.empty() || !running; });
            
            if (!running) break;
            
            input_frame = input_queue.front();
            input_queue.pop();
        }
        
        // Find available pipeline
        ProcessingPipeline* available_pipeline = nullptr;
        for (auto& pipeline : pipelines) {
            if (!pipeline.is_busy) {
                available_pipeline = &pipeline;
                pipeline.is_busy = true;
                break;
            }
        }
        
        if (available_pipeline) {
            process_pipeline(*available_pipeline, input_frame);
        }
    }
}
```

### 2. Motion Detection Algorithm

```cpp
class MotionDetector {
private:
    cv::cuda::GpuMat background_model;
    cv::cuda::GpuMat previous_frame;
    cv::cuda::GpuMat current_frame;
    cv::cuda::GpuMat motion_mask;
    cv::cuda::GpuMat temp_buffers[3];
    
    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> bg_subtractor;
    
    // Custom CUDA kernels
    void launch_motion_enhancement_kernel(cv::cuda::GpuMat& motion_mask);
    void launch_noise_reduction_kernel(cv::cuda::GpuMat& mask);
    
public:
    MotionDetector() {
        bg_subtractor = cv::cuda::createBackgroundSubtractorMOG2();
        bg_subtractor->setDetectShadows(true);
        bg_subtractor->setVarThreshold(25);
        bg_subtractor->setHistory(500);
    }
    
    void detect_motion(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, 
                      cv::cuda::Stream& stream);
    void get_motion_regions(std::vector<cv::Rect>& regions);
};

// Custom CUDA kernel for motion enhancement
__global__ void motion_enhancement_kernel(cv::cuda::PtrStepSz<uchar> motion_mask,
                                        cv::cuda::PtrStepSz<uchar> enhanced_mask,
                                        int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= motion_mask.cols || y >= motion_mask.rows) return;
    
    // Apply temporal filtering
    int motion_count = 0;
    int window_size = 3;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            int nx = min(max(x + dx, 0), motion_mask.cols - 1);
            int ny = min(max(y + dy, 0), motion_mask.rows - 1);
            
            if (motion_mask(ny, nx) > 0) {
                motion_count++;
            }
        }
    }
    
    // Enhanced motion detection
    if (motion_count >= threshold) {
        enhanced_mask(y, x) = 255;
    } else {
        enhanced_mask(y, x) = 0;
    }
}

void MotionDetector::detect_motion(cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                 cv::cuda::Stream& stream) {
    // Convert to grayscale
    cv::cuda::cvtColor(input, current_frame, cv::COLOR_BGR2GRAY, 0, stream);
    
    // Apply background subtraction
    bg_subtractor->apply(current_frame, motion_mask, -1, stream);
    
    // Custom motion enhancement
    launch_motion_enhancement_kernel(motion_mask);
    
    // Morphological operations to clean up noise
    cv::Ptr<cv::cuda::Filter> morph_close = 
        cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, 
                                       cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    morph_close->apply(motion_mask, temp_buffers[0], stream);
    
    cv::Ptr<cv::cuda::Filter> morph_open = 
        cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, 
                                       cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    morph_open->apply(temp_buffers[0], motion_mask, stream);
    
    // Overlay motion mask on original image
    cv::cuda::cvtColor(motion_mask, temp_buffers[1], cv::COLOR_GRAY2BGR, 0, stream);
    cv::cuda::addWeighted(input, 0.7, temp_buffers[1], 0.3, 0, output, -1, stream);
}

void MotionDetector::launch_motion_enhancement_kernel(cv::cuda::GpuMat& motion_mask) {
    dim3 block(16, 16);
    dim3 grid((motion_mask.cols + block.x - 1) / block.x,
              (motion_mask.rows + block.y - 1) / block.y);
    
    motion_enhancement_kernel<<<grid, block>>>(motion_mask, temp_buffers[2], 5);
    cudaDeviceSynchronize();
    
    temp_buffers[2].copyTo(motion_mask);
}
```

### 3. Object Tracking System

```cpp
class MultiObjectTracker {
private:
    struct TrackedObject {
        int id;
        cv::Rect bounding_box;
        cv::Point2f velocity;
        std::vector<cv::Point2f> trajectory;
        int frames_since_update;
        bool is_active;
        
        // CUDA-based template matching
        cv::cuda::GpuMat object_template;
        float confidence;
    };
    
    std::vector<TrackedObject> tracked_objects;
    int next_object_id;
    
    // CUDA template matcher
    cv::Ptr<cv::cuda::TemplateMatching> template_matcher;
    
    // Kalman filter for prediction
    std::vector<cv::KalmanFilter> kalman_filters;
    
public:
    MultiObjectTracker();
    void update(cv::cuda::GpuMat& frame, const std::vector<cv::Rect>& detections,
               cv::cuda::Stream& stream);
    void draw_tracks(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream);
    
private:
    void predict_positions();
    void update_tracks(cv::cuda::GpuMat& frame, const std::vector<cv::Rect>& detections);
    void create_new_tracks(cv::cuda::GpuMat& frame, const std::vector<cv::Rect>& unmatched_detections);
    float calculate_iou(const cv::Rect& rect1, const cv::Rect& rect2);
};

// CUDA kernel for trajectory visualization
__global__ void draw_trajectory_kernel(cv::cuda::PtrStepSz<uchar3> image,
                                     cv::Point2f* trajectory_points,
                                     int num_points,
                                     uchar3 color) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points - 1) return;
    
    cv::Point2f p1 = trajectory_points[idx];
    cv::Point2f p2 = trajectory_points[idx + 1];
    
    // Simple line drawing (Bresenham's algorithm on GPU)
    int x0 = (int)p1.x, y0 = (int)p1.y;
    int x1 = (int)p2.x, y1 = (int)p2.y;
    
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        if (x0 >= 0 && x0 < image.cols && y0 >= 0 && y0 < image.rows) {
            image(y0, x0) = color;
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void MultiObjectTracker::update(cv::cuda::GpuMat& frame, 
                              const std::vector<cv::Rect>& detections,
                              cv::cuda::Stream& stream) {
    predict_positions();
    update_tracks(frame, detections);
    
    // Remove inactive tracks
    tracked_objects.erase(
        std::remove_if(tracked_objects.begin(), tracked_objects.end(),
                      [](const TrackedObject& obj) { 
                          return obj.frames_since_update > 10; 
                      }),
        tracked_objects.end()
    );
}

void MultiObjectTracker::draw_tracks(cv::cuda::GpuMat& frame, cv::cuda::Stream& stream) {
    for (const auto& obj : tracked_objects) {
        if (!obj.is_active || obj.trajectory.size() < 2) continue;
        
        // Copy trajectory to device memory
        cv::cuda::GpuMat d_trajectory;
        cv::Mat h_trajectory(obj.trajectory.size(), 1, CV_32FC2, (void*)obj.trajectory.data());
        d_trajectory.upload(h_trajectory);
        
        // Launch trajectory drawing kernel
        dim3 block(256);
        dim3 grid((obj.trajectory.size() + block.x - 1) / block.x);
        
        uchar3 color = make_uchar3(0, 255, 0); // Green trajectory
        draw_trajectory_kernel<<<grid, block, 0, cv::cuda::StreamAccessor::getStream(stream)>>>(
            frame, (cv::Point2f*)d_trajectory.ptr(), obj.trajectory.size(), color);
        
        // Draw bounding box
        cv::cuda::rectangle(frame, obj.bounding_box, cv::Scalar(0, 255, 0), 2, 8, 0, stream);
        
        // Draw object ID
        std::string id_text = "ID: " + std::to_string(obj.id);
        cv::Point text_pos(obj.bounding_box.x, obj.bounding_box.y - 10);
        // Note: Text rendering on GPU requires custom implementation or CPU fallback
    }
}
```

### 4. Performance Optimization

```cpp
class PerformanceOptimizer {
private:
    struct PerformanceMetrics {
        float fps;
        float gpu_utilization;
        float memory_usage;
        float latency_ms;
        
        std::vector<float> frame_times;
        std::chrono::high_resolution_clock::time_point last_update;
    };
    
    PerformanceMetrics metrics;
    
    // Memory pool for efficient allocation
    class GpuMemoryPool {
    private:
        std::vector<cv::cuda::GpuMat> pool;
        std::vector<bool> available;
        cv::Size mat_size;
        int mat_type;
        
    public:
        GpuMemoryPool(cv::Size size, int type, int pool_size);
        cv::cuda::GpuMat* acquire();
        void release(cv::cuda::GpuMat* mat);
    };
    
    std::unique_ptr<GpuMemoryPool> memory_pool;
    
public:
    PerformanceOptimizer(cv::Size frame_size);
    
    void update_metrics();
    void print_performance_report();
    void optimize_memory_usage();
    void adjust_processing_parameters();
    
    cv::cuda::GpuMat* get_temp_buffer() { return memory_pool->acquire(); }
    void release_temp_buffer(cv::cuda::GpuMat* buffer) { memory_pool->release(buffer); }
};

// CUDA profiling integration
class CudaProfiler {
private:
    std::map<std::string, float> kernel_times;
    std::map<std::string, int> kernel_calls;
    
public:
    void start_profiling(const std::string& kernel_name) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        // Store events for later retrieval
        profile_events[kernel_name] = {start, stop};
    }
    
    void end_profiling(const std::string& kernel_name) {
        auto& events = profile_events[kernel_name];
        cudaEventRecord(events.second);
        cudaEventSynchronize(events.second);
        
        float ms;
        cudaEventElapsedTime(&ms, events.first, events.second);
        
        kernel_times[kernel_name] += ms;
        kernel_calls[kernel_name]++;
        
        cudaEventDestroy(events.first);
        cudaEventDestroy(events.second);
        profile_events.erase(kernel_name);
    }
    
    void print_profiling_report() {
        std::cout << "\n=== CUDA Profiling Report ===" << std::endl;
        for (const auto& entry : kernel_times) {
            const std::string& name = entry.first;
            float total_time = entry.second;
            int calls = kernel_calls[name];
            
            std::cout << name << ":" << std::endl;
            std::cout << "  Total time: " << total_time << " ms" << std::endl;
            std::cout << "  Calls: " << calls << std::endl;
            std::cout << "  Average: " << total_time / calls << " ms" << std::endl;
        }
    }
    
private:
    std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> profile_events;
};
```

### 5. Main Application

```cpp
class VideoProcessingApp {
private:
    std::unique_ptr<VideoProcessingSystem> processing_system;
    std::unique_ptr<MotionDetector> motion_detector;
    std::unique_ptr<MultiObjectTracker> object_tracker;
    std::unique_ptr<PerformanceOptimizer> performance_optimizer;
    std::unique_ptr<CudaProfiler> profiler;
    
    cv::VideoCapture input_capture;
    cv::VideoWriter output_writer;
    
    // GUI for parameter adjustment
    bool show_motion_mask;
    bool show_trajectories;
    float motion_threshold;
    int tracking_sensitivity;
    
public:
    VideoProcessingApp() :
        show_motion_mask(true),
        show_trajectories(true),
        motion_threshold(25.0f),
        tracking_sensitivity(5) {}
    
    bool initialize(const std::string& input_source, const std::string& output_path);
    void run();
    void cleanup();
    
private:
    void setup_gui();
    void process_frame(const cv::Mat& input_frame);
    void handle_gui_events();
};

bool VideoProcessingApp::initialize(const std::string& input_source, 
                                  const std::string& output_path) {
    // Initialize video capture
    if (input_source == "camera") {
        input_capture.open(0);
    } else {
        input_capture.open(input_source);
    }
    
    if (!input_capture.isOpened()) {
        std::cerr << "Failed to open input source: " << input_source << std::endl;
        return false;
    }
    
    // Get video properties
    int frame_width = input_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = input_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = input_capture.get(cv::CAP_PROP_FPS);
    
    // Initialize output writer
    output_writer.open(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                      fps, cv::Size(frame_width, frame_height));
    
    // Initialize processing components
    processing_system = std::make_unique<VideoProcessingSystem>(2);
    motion_detector = std::make_unique<MotionDetector>();
    object_tracker = std::make_unique<MultiObjectTracker>();
    performance_optimizer = std::make_unique<PerformanceOptimizer>(
        cv::Size(frame_width, frame_height));
    profiler = std::make_unique<CudaProfiler>();
    
    processing_system->start();
    
    return true;
}

void VideoProcessingApp::run() {
    cv::Mat frame;
    auto last_fps_update = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    while (true) {
        if (!input_capture.read(frame)) {
            break; // End of video or camera disconnected
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // Process frame
        processing_system->process_frame(frame);
        
        // Get processed result
        cv::Mat processed_frame;
        if (processing_system->get_processed_frame(processed_frame)) {
            // Write to output
            output_writer.write(processed_frame);
            
            // Display result
            cv::imshow("Video Processing System", processed_frame);
        }
        
        // Handle GUI events
        handle_gui_events();
        
        // Calculate FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_update);
        
        if (duration.count() >= 1) {
            float fps = frame_count / duration.count();
            std::cout << "FPS: " << fps << std::endl;
            frame_count = 0;
            last_fps_update = now;
            
            // Update performance metrics
            performance_optimizer->update_metrics();
        }
        
        // Exit on 'q' key
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}

void VideoProcessingApp::handle_gui_events() {
    // Create parameter adjustment GUI using OpenCV trackbars
    cv::createTrackbar("Motion Threshold", "Controls", nullptr, 100,
                      [](int val, void* userdata) {
                          VideoProcessingApp* app = static_cast<VideoProcessingApp*>(userdata);
                          app->motion_threshold = val;
                      }, this);
    
    cv::createTrackbar("Tracking Sensitivity", "Controls", nullptr, 20,
                      [](int val, void* userdata) {
                          VideoProcessingApp* app = static_cast<VideoProcessingApp*>(userdata);
                          app->tracking_sensitivity = val;
                      }, this);
}
```

## Testing and Validation

```cpp
// Comprehensive test suite
class VideoProcessingTestSuite {
public:
    void run_all_tests() {
        test_performance_benchmarks();
        test_accuracy_metrics();
        test_memory_usage();
        test_multi_threading();
    }
    
private:
    void test_performance_benchmarks() {
        std::cout << "Running performance benchmarks..." << std::endl;
        
        // Test different video resolutions
        std::vector<cv::Size> resolutions = {
            cv::Size(640, 480),   // VGA
            cv::Size(1280, 720),  // HD
            cv::Size(1920, 1080), // Full HD
            cv::Size(3840, 2160)  // 4K
        };
        
        for (const auto& resolution : resolutions) {
            benchmark_resolution(resolution);
        }
    }
    
    void benchmark_resolution(cv::Size resolution) {
        VideoProcessingApp app;
        
        // Create synthetic test video
        std::string test_video = create_test_video(resolution, 100); // 100 frames
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process test video
        app.initialize(test_video, "output_" + std::to_string(resolution.width) + 
                      "x" + std::to_string(resolution.height) + ".mp4");
        app.run();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        float fps = 100.0f * 1000.0f / duration.count();
        
        std::cout << "Resolution " << resolution.width << "x" << resolution.height 
                  << ": " << fps << " FPS" << std::endl;
    }
    
    std::string create_test_video(cv::Size resolution, int num_frames) {
        std::string filename = "test_" + std::to_string(resolution.width) + 
                              "x" + std::to_string(resolution.height) + ".mp4";
        
        cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              30.0, resolution);
        
        // Generate synthetic moving objects
        for (int i = 0; i < num_frames; i++) {
            cv::Mat frame = cv::Mat::zeros(resolution, CV_8UC3);
            
            // Add moving rectangles
            int x = (i * 5) % (resolution.width - 50);
            int y = (i * 3) % (resolution.height - 50);
            cv::rectangle(frame, cv::Rect(x, y, 50, 50), cv::Scalar(0, 255, 0), -1);
            
            writer.write(frame);
        }
        
        return filename;
    }
};
```

## Usage Instructions

1. **Build Requirements**:
   - CUDA Toolkit 11.0+
   - OpenCV 4.5+ with CUDA support
   - CMake 3.16+

2. **Compilation**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j8
   ```

3. **Running the System**:
   ```bash
   ./video_processing_system --input camera --output processed_output.mp4
   ./video_processing_system --input input_video.mp4 --output output.mp4
   ```

## Performance Targets

- **Real-time Processing**: 30+ FPS for 1080p video
- **Low Latency**: <50ms end-to-end processing delay
- **Memory Efficiency**: <2GB GPU memory usage
- **Multi-object Tracking**: Support for 20+ simultaneous objects

## Extensions and Improvements

1. **Advanced Algorithms**: Implement optical flow-based tracking
2. **Deep Learning Integration**: Add neural network-based object detection
3. **Multi-GPU Support**: Scale across multiple GPUs
4. **Cloud Deployment**: Create containerized version for cloud processing
5. **Mobile Optimization**: Optimize for embedded GPU platforms

## Key Learning Outcomes

- Advanced CUDA streaming and memory management
- Real-time video processing optimization techniques
- Multi-threaded CPU-GPU coordination
- Performance profiling and optimization
- Production-ready system architecture design
