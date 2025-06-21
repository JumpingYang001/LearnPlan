# Heterogeneous Programming

*Duration: 1 week*

## Overview

Heterogeneous programming involves coordinating computation between CPUs and GPUs to maximize overall system performance. This section covers work distribution strategies, CPU-GPU synchronization, load balancing, and cooperative computing patterns.

## CPU-GPU Work Distribution

### Task Partitioning Strategies

```cpp
#include <cuda_runtime.h>
#include <thread>
#include <future>
#include <vector>
#include <chrono>

// Heterogeneous task scheduler
class HeterogeneousTaskScheduler {
private:
    struct Task {
        std::function<void()> cpu_work;
        std::function<void(cudaStream_t)> gpu_work;
        float cpu_intensity;  // 0.0 = GPU only, 1.0 = CPU only
        size_t data_size;
        int priority;
    };
    
    std::vector<Task> task_queue;
    std::vector<cudaStream_t> gpu_streams;
    std::vector<std::thread> cpu_threads;
    int num_cpu_cores;
    int num_gpu_streams;
    
public:
    HeterogeneousTaskScheduler(int cpu_cores = 4, int gpu_streams = 4) 
        : num_cpu_cores(cpu_cores), num_gpu_streams(gpu_streams) {
        
        // Create GPU streams
        gpu_streams.resize(num_gpu_streams);
        for (int i = 0; i < num_gpu_streams; i++) {
            cudaStreamCreate(&gpu_streams[i]);
        }
    }
    
    ~HeterogeneousTaskScheduler() {
        for (auto& stream : gpu_streams) {
            cudaStreamDestroy(stream);
        }
    }
    
    void add_task(std::function<void()> cpu_func, 
                  std::function<void(cudaStream_t)> gpu_func,
                  float cpu_intensity, size_t data_size, int priority = 0) {
        Task task;
        task.cpu_work = cpu_func;
        task.gpu_work = gpu_func;
        task.cpu_intensity = cpu_intensity;
        task.data_size = data_size;
        task.priority = priority;
        
        task_queue.push_back(task);
    }
    
    void execute_tasks() {
        // Sort tasks by priority
        std::sort(task_queue.begin(), task_queue.end(),
                 [](const Task& a, const Task& b) {
                     return a.priority > b.priority;
                 });
        
        std::vector<std::future<void>> cpu_futures;
        std::vector<std::future<void>> gpu_futures;
        
        int cpu_task_index = 0;
        int gpu_task_index = 0;
        
        // Distribute tasks based on CPU intensity
        for (const auto& task : task_queue) {
            if (task.cpu_intensity > 0.5f) {
                // CPU-intensive task
                auto cpu_future = std::async(std::launch::async, task.cpu_work);
                cpu_futures.push_back(std::move(cpu_future));
            } else {
                // GPU-intensive task
                int stream_id = gpu_task_index % num_gpu_streams;
                auto gpu_future = std::async(std::launch::async, 
                    [&task, stream = gpu_streams[stream_id]]() {
                        task.gpu_work(stream);
                        cudaStreamSynchronize(stream);
                    });
                gpu_futures.push_back(std::move(gpu_future));
                gpu_task_index++;
            }
        }
        
        // Wait for all tasks to complete
        for (auto& future : cpu_futures) {
            future.wait();
        }
        for (auto& future : gpu_futures) {
            future.wait();
        }
        
        task_queue.clear();
    }
    
    // Hybrid execution for single large task
    void execute_hybrid_task(float* data, size_t n, 
                           std::function<void(float*, size_t)> cpu_process,
                           std::function<void(float*, size_t, cudaStream_t)> gpu_process) {
        
        // Determine optimal split based on profiling
        float cpu_portion = estimate_optimal_split(n);
        size_t cpu_elements = static_cast<size_t>(n * cpu_portion);
        size_t gpu_elements = n - cpu_elements;
        
        // Allocate GPU memory
        float* d_data;
        cudaMalloc(&d_data, gpu_elements * sizeof(float));
        cudaMemcpy(d_data, data + cpu_elements, gpu_elements * sizeof(float), cudaMemcpyHostToDevice);
        
        // Execute CPU and GPU portions concurrently
        auto cpu_future = std::async(std::launch::async, 
            [cpu_process, data, cpu_elements]() {
                cpu_process(data, cpu_elements);
            });
        
        auto gpu_future = std::async(std::launch::async,
            [gpu_process, d_data, gpu_elements, this]() {
                gpu_process(d_data, gpu_elements, gpu_streams[0]);
                cudaStreamSynchronize(gpu_streams[0]);
            });
        
        // Wait for both to complete
        cpu_future.wait();
        gpu_future.wait();
        
        // Copy GPU results back
        cudaMemcpy(data + cpu_elements, d_data, gpu_elements * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
    
private:
    float estimate_optimal_split(size_t n) {
        // Simple heuristic - would be replaced with actual profiling
        if (n < 10000) return 0.8f;  // Small data - CPU heavy
        if (n < 100000) return 0.5f; // Medium data - balanced
        return 0.2f;                 // Large data - GPU heavy
    }
};

// Example usage of heterogeneous task scheduler
void heterogeneous_example() {
    HeterogeneousTaskScheduler scheduler;
    
    const size_t n = 1000000;
    std::vector<float> data(n);
    
    // Initialize data
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // CPU-intensive task
    auto cpu_task = [&data, n]() {
        for (size_t i = 0; i < n / 4; i++) {
            data[i] = std::sin(data[i]) * std::cos(data[i]);
        }
    };
    
    // GPU-intensive task
    auto gpu_task = [&data, n](cudaStream_t stream) {
        float* d_data;
        size_t gpu_size = (n * 3) / 4;
        
        cudaMalloc(&d_data, gpu_size * sizeof(float));
        cudaMemcpyAsync(d_data, data.data() + n/4, gpu_size * sizeof(float), 
                       cudaMemcpyHostToDevice, stream);
        
        dim3 block(256);
        dim3 grid((gpu_size + block.x - 1) / block.x);
        
        gpu_math_kernel<<<grid, block, 0, stream>>>(d_data, gpu_size);
        
        cudaMemcpyAsync(data.data() + n/4, d_data, gpu_size * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        
        cudaFree(d_data);
    };
    
    scheduler.add_task(cpu_task, gpu_task, 0.3f, n * sizeof(float), 1);
    scheduler.execute_tasks();
}

__global__ void gpu_math_kernel(float* data, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sinf(data[idx]) * cosf(data[idx]);
    }
}
```

### Load Balancing Strategies

```cpp
// Dynamic load balancer
class DynamicLoadBalancer {
private:
    struct WorkItem {
        size_t start_idx;
        size_t end_idx;
        std::chrono::high_resolution_clock::time_point submit_time;
        std::chrono::high_resolution_clock::time_point complete_time;
        bool is_complete;
        bool is_gpu_task;
    };
    
    std::vector<WorkItem> work_history;
    std::mutex history_mutex;
    
    // Performance metrics
    float cpu_throughput;  // items per second
    float gpu_throughput;  // items per second
    
public:
    DynamicLoadBalancer() : cpu_throughput(1000.0f), gpu_throughput(10000.0f) {}
    
    void update_performance_metrics() {
        std::lock_guard<std::mutex> lock(history_mutex);
        
        auto now = std::chrono::high_resolution_clock::now();
        auto cutoff = now - std::chrono::seconds(10); // Last 10 seconds
        
        size_t cpu_items = 0, gpu_items = 0;
        std::chrono::duration<float> cpu_time(0), gpu_time(0);
        
        for (const auto& item : work_history) {
            if (item.is_complete && item.submit_time > cutoff) {
                auto duration = item.complete_time - item.submit_time;
                size_t items_processed = item.end_idx - item.start_idx;
                
                if (item.is_gpu_task) {
                    gpu_items += items_processed;
                    gpu_time += duration;
                } else {
                    cpu_items += items_processed;
                    cpu_time += duration;
                }
            }
        }
        
        if (cpu_time.count() > 0) {
            cpu_throughput = cpu_items / cpu_time.count();
        }
        if (gpu_time.count() > 0) {
            gpu_throughput = gpu_items / gpu_time.count();
        }
    }
    
    std::pair<size_t, size_t> calculate_optimal_split(size_t total_items) {
        update_performance_metrics();
        
        // Calculate optimal split based on relative throughput
        float total_throughput = cpu_throughput + gpu_throughput;
        float cpu_ratio = cpu_throughput / total_throughput;
        
        size_t cpu_items = static_cast<size_t>(total_items * cpu_ratio);
        size_t gpu_items = total_items - cpu_items;
        
        return {cpu_items, gpu_items};
    }
    
    void record_work_completion(size_t start_idx, size_t end_idx, bool is_gpu,
                               std::chrono::high_resolution_clock::time_point submit_time) {
        std::lock_guard<std::mutex> lock(history_mutex);
        
        WorkItem item;
        item.start_idx = start_idx;
        item.end_idx = end_idx;
        item.submit_time = submit_time;
        item.complete_time = std::chrono::high_resolution_clock::now();
        item.is_complete = true;
        item.is_gpu_task = is_gpu;
        
        work_history.push_back(item);
        
        // Keep only recent history
        if (work_history.size() > 1000) {
            work_history.erase(work_history.begin(), work_history.begin() + 100);
        }
    }
    
    // Adaptive workload distribution
    void distribute_workload(float* data, size_t n,
                           std::function<void(float*, size_t, size_t)> cpu_func,
                           std::function<void(float*, size_t, size_t, cudaStream_t)> gpu_func) {
        
        auto [cpu_items, gpu_items] = calculate_optimal_split(n);
        
        auto submit_time = std::chrono::high_resolution_clock::now();
        
        // Execute CPU portion
        auto cpu_future = std::async(std::launch::async,
            [this, cpu_func, data, cpu_items, submit_time]() {
                cpu_func(data, 0, cpu_items);
                record_work_completion(0, cpu_items, false, submit_time);
            });
        
        // Execute GPU portion
        auto gpu_future = std::async(std::launch::async,
            [this, gpu_func, data, n, cpu_items, gpu_items, submit_time]() {
                cudaStream_t stream;
                cudaStreamCreate(&stream);
                
                gpu_func(data, cpu_items, n, stream);
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
                
                record_work_completion(cpu_items, n, true, submit_time);
            });
        
        cpu_future.wait();
        gpu_future.wait();
    }
};

// Work stealing scheduler
class WorkStealingScheduler {
private:
    struct WorkQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex mutex;
        std::condition_variable cv;
        bool shutdown;
        
        WorkQueue() : shutdown(false) {}
    };
    
    std::vector<std::unique_ptr<WorkQueue>> cpu_queues;
    std::vector<std::unique_ptr<WorkQueue>> gpu_queues;
    std::vector<std::thread> cpu_workers;
    std::vector<std::thread> gpu_workers;
    
public:
    WorkStealingScheduler(int num_cpu_workers = 4, int num_gpu_workers = 2) {
        // Initialize CPU workers
        cpu_queues.resize(num_cpu_workers);
        for (int i = 0; i < num_cpu_workers; i++) {
            cpu_queues[i] = std::make_unique<WorkQueue>();
            cpu_workers.emplace_back([this, i]() { cpu_worker_loop(i); });
        }
        
        // Initialize GPU workers
        gpu_queues.resize(num_gpu_workers);
        for (int i = 0; i < num_gpu_workers; i++) {
            gpu_queues[i] = std::make_unique<WorkQueue>();
            gpu_workers.emplace_back([this, i]() { gpu_worker_loop(i); });
        }
    }
    
    ~WorkStealingScheduler() {
        shutdown();
    }
    
    void add_cpu_task(std::function<void()> task) {
        // Add to least loaded CPU queue
        int min_queue = 0;
        size_t min_size = SIZE_MAX;
        
        for (int i = 0; i < cpu_queues.size(); i++) {
            std::lock_guard<std::mutex> lock(cpu_queues[i]->mutex);
            if (cpu_queues[i]->tasks.size() < min_size) {
                min_size = cpu_queues[i]->tasks.size();
                min_queue = i;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(cpu_queues[min_queue]->mutex);
            cpu_queues[min_queue]->tasks.push_back(std::move(task));
        }
        cpu_queues[min_queue]->cv.notify_one();
    }
    
    void add_gpu_task(std::function<void()> task) {
        // Add to least loaded GPU queue
        int min_queue = 0;
        size_t min_size = SIZE_MAX;
        
        for (int i = 0; i < gpu_queues.size(); i++) {
            std::lock_guard<std::mutex> lock(gpu_queues[i]->mutex);
            if (gpu_queues[i]->tasks.size() < min_size) {
                min_size = gpu_queues[i]->tasks.size();
                min_queue = i;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(gpu_queues[min_queue]->mutex);
            gpu_queues[min_queue]->tasks.push_back(std::move(task));
        }
        gpu_queues[min_queue]->cv.notify_one();
    }
    
    void shutdown() {
        // Shutdown CPU workers
        for (auto& queue : cpu_queues) {
            {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->shutdown = true;
            }
            queue->cv.notify_all();
        }
        
        // Shutdown GPU workers
        for (auto& queue : gpu_queues) {
            {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->shutdown = true;
            }
            queue->cv.notify_all();
        }
        
        // Join all threads
        for (auto& worker : cpu_workers) {
            worker.join();
        }
        for (auto& worker : gpu_workers) {
            worker.join();
        }
    }
    
private:
    void cpu_worker_loop(int worker_id) {
        while (true) {
            std::function<void()> task;
            
            // Try to get task from own queue
            {
                std::unique_lock<std::mutex> lock(cpu_queues[worker_id]->mutex);
                cpu_queues[worker_id]->cv.wait(lock, [this, worker_id]() {
                    return !cpu_queues[worker_id]->tasks.empty() || cpu_queues[worker_id]->shutdown;
                });
                
                if (cpu_queues[worker_id]->shutdown) {
                    break;
                }
                
                if (!cpu_queues[worker_id]->tasks.empty()) {
                    task = std::move(cpu_queues[worker_id]->tasks.front());
                    cpu_queues[worker_id]->tasks.pop_front();
                }
            }
            
            // If no task, try to steal from another queue
            if (!task) {
                task = steal_cpu_task(worker_id);
            }
            
            if (task) {
                task();
            }
        }
    }
    
    void gpu_worker_loop(int worker_id) {
        // Set GPU context
        cudaSetDevice(worker_id % 4); // Assuming up to 4 GPUs
        
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(gpu_queues[worker_id]->mutex);
                gpu_queues[worker_id]->cv.wait(lock, [this, worker_id]() {
                    return !gpu_queues[worker_id]->tasks.empty() || gpu_queues[worker_id]->shutdown;
                });
                
                if (gpu_queues[worker_id]->shutdown) {
                    break;
                }
                
                if (!gpu_queues[worker_id]->tasks.empty()) {
                    task = std::move(gpu_queues[worker_id]->tasks.front());
                    gpu_queues[worker_id]->tasks.pop_front();
                }
            }
            
            if (!task) {
                task = steal_gpu_task(worker_id);
            }
            
            if (task) {
                task();
            }
        }
    }
    
    std::function<void()> steal_cpu_task(int exclude_worker) {
        for (int i = 0; i < cpu_queues.size(); i++) {
            if (i == exclude_worker) continue;
            
            std::lock_guard<std::mutex> lock(cpu_queues[i]->mutex);
            if (!cpu_queues[i]->tasks.empty()) {
                auto task = std::move(cpu_queues[i]->tasks.back());
                cpu_queues[i]->tasks.pop_back();
                return task;
            }
        }
        return nullptr;
    }
    
    std::function<void()> steal_gpu_task(int exclude_worker) {
        for (int i = 0; i < gpu_queues.size(); i++) {
            if (i == exclude_worker) continue;
            
            std::lock_guard<std::mutex> lock(gpu_queues[i]->mutex);
            if (!gpu_queues[i]->tasks.empty()) {
                auto task = std::move(gpu_queues[i]->tasks.back());
                gpu_queues[i]->tasks.pop_back();
                return task;
            }
        }
        return nullptr;
    }
};
```

## CPU-GPU Synchronization

### Event-based Synchronization

```cpp
// Advanced CPU-GPU synchronization
class HeterogeneousSynchronizer {
private:
    struct SyncPoint {
        cudaEvent_t gpu_event;
        std::promise<void> cpu_promise;
        std::future<void> cpu_future;
        bool is_active;
        
        SyncPoint() : is_active(false) {
            cudaEventCreate(&gpu_event);
            cpu_future = cpu_promise.get_future();
        }
        
        ~SyncPoint() {
            cudaEventDestroy(gpu_event);
        }
    };
    
    std::vector<std::unique_ptr<SyncPoint>> sync_points;
    std::mutex sync_mutex;
    
public:
    int create_sync_point() {
        std::lock_guard<std::mutex> lock(sync_mutex);
        
        sync_points.push_back(std::make_unique<SyncPoint>());
        return sync_points.size() - 1;
    }
    
    void gpu_signal(int sync_id, cudaStream_t stream) {
        if (sync_id < 0 || sync_id >= sync_points.size()) return;
        
        auto& sync_point = sync_points[sync_id];
        cudaEventRecord(sync_point->gpu_event, stream);
        sync_point->is_active = true;
    }
    
    void cpu_signal(int sync_id) {
        if (sync_id < 0 || sync_id >= sync_points.size()) return;
        
        auto& sync_point = sync_points[sync_id];
        sync_point->cpu_promise.set_value();
        sync_point->is_active = true;
    }
    
    void wait_for_gpu(int sync_id) {
        if (sync_id < 0 || sync_id >= sync_points.size()) return;
        
        auto& sync_point = sync_points[sync_id];
        if (sync_point->is_active) {
            cudaEventSynchronize(sync_point->gpu_event);
        }
    }
    
    void wait_for_cpu(int sync_id) {
        if (sync_id < 0 || sync_id >= sync_points.size()) return;
        
        auto& sync_point = sync_points[sync_id];
        if (sync_point->is_active) {
            sync_point->cpu_future.wait();
        }
    }
    
    void wait_for_all() {
        for (auto& sync_point : sync_points) {
            if (sync_point->is_active) {
                cudaEventSynchronize(sync_point->gpu_event);
                sync_point->cpu_future.wait();
            }
        }
    }
    
    // Complex synchronization pattern
    void execute_pipeline(std::vector<std::function<void()>> cpu_stages,
                         std::vector<std::function<void(cudaStream_t)>> gpu_stages) {
        
        std::vector<cudaStream_t> streams(gpu_stages.size());
        std::vector<int> sync_points(cpu_stages.size() + gpu_stages.size());
        
        // Create streams and sync points
        for (int i = 0; i < gpu_stages.size(); i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        for (int i = 0; i < sync_points.size(); i++) {
            sync_points[i] = create_sync_point();
        }
        
        // Launch CPU stages
        std::vector<std::future<void>> cpu_futures;
        for (int i = 0; i < cpu_stages.size(); i++) {
            auto future = std::async(std::launch::async, [this, &cpu_stages, i, &sync_points]() {
                cpu_stages[i]();
                cpu_signal(sync_points[i]);
            });
            cpu_futures.push_back(std::move(future));
        }
        
        // Launch GPU stages with dependencies
        std::vector<std::future<void>> gpu_futures;
        for (int i = 0; i < gpu_stages.size(); i++) {
            auto future = std::async(std::launch::async, 
                [this, &gpu_stages, i, &streams, &sync_points, cpu_stages_size = cpu_stages.size()]() {
                    // Wait for corresponding CPU stage if exists
                    if (i < cpu_stages_size) {
                        wait_for_cpu(sync_points[i]);
                    }
                    
                    gpu_stages[i](streams[i]);
                    gpu_signal(sync_points[cpu_stages_size + i], streams[i]);
                });
            gpu_futures.push_back(std::move(future));
        }
        
        // Wait for all stages to complete
        for (auto& future : cpu_futures) {
            future.wait();
        }
        for (auto& future : gpu_futures) {
            future.wait();
        }
        
        // Cleanup
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
    }
};

// Producer-Consumer pattern with CPU-GPU
template<typename T>
class HeterogeneousQueue {
private:
    std::queue<T> cpu_to_gpu_queue;
    std::queue<T> gpu_to_cpu_queue;
    std::mutex cpu_to_gpu_mutex;
    std::mutex gpu_to_cpu_mutex;
    std::condition_variable cpu_to_gpu_cv;
    std::condition_variable gpu_to_cpu_cv;
    
    size_t max_queue_size;
    bool shutdown_flag;
    
public:
    HeterogeneousQueue(size_t max_size = 100) 
        : max_queue_size(max_size), shutdown_flag(false) {}
    
    void cpu_produce(const T& item) {
        std::unique_lock<std::mutex> lock(cpu_to_gpu_mutex);
        cpu_to_gpu_cv.wait(lock, [this]() {
            return cpu_to_gpu_queue.size() < max_queue_size || shutdown_flag;
        });
        
        if (!shutdown_flag) {
            cpu_to_gpu_queue.push(item);
            cpu_to_gpu_cv.notify_one();
        }
    }
    
    bool gpu_consume(T& item) {
        std::unique_lock<std::mutex> lock(cpu_to_gpu_mutex);
        cpu_to_gpu_cv.wait(lock, [this]() {
            return !cpu_to_gpu_queue.empty() || shutdown_flag;
        });
        
        if (shutdown_flag && cpu_to_gpu_queue.empty()) {
            return false;
        }
        
        item = cpu_to_gpu_queue.front();
        cpu_to_gpu_queue.pop();
        cpu_to_gpu_cv.notify_one();
        return true;
    }
    
    void gpu_produce(const T& item) {
        std::unique_lock<std::mutex> lock(gpu_to_cpu_mutex);
        gpu_to_cpu_cv.wait(lock, [this]() {
            return gpu_to_cpu_queue.size() < max_queue_size || shutdown_flag;
        });
        
        if (!shutdown_flag) {
            gpu_to_cpu_queue.push(item);
            gpu_to_cpu_cv.notify_one();
        }
    }
    
    bool cpu_consume(T& item) {
        std::unique_lock<std::mutex> lock(gpu_to_cpu_mutex);
        gpu_to_cpu_cv.wait(lock, [this]() {
            return !gpu_to_cpu_queue.empty() || shutdown_flag;
        });
        
        if (shutdown_flag && gpu_to_cpu_queue.empty()) {
            return false;
        }
        
        item = gpu_to_cpu_queue.front();
        gpu_to_cpu_queue.pop();
        gpu_to_cpu_cv.notify_one();
        return true;
    }
    
    void shutdown() {
        shutdown_flag = true;
        cpu_to_gpu_cv.notify_all();
        gpu_to_cpu_cv.notify_all();
    }
};
```

## Cooperative Computing Patterns

### Pipeline Processing

```cpp
// Heterogeneous pipeline processing
template<typename InputType, typename OutputType>
class HeterogeneousPipeline {
private:
    struct PipelineStage {
        std::string name;
        std::function<void(InputType&, OutputType&)> cpu_process;
        std::function<void(InputType*, OutputType*, size_t, cudaStream_t)> gpu_process;
        bool use_gpu;
        float processing_time_estimate;
    };
    
    std::vector<PipelineStage> stages;
    HeterogeneousQueue<InputType> input_queue;
    HeterogeneousQueue<OutputType> output_queue;
    std::vector<cudaStream_t> gpu_streams;
    
public:
    HeterogeneousPipeline() {
        // Create GPU streams for pipeline stages
        gpu_streams.resize(4);
        for (int i = 0; i < 4; i++) {
            cudaStreamCreate(&gpu_streams[i]);
        }
    }
    
    ~HeterogeneousPipeline() {
        for (auto& stream : gpu_streams) {
            cudaStreamDestroy(stream);
        }
    }
    
    void add_stage(const std::string& name, bool use_gpu,
                   std::function<void(InputType&, OutputType&)> cpu_func = nullptr,
                   std::function<void(InputType*, OutputType*, size_t, cudaStream_t)> gpu_func = nullptr) {
        PipelineStage stage;
        stage.name = name;
        stage.use_gpu = use_gpu;
        stage.cpu_process = cpu_func;
        stage.gpu_process = gpu_func;
        stage.processing_time_estimate = 1.0f; // Default estimate
        
        stages.push_back(stage);
    }
    
    void process_batch(std::vector<InputType>& inputs, std::vector<OutputType>& outputs) {
        outputs.resize(inputs.size());
        
        // Determine optimal batch distribution
        auto batch_distribution = optimize_batch_distribution(inputs.size());
        
        // Process different portions on CPU and GPU simultaneously
        std::vector<std::future<void>> futures;
        
        for (const auto& batch : batch_distribution) {
            if (batch.use_gpu) {
                auto future = std::async(std::launch::async,
                    [this, &inputs, &outputs, batch]() {
                        process_gpu_batch(inputs, outputs, batch.start_idx, batch.end_idx, batch.stream_id);
                    });
                futures.push_back(std::move(future));
            } else {
                auto future = std::async(std::launch::async,
                    [this, &inputs, &outputs, batch]() {
                        process_cpu_batch(inputs, outputs, batch.start_idx, batch.end_idx);
                    });
                futures.push_back(std::move(future));
            }
        }
        
        // Wait for all batches to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
    
    void start_streaming_pipeline() {
        // Start producer thread
        std::thread producer([this]() {
            InputType item;
            while (input_queue.cpu_consume(item)) {
                // Process through pipeline stages
                process_item_through_pipeline(item);
            }
        });
        
        // Start consumer thread
        std::thread consumer([this]() {
            OutputType result;
            while (output_queue.cpu_consume(result)) {
                // Handle processed result
                handle_pipeline_output(result);
            }
        });
        
        producer.detach();
        consumer.detach();
    }
    
private:
    struct BatchInfo {
        size_t start_idx;
        size_t end_idx;
        bool use_gpu;
        int stream_id;
    };
    
    std::vector<BatchInfo> optimize_batch_distribution(size_t total_items) {
        std::vector<BatchInfo> batches;
        
        // Simple heuristic - would be replaced with actual optimization
        size_t gpu_items = total_items * 0.7; // 70% on GPU
        size_t cpu_items = total_items - gpu_items;
        
        if (cpu_items > 0) {
            BatchInfo cpu_batch;
            cpu_batch.start_idx = 0;
            cpu_batch.end_idx = cpu_items;
            cpu_batch.use_gpu = false;
            cpu_batch.stream_id = -1;
            batches.push_back(cpu_batch);
        }
        
        if (gpu_items > 0) {
            // Split GPU work across multiple streams
            size_t items_per_stream = gpu_items / gpu_streams.size();
            for (int i = 0; i < gpu_streams.size(); i++) {
                BatchInfo gpu_batch;
                gpu_batch.start_idx = cpu_items + i * items_per_stream;
                gpu_batch.end_idx = (i == gpu_streams.size() - 1) ? 
                                   total_items : 
                                   cpu_items + (i + 1) * items_per_stream;
                gpu_batch.use_gpu = true;
                gpu_batch.stream_id = i;
                batches.push_back(gpu_batch);
            }
        }
        
        return batches;
    }
    
    void process_cpu_batch(std::vector<InputType>& inputs, std::vector<OutputType>& outputs,
                          size_t start_idx, size_t end_idx) {
        for (size_t i = start_idx; i < end_idx; i++) {
            OutputType temp_output = outputs[i];
            
            for (const auto& stage : stages) {
                if (!stage.use_gpu && stage.cpu_process) {
                    if (i == start_idx) { // First item
                        stage.cpu_process(inputs[i], temp_output);
                    } else {
                        InputType temp_input = outputs[i-1]; // Use previous output as input
                        stage.cpu_process(temp_input, temp_output);
                    }
                }
            }
            
            outputs[i] = temp_output;
        }
    }
    
    void process_gpu_batch(std::vector<InputType>& inputs, std::vector<OutputType>& outputs,
                          size_t start_idx, size_t end_idx, int stream_id) {
        size_t batch_size = end_idx - start_idx;
        
        // Allocate GPU memory
        InputType* d_inputs;
        OutputType* d_outputs;
        
        cudaMalloc(&d_inputs, batch_size * sizeof(InputType));
        cudaMalloc(&d_outputs, batch_size * sizeof(OutputType));
        
        // Copy input data to GPU
        cudaMemcpyAsync(d_inputs, &inputs[start_idx], batch_size * sizeof(InputType),
                       cudaMemcpyHostToDevice, gpu_streams[stream_id]);
        
        // Execute GPU stages
        for (const auto& stage : stages) {
            if (stage.use_gpu && stage.gpu_process) {
                stage.gpu_process(d_inputs, d_outputs, batch_size, gpu_streams[stream_id]);
                
                // Swap input and output for next stage
                std::swap(d_inputs, d_outputs);
            }
        }
        
        // Copy results back to CPU
        cudaMemcpyAsync(&outputs[start_idx], d_outputs, batch_size * sizeof(OutputType),
                       cudaMemcpyDeviceToHost, gpu_streams[stream_id]);
        
        cudaStreamSynchronize(gpu_streams[stream_id]);
        
        // Cleanup
        cudaFree(d_inputs);
        cudaFree(d_outputs);
    }
    
    void process_item_through_pipeline(const InputType& input) {
        // Implementation for streaming processing
        OutputType output;
        
        // Apply all pipeline stages
        for (const auto& stage : stages) {
            if (stage.use_gpu) {
                // Process single item on GPU (for demonstration)
                InputType* d_input;
                OutputType* d_output;
                
                cudaMalloc(&d_input, sizeof(InputType));
                cudaMalloc(&d_output, sizeof(OutputType));
                
                cudaMemcpy(d_input, &input, sizeof(InputType), cudaMemcpyHostToDevice);
                
                if (stage.gpu_process) {
                    stage.gpu_process(d_input, d_output, 1, gpu_streams[0]);
                }
                
                cudaMemcpy(&output, d_output, sizeof(OutputType), cudaMemcpyDeviceToHost);
                
                cudaFree(d_input);
                cudaFree(d_output);
            } else {
                if (stage.cpu_process) {
                    InputType temp_input = input;
                    stage.cpu_process(temp_input, output);
                }
            }
        }
        
        output_queue.gpu_produce(output);
    }
    
    void handle_pipeline_output(const OutputType& output) {
        // Handle the processed output
        // This would be application-specific
    }
};

// Example usage of heterogeneous pipeline
void pipeline_example() {
    HeterogeneousPipeline<float, float> pipeline;
    
    // Add pipeline stages
    pipeline.add_stage("Preprocessing", false, 
        [](float& input, float& output) {
            output = input * 2.0f; // Simple CPU preprocessing
        });
    
    pipeline.add_stage("Main Processing", true, nullptr,
        [](float* inputs, float* outputs, size_t n, cudaStream_t stream) {
            dim3 block(256);
            dim3 grid((n + block.x - 1) / block.x);
            
            main_processing_kernel<<<grid, block, 0, stream>>>(inputs, outputs, n);
        });
    
    pipeline.add_stage("Postprocessing", false,
        [](float& input, float& output) {
            output = sqrtf(input); // Simple CPU postprocessing
        });
    
    // Process a batch
    std::vector<float> inputs(10000);
    std::vector<float> outputs;
    
    // Initialize inputs
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = static_cast<float>(i);
    }
    
    pipeline.process_batch(inputs, outputs);
}

__global__ void main_processing_kernel(float* inputs, float* outputs, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        outputs[idx] = sinf(inputs[idx]) * cosf(inputs[idx]);
    }
}
```

## Performance Monitoring and Optimization

```cpp
// Heterogeneous performance profiler
class HeterogeneousProfiler {
private:
    struct ProfileData {
        std::string operation_name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        bool is_gpu_operation;
        size_t data_size;
        float gflops;
    };
    
    std::vector<ProfileData> profile_history;
    std::mutex profile_mutex;
    
public:
    void start_profile(const std::string& name, bool is_gpu, size_t data_size = 0) {
        std::lock_guard<std::mutex> lock(profile_mutex);
        
        ProfileData data;
        data.operation_name = name;
        data.start_time = std::chrono::high_resolution_clock::now();
        data.is_gpu_operation = is_gpu;
        data.data_size = data_size;
        data.gflops = 0.0f;
        
        profile_history.push_back(data);
    }
    
    void end_profile(const std::string& name, float operations = 0.0f) {
        std::lock_guard<std::mutex> lock(profile_mutex);
        
        auto now = std::chrono::high_resolution_clock::now();
        
        // Find the most recent matching profile
        for (auto it = profile_history.rbegin(); it != profile_history.rend(); ++it) {
            if (it->operation_name == name && it->end_time == std::chrono::high_resolution_clock::time_point{}) {
                it->end_time = now;
                
                if (operations > 0.0f) {
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(it->end_time - it->start_time);
                    it->gflops = operations / (duration.count() / 1e6f) / 1e9f;
                }
                break;
            }
        }
    }
    
    void print_performance_summary() {
        std::lock_guard<std::mutex> lock(profile_mutex);
        
        std::map<std::string, std::vector<ProfileData*>> grouped_profiles;
        
        for (auto& profile : profile_history) {
            if (profile.end_time != std::chrono::high_resolution_clock::time_point{}) {
                grouped_profiles[profile.operation_name].push_back(&profile);
            }
        }
        
        printf("\n=== Performance Summary ===\n");
        for (const auto& [name, profiles] : grouped_profiles) {
            float total_time = 0.0f;
            float avg_gflops = 0.0f;
            int gpu_count = 0, cpu_count = 0;
            
            for (const auto* profile : profiles) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    profile->end_time - profile->start_time);
                total_time += duration.count() / 1000.0f; // Convert to ms
                avg_gflops += profile->gflops;
                
                if (profile->is_gpu_operation) gpu_count++;
                else cpu_count++;
            }
            
            avg_gflops /= profiles.size();
            
            printf("Operation: %s\n", name.c_str());
            printf("  Total executions: %zu (GPU: %d, CPU: %d)\n", profiles.size(), gpu_count, cpu_count);
            printf("  Total time: %.2f ms\n", total_time);
            printf("  Average time: %.2f ms\n", total_time / profiles.size());
            printf("  Average GFLOPS: %.2f\n", avg_gflops);
            printf("\n");
        }
    }
    
    float get_cpu_gpu_efficiency_ratio() {
        std::lock_guard<std::mutex> lock(profile_mutex);
        
        float cpu_total_time = 0.0f;
        float gpu_total_time = 0.0f;
        int cpu_ops = 0, gpu_ops = 0;
        
        for (const auto& profile : profile_history) {
            if (profile.end_time != std::chrono::high_resolution_clock::time_point{}) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    profile.end_time - profile.start_time);
                float time_ms = duration.count() / 1000.0f;
                
                if (profile.is_gpu_operation) {
                    gpu_total_time += time_ms;
                    gpu_ops++;
                } else {
                    cpu_total_time += time_ms;
                    cpu_ops++;
                }
            }
        }
        
        if (cpu_ops == 0 || gpu_ops == 0) return 1.0f;
        
        float cpu_avg = cpu_total_time / cpu_ops;
        float gpu_avg = gpu_total_time / gpu_ops;
        
        return cpu_avg / gpu_avg; // > 1 means GPU is faster on average
    }
};

// RAII profiler helper
class ScopedProfiler {
private:
    HeterogeneousProfiler* profiler;
    std::string operation_name;
    float operations;
    
public:
    ScopedProfiler(HeterogeneousProfiler* prof, const std::string& name, 
                   bool is_gpu, size_t data_size = 0, float ops = 0.0f)
        : profiler(prof), operation_name(name), operations(ops) {
        profiler->start_profile(name, is_gpu, data_size);
    }
    
    ~ScopedProfiler() {
        profiler->end_profile(operation_name, operations);
    }
};

#define PROFILE_SCOPE(profiler, name, is_gpu, data_size, ops) \
    ScopedProfiler _prof(profiler, name, is_gpu, data_size, ops)

// Example usage of heterogeneous profiling
void profiling_example() {
    HeterogeneousProfiler profiler;
    
    const size_t n = 1000000;
    std::vector<float> data(n);
    float* d_data;
    
    cudaMalloc(&d_data, n * sizeof(float));
    
    // Profile CPU operation
    {
        PROFILE_SCOPE(&profiler, "CPU Processing", false, n * sizeof(float), n);
        
        for (size_t i = 0; i < n; i++) {
            data[i] = sinf(static_cast<float>(i)) * cosf(static_cast<float>(i));
        }
    }
    
    // Profile GPU operation
    {
        PROFILE_SCOPE(&profiler, "GPU Processing", true, n * sizeof(float), n);
        
        cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        gpu_math_kernel<<<grid, block>>>(d_data, n);
        cudaDeviceSynchronize();
        
        cudaMemcpy(data.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    profiler.print_performance_summary();
    
    float efficiency_ratio = profiler.get_cpu_gpu_efficiency_ratio();
    printf("CPU/GPU efficiency ratio: %.2f\n", efficiency_ratio);
    
    cudaFree(d_data);
}
```

## Exercises

1. **Dynamic Load Balancer**: Implement a load balancer that adapts workload distribution based on real-time performance metrics.

2. **Heterogeneous Pipeline**: Create a multi-stage pipeline that automatically determines the optimal placement of each stage (CPU or GPU).

3. **Work Stealing Scheduler**: Implement a work stealing scheduler that can dynamically redistribute tasks between CPU and GPU workers.

4. **Synchronization Patterns**: Design and implement complex synchronization patterns for producer-consumer scenarios.

5. **Performance Profiler**: Build a comprehensive profiling system that can identify bottlenecks in heterogeneous applications.

## Key Takeaways

- Effective heterogeneous programming requires careful workload distribution and synchronization
- Dynamic load balancing can significantly improve overall system utilization
- Pipeline processing enables concurrent execution of different stages on different processors
- Profiling and monitoring are essential for optimizing heterogeneous applications
- Work stealing can help balance load dynamically across available resources

## Next Steps

Proceed to [CUDA Graph API](11_CUDA_Graph_API.md) to learn about optimizing kernel launch overhead and creating efficient execution graphs.
