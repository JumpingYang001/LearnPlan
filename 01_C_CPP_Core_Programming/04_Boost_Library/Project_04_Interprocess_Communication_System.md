# Project 4: Interprocess Communication System

*Estimated Duration: 3-4 weeks*
*Difficulty: Advanced*

## Project Overview

Design and implement a comprehensive interprocess communication (IPC) system using Boost.Interprocess. This project demonstrates advanced IPC concepts including shared memory, message queues, named pipes, and synchronization primitives across multiple processes.

## Learning Objectives

- Master Boost.Interprocess for various IPC mechanisms
- Understand shared memory management and synchronization
- Implement message passing systems with different patterns
- Handle cross-platform IPC considerations
- Design fault-tolerant distributed systems
- Implement process discovery and lifecycle management

## Project Requirements

### Core Features

1. **Shared Memory Management**
   - Dynamic shared memory allocation and deallocation
   - Object construction/destruction in shared memory
   - Memory mapping with custom allocators
   - Version control and schema evolution

2. **Message Passing Systems**
   - Reliable message queues with persistence
   - Publish/subscribe messaging patterns
   - Request/response communication patterns
   - Broadcast and multicast messaging

3. **Process Coordination**
   - Named semaphores and mutexes
   - Condition variables for process synchronization
   - Reader/writer locks for shared resources
   - Barrier synchronization for multiple processes

### Advanced Features

4. **Service Discovery**
   - Process registration and discovery
   - Health monitoring and heartbeat systems
   - Load balancing across multiple service instances
   - Failover and redundancy management

5. **Data Serialization**
   - Efficient binary serialization
   - Schema versioning and compatibility
   - Cross-platform data format handling
   - Compression and encryption support

6. **Monitoring and Management**
   - Real-time performance metrics
   - Process lifecycle management
   - Resource usage monitoring
   - Administrative control interface

## Implementation Guide

### Step 1: Core IPC Infrastructure

```cpp
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <sstream>

namespace bip = boost::interprocess;
namespace bser = boost::serialization;

// Type definitions for shared memory allocators
typedef bip::managed_shared_memory::segment_manager SegmentManager;
typedef bip::allocator<void, SegmentManager> VoidAllocator;
typedef bip::allocator<char, SegmentManager> CharAllocator;
typedef bip::basic_string<char, std::char_traits<char>, CharAllocator> ShmString;
typedef bip::allocator<ShmString, SegmentManager> StringAllocator;
typedef bip::vector<ShmString, StringAllocator> ShmStringVector;

// Forward declarations
class IPCManager;
class SharedMemoryManager;
class MessageQueue;
class ProcessRegistry;
```

### Step 2: Shared Memory Management

```cpp
class SharedMemoryManager {
public:
    SharedMemoryManager(const std::string& segment_name, size_t size) 
        : segment_name_(segment_name) {
        try {
            // Try to open existing segment
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::open_only, segment_name_.c_str());
        } catch (const bip::interprocess_exception&) {
            // Create new segment if it doesn't exist
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::create_only, segment_name_.c_str(), size);
            initialize_segment();
        }
    }
    
    ~SharedMemoryManager() {
        cleanup();
    }
    
    template<typename T>
    T* construct_object(const std::string& name) {
        try {
            return segment_->construct<T>(name.c_str())();
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error constructing object " << name << ": " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    template<typename T, typename... Args>
    T* construct_object(const std::string& name, Args&&... args) {
        try {
            return segment_->construct<T>(name.c_str())(std::forward<Args>(args)...);
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error constructing object " << name << ": " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    template<typename T>
    T* find_object(const std::string& name) {
        try {
            auto result = segment_->find<T>(name.c_str());
            return result.first;
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error finding object " << name << ": " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    bool destroy_object(const std::string& name) {
        try {
            return segment_->destroy<bip::named_object>(name.c_str());
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error destroying object " << name << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    VoidAllocator get_allocator() {
        return VoidAllocator(segment_->get_segment_manager());
    }
    
    size_t get_free_memory() const {
        return segment_->get_free_memory();
    }
    
    size_t get_size() const {
        return segment_->get_size();
    }
    
    void print_memory_stats() const {
        std::cout << "Shared Memory Statistics:" << std::endl;
        std::cout << "  Segment name: " << segment_name_ << std::endl;
        std::cout << "  Total size: " << get_size() << " bytes" << std::endl;
        std::cout << "  Free memory: " << get_free_memory() << " bytes" << std::endl;
        std::cout << "  Used memory: " << (get_size() - get_free_memory()) << " bytes" << std::endl;
    }
    
    static void remove_segment(const std::string& segment_name) {
        bip::shared_memory_object::remove(segment_name.c_str());
    }
    
private:
    std::string segment_name_;
    std::unique_ptr<bip::managed_shared_memory> segment_;
    
    void initialize_segment() {
        // Initialize segment metadata
        auto allocator = get_allocator();
        
        // Create segment info structure
        struct SegmentInfo {
            std::atomic<int> process_count{0};
            std::atomic<bool> shutdown_requested{false};
            bip::interprocess_mutex mutex;
            
            SegmentInfo() = default;
        };
        
        construct_object<SegmentInfo>("segment_info");
    }
    
    void cleanup() {
        // Decrement process count and cleanup if necessary
        auto* info = find_object<struct SegmentInfo>("segment_info");
        if (info) {
            int count = --info->process_count;
            if (count <= 0 && info->shutdown_requested) {
                // Last process - cleanup segment
                remove_segment(segment_name_);
            }
        }
    }
    
    struct SegmentInfo {
        std::atomic<int> process_count{0};
        std::atomic<bool> shutdown_requested{false};
        bip::interprocess_mutex mutex;
    };
};

// Shared data structures
template<typename T>
class SharedQueue {
public:
    SharedQueue(SharedMemoryManager& shm_manager, const std::string& name, size_t max_size = 1000)
        : shm_manager_(shm_manager), name_(name), max_size_(max_size) {
        
        // Try to find existing queue
        queue_data_ = shm_manager_.find_object<QueueData>(name_);
        
        if (!queue_data_) {
            // Create new queue
            auto allocator = shm_manager_.get_allocator();
            queue_data_ = shm_manager_.construct_object<QueueData>(name_, allocator, max_size_);
        }
    }
    
    bool push(const T& item) {
        bip::scoped_lock<bip::interprocess_mutex> lock(queue_data_->mutex);
        
        if (queue_data_->items.size() >= max_size_) {
            return false; // Queue is full
        }
        
        queue_data_->items.push_back(item);
        queue_data_->condition.notify_one();
        return true;
    }
    
    bool pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) {
        bip::scoped_lock<bip::interprocess_mutex> lock(queue_data_->mutex);
        
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (queue_data_->items.empty()) {
            if (timeout.count() == 0) {
                return false; // Non-blocking pop
            }
            
            auto remaining = deadline - std::chrono::steady_clock::now();
            if (remaining <= std::chrono::milliseconds(0)) {
                return false; // Timeout
            }
            
            // Wait with timeout (simplified - actual implementation would use timed_wait)
            queue_data_->condition.wait(lock);
        }
        
        item = queue_data_->items.front();
        queue_data_->items.pop_front();
        return true;
    }
    
    size_t size() const {
        bip::scoped_lock<bip::interprocess_mutex> lock(queue_data_->mutex);
        return queue_data_->items.size();
    }
    
    bool empty() const {
        bip::scoped_lock<bip::interprocess_mutex> lock(queue_data_->mutex);
        return queue_data_->items.empty();
    }
    
private:
    typedef bip::allocator<T, SegmentManager> ItemAllocator;
    typedef bip::deque<T, ItemAllocator> ItemQueue;
    
    struct QueueData {
        ItemQueue items;
        mutable bip::interprocess_mutex mutex;
        bip::interprocess_condition condition;
        
        QueueData(const VoidAllocator& allocator, size_t max_size)
            : items(ItemAllocator(allocator)) {}
    };
    
    SharedMemoryManager& shm_manager_;
    std::string name_;
    size_t max_size_;
    QueueData* queue_data_;
};

template<typename Key, typename Value>
class SharedMap {
public:
    SharedMap(SharedMemoryManager& shm_manager, const std::string& name)
        : shm_manager_(shm_manager), name_(name) {
        
        // Try to find existing map
        map_data_ = shm_manager_.find_object<MapData>(name_);
        
        if (!map_data_) {
            // Create new map
            auto allocator = shm_manager_.get_allocator();
            map_data_ = shm_manager_.construct_object<MapData>(name_, allocator);
        }
    }
    
    bool insert(const Key& key, const Value& value) {
        bip::scoped_lock<bip::interprocess_mutex> lock(map_data_->mutex);
        auto result = map_data_->map.insert(std::make_pair(key, value));
        return result.second;
    }
    
    bool find(const Key& key, Value& value) const {
        bip::scoped_lock<bip::interprocess_mutex> lock(map_data_->mutex);
        auto it = map_data_->map.find(key);
        if (it != map_data_->map.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    bool erase(const Key& key) {
        bip::scoped_lock<bip::interprocess_mutex> lock(map_data_->mutex);
        return map_data_->map.erase(key) > 0;
    }
    
    size_t size() const {
        bip::scoped_lock<bip::interprocess_mutex> lock(map_data_->mutex);
        return map_data_->map.size();
    }
    
    void clear() {
        bip::scoped_lock<bip::interprocess_mutex> lock(map_data_->mutex);
        map_data_->map.clear();
    }
    
private:
    typedef std::pair<const Key, Value> ValueType;
    typedef bip::allocator<ValueType, SegmentManager> PairAllocator;
    typedef bip::map<Key, Value, std::less<Key>, PairAllocator> SharedMapType;
    
    struct MapData {
        SharedMapType map;
        mutable bip::interprocess_mutex mutex;
        
        MapData(const VoidAllocator& allocator)
            : map(std::less<Key>(), PairAllocator(allocator)) {}
    };
    
    SharedMemoryManager& shm_manager_;
    std::string name_;
    MapData* map_data_;
};
```

### Step 3: Message Queue System

```cpp
class Message {
public:
    enum Type {
        REQUEST,
        RESPONSE,
        NOTIFICATION,
        HEARTBEAT,
        SHUTDOWN
    };
    
    Message() : type_(REQUEST), timestamp_(std::chrono::system_clock::now()) {}
    
    Message(Type type, const std::string& sender, const std::string& data)
        : type_(type), sender_(sender), data_(data), 
          timestamp_(std::chrono::system_clock::now()) {
        id_ = generate_message_id();
    }
    
    Type type() const { return type_; }
    const std::string& id() const { return id_; }
    const std::string& sender() const { return sender_; }
    const std::string& data() const { return data_; }
    const std::string& correlation_id() const { return correlation_id_; }
    auto timestamp() const { return timestamp_; }
    
    void set_correlation_id(const std::string& correlation_id) {
        correlation_id_ = correlation_id;
    }
    
    void set_reply_to(const std::string& reply_to) {
        reply_to_ = reply_to;
    }
    
    const std::string& reply_to() const { return reply_to_; }
    
    std::string serialize() const {
        std::ostringstream oss;
        boost::archive::binary_oarchive oa(oss);
        oa << *this;
        return oss.str();
    }
    
    static Message deserialize(const std::string& data) {
        std::istringstream iss(data);
        boost::archive::binary_iarchive ia(iss);
        Message msg;
        ia >> msg;
        return msg;
    }
    
private:
    Type type_;
    std::string id_;
    std::string sender_;
    std::string data_;
    std::string correlation_id_;
    std::string reply_to_;
    std::chrono::system_clock::time_point timestamp_;
    
    std::string generate_message_id() const {
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        return boost::uuids::to_string(uuid);
    }
    
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & type_;
        ar & id_;
        ar & sender_;
        ar & data_;
        ar & correlation_id_;
        ar & reply_to_;
        ar & timestamp_;
    }
};

class MessageQueue {
public:
    MessageQueue(const std::string& queue_name, size_t max_messages = 100, 
                size_t max_message_size = 4096)
        : queue_name_(queue_name), max_message_size_(max_message_size) {
        
        try {
            // Try to open existing queue
            queue_ = std::make_unique<bip::message_queue>(
                bip::open_only, queue_name_.c_str());
        } catch (const bip::interprocess_exception&) {
            // Create new queue
            bip::message_queue::remove(queue_name_.c_str());
            queue_ = std::make_unique<bip::message_queue>(
                bip::create_only, queue_name_.c_str(), 
                max_messages, max_message_size_);
        }
    }
    
    ~MessageQueue() {
        // Note: Don't remove queue in destructor as other processes might be using it
    }
    
    bool send(const Message& message, unsigned int priority = 0) {
        try {
            std::string serialized = message.serialize();
            if (serialized.size() > max_message_size_) {
                std::cerr << "Message too large: " << serialized.size() 
                          << " > " << max_message_size_ << std::endl;
                return false;
            }
            
            queue_->send(serialized.c_str(), serialized.size(), priority);
            return true;
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error sending message: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool try_send(const Message& message, unsigned int priority = 0) {
        try {
            std::string serialized = message.serialize();
            if (serialized.size() > max_message_size_) {
                return false;
            }
            
            return queue_->try_send(serialized.c_str(), serialized.size(), priority);
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error sending message (non-blocking): " << e.what() << std::endl;
            return false;
        }
    }
    
    bool receive(Message& message, unsigned int& priority) {
        try {
            std::vector<char> buffer(max_message_size_);
            size_t received_size;
            
            queue_->receive(buffer.data(), buffer.size(), received_size, priority);
            
            std::string serialized(buffer.data(), received_size);
            message = Message::deserialize(serialized);
            return true;
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error receiving message: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool try_receive(Message& message, unsigned int& priority) {
        try {
            std::vector<char> buffer(max_message_size_);
            size_t received_size;
            
            bool result = queue_->try_receive(buffer.data(), buffer.size(), 
                                            received_size, priority);
            if (result) {
                std::string serialized(buffer.data(), received_size);
                message = Message::deserialize(serialized);
            }
            return result;
        } catch (const bip::interprocess_exception& e) {
            std::cerr << "Error receiving message (non-blocking): " << e.what() << std::endl;
            return false;
        }
    }
    
    size_t get_num_msg() const {
        return queue_->get_num_msg();
    }
    
    static void remove_queue(const std::string& queue_name) {
        bip::message_queue::remove(queue_name.c_str());
    }
    
private:
    std::string queue_name_;
    size_t max_message_size_;
    std::unique_ptr<bip::message_queue> queue_;
};

class MessageBroker {
public:
    MessageBroker(const std::string& broker_name) 
        : broker_name_(broker_name), running_(false) {
        
        // Create broker control queue
        control_queue_ = std::make_unique<MessageQueue>(broker_name_ + "_control");
    }
    
    void start() {
        running_ = true;
        broker_thread_ = std::thread(&MessageBroker::broker_loop, this);
    }
    
    void stop() {
        running_ = false;
        if (broker_thread_.joinable()) {
            broker_thread_.join();
        }
    }
    
    bool subscribe(const std::string& topic, const std::string& subscriber_queue) {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        subscriptions_[topic].insert(subscriber_queue);
        return true;
    }
    
    bool unsubscribe(const std::string& topic, const std::string& subscriber_queue) {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        auto it = subscriptions_.find(topic);
        if (it != subscriptions_.end()) {
            it->second.erase(subscriber_queue);
            if (it->second.empty()) {
                subscriptions_.erase(it);
            }
            return true;
        }
        return false;
    }
    
    bool publish(const std::string& topic, const Message& message) {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        auto it = subscriptions_.find(topic);
        if (it == subscriptions_.end()) {
            return false; // No subscribers
        }
        
        // Send message to all subscribers
        bool all_sent = true;
        for (const auto& subscriber_queue : it->second) {
            try {
                MessageQueue queue(subscriber_queue);
                if (!queue.try_send(message)) {
                    all_sent = false;
                }
            } catch (...) {
                all_sent = false;
            }
        }
        
        return all_sent;
    }
    
private:
    std::string broker_name_;
    std::atomic<bool> running_;
    std::thread broker_thread_;
    std::unique_ptr<MessageQueue> control_queue_;
    
    std::mutex subscriptions_mutex_;
    std::map<std::string, std::set<std::string>> subscriptions_;
    
    void broker_loop() {
        while (running_) {
            Message control_message;
            unsigned int priority;
            
            if (control_queue_->try_receive(control_message, priority)) {
                handle_control_message(control_message);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void handle_control_message(const Message& message) {
        // Handle broker control messages (subscribe, unsubscribe, etc.)
        // Implementation details depend on specific control protocol
    }
};
```

### Step 4: Process Registry and Discovery

```cpp
struct ProcessInfo {
    std::string process_id;
    std::string process_name;
    int pid;
    std::string queue_name;
    std::chrono::system_clock::time_point last_heartbeat;
    std::map<std::string, std::string> properties;
    
    ProcessInfo() = default;
    
    ProcessInfo(const std::string& id, const std::string& name, int process_pid, 
               const std::string& queue)
        : process_id(id), process_name(name), pid(process_pid), queue_name(queue),
          last_heartbeat(std::chrono::system_clock::now()) {}
    
    bool is_alive(std::chrono::seconds timeout = std::chrono::seconds(30)) const {
        auto now = std::chrono::system_clock::now();
        return (now - last_heartbeat) < timeout;
    }
    
private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & process_id;
        ar & process_name;
        ar & pid;
        ar & queue_name;
        ar & last_heartbeat;
        ar & properties;
    }
};

class ProcessRegistry {
public:
    ProcessRegistry(SharedMemoryManager& shm_manager) 
        : shm_manager_(shm_manager) {
        
        // Get or create process registry in shared memory
        registry_map_ = std::make_unique<SharedMap<ShmString, ShmString>>(
            shm_manager_, "process_registry");
    }
    
    bool register_process(const ProcessInfo& info) {
        try {
            std::string serialized = serialize_process_info(info);
            
            auto allocator = shm_manager_.get_allocator();
            ShmString key(info.process_id.c_str(), allocator);
            ShmString value(serialized.c_str(), allocator);
            
            bool result = registry_map_->insert(key, value);
            
            if (result) {
                std::cout << "Registered process: " << info.process_name 
                          << " (ID: " << info.process_id << ")" << std::endl;
            }
            
            return result;
        } catch (const std::exception& e) {
            std::cerr << "Error registering process: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool unregister_process(const std::string& process_id) {
        try {
            auto allocator = shm_manager_.get_allocator();
            ShmString key(process_id.c_str(), allocator);
            
            bool result = registry_map_->erase(key);
            
            if (result) {
                std::cout << "Unregistered process ID: " << process_id << std::endl;
            }
            
            return result;
        } catch (const std::exception& e) {
            std::cerr << "Error unregistering process: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool find_process(const std::string& process_id, ProcessInfo& info) {
        try {
            auto allocator = shm_manager_.get_allocator();
            ShmString key(process_id.c_str(), allocator);
            ShmString value(allocator);
            
            if (registry_map_->find(key, value)) {
                std::string serialized(value.c_str(), value.size());
                info = deserialize_process_info(serialized);
                return true;
            }
            
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Error finding process: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<ProcessInfo> find_processes_by_name(const std::string& process_name) {
        std::vector<ProcessInfo> result;
        
        // This is a simplified implementation
        // In practice, you'd need to iterate through the shared map
        // which requires more complex shared memory container handling
        
        return result;
    }
    
    bool update_heartbeat(const std::string& process_id) {
        ProcessInfo info;
        if (find_process(process_id, info)) {
            info.last_heartbeat = std::chrono::system_clock::now();
            
            // Remove old entry and insert updated one
            unregister_process(process_id);
            return register_process(info);
        }
        return false;
    }
    
    void cleanup_dead_processes(std::chrono::seconds timeout = std::chrono::seconds(60)) {
        // This would require iterating through the shared map
        // Implementation omitted for brevity but would check last_heartbeat
        // and remove processes that haven't sent heartbeats within timeout
    }
    
private:
    SharedMemoryManager& shm_manager_;
    std::unique_ptr<SharedMap<ShmString, ShmString>> registry_map_;
    
    std::string serialize_process_info(const ProcessInfo& info) {
        std::ostringstream oss;
        boost::archive::binary_oarchive oa(oss);
        oa << info;
        return oss.str();
    }
    
    ProcessInfo deserialize_process_info(const std::string& data) {
        std::istringstream iss(data);
        boost::archive::binary_iarchive ia(iss);
        ProcessInfo info;
        ia >> info;
        return info;
    }
};

class ServiceDiscovery {
public:
    ServiceDiscovery(ProcessRegistry& registry) : registry_(registry) {}
    
    bool register_service(const std::string& service_name, const ProcessInfo& provider) {
        std::lock_guard<std::mutex> lock(services_mutex_);
        services_[service_name].push_back(provider);
        return true;
    }
    
    bool unregister_service(const std::string& service_name, const std::string& process_id) {
        std::lock_guard<std::mutex> lock(services_mutex_);
        auto it = services_.find(service_name);
        if (it != services_.end()) {
            auto& providers = it->second;
            providers.erase(
                std::remove_if(providers.begin(), providers.end(),
                    [&process_id](const ProcessInfo& info) {
                        return info.process_id == process_id;
                    }),
                providers.end()
            );
            return true;
        }
        return false;
    }
    
    std::vector<ProcessInfo> discover_service(const std::string& service_name) {
        std::lock_guard<std::mutex> lock(services_mutex_);
        auto it = services_.find(service_name);
        if (it != services_.end()) {
            // Filter out dead processes
            std::vector<ProcessInfo> alive_providers;
            for (const auto& provider : it->second) {
                if (provider.is_alive()) {
                    alive_providers.push_back(provider);
                }
            }
            return alive_providers;
        }
        return {};
    }
    
    ProcessInfo select_service_provider(const std::string& service_name, 
                                       const std::string& selection_strategy = "round_robin") {
        auto providers = discover_service(service_name);
        if (providers.empty()) {
            throw std::runtime_error("No providers available for service: " + service_name);
        }
        
        if (selection_strategy == "round_robin") {
            static std::map<std::string, size_t> round_robin_counters;
            size_t& counter = round_robin_counters[service_name];
            ProcessInfo selected = providers[counter % providers.size()];
            counter++;
            return selected;
        } else if (selection_strategy == "random") {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, providers.size() - 1);
            return providers[dis(gen)];
        }
        
        // Default: return first available
        return providers[0];
    }
    
private:
    ProcessRegistry& registry_;
    std::mutex services_mutex_;
    std::map<std::string, std::vector<ProcessInfo>> services_;
};
```

### Step 5: High-Level IPC Manager

```cpp
class IPCManager {
public:
    IPCManager(const std::string& process_name, const std::string& shared_memory_name = "ipc_system")
        : process_name_(process_name), 
          process_id_(generate_process_id()),
          shm_manager_(shared_memory_name, 1024 * 1024 * 10), // 10MB shared memory
          registry_(shm_manager_),
          service_discovery_(registry_),
          running_(false) {
        
        // Create process-specific message queue
        queue_name_ = "queue_" + process_id_;
        message_queue_ = std::make_unique<MessageQueue>(queue_name_);
        
        // Register this process
        ProcessInfo info(process_id_, process_name_, getpid(), queue_name_);
        registry_.register_process(info);
        
        std::cout << "IPC Manager initialized for process: " << process_name_ 
                  << " (ID: " << process_id_ << ")" << std::endl;
    }
    
    ~IPCManager() {
        stop();
        registry_.unregister_process(process_id_);
    }
    
    void start() {
        if (running_) return;
        
        running_ = true;
        
        // Start message processing thread
        message_thread_ = std::thread(&IPCManager::message_processing_loop, this);
        
        // Start heartbeat thread
        heartbeat_thread_ = std::thread(&IPCManager::heartbeat_loop, this);
        
        std::cout << "IPC Manager started" << std::endl;
    }
    
    void stop() {
        if (!running_) return;
        
        running_ = false;
        
        if (message_thread_.joinable()) {
            message_thread_.join();
        }
        
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
        
        std::cout << "IPC Manager stopped" << std::endl;
    }
    
    // Message sending methods
    bool send_message(const std::string& target_process_id, const Message& message) {
        ProcessInfo target_info;
        if (!registry_.find_process(target_process_id, target_info)) {
            std::cerr << "Target process not found: " << target_process_id << std::endl;
            return false;
        }
        
        try {
            MessageQueue target_queue(target_info.queue_name);
            return target_queue.try_send(message);
        } catch (const std::exception& e) {
            std::cerr << "Error sending message to " << target_process_id 
                      << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    std::string send_request(const std::string& target_process_id, const std::string& request_data,
                           std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        Message request(Message::REQUEST, process_id_, request_data);
        request.set_reply_to(queue_name_);
        
        std::string correlation_id = generate_correlation_id();
        request.set_correlation_id(correlation_id);
        
        if (!send_message(target_process_id, request)) {
            throw std::runtime_error("Failed to send request");
        }
        
        // Wait for response
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (std::chrono::steady_clock::now() < deadline) {
            Message response;
            unsigned int priority;
            
            if (message_queue_->try_receive(response, priority)) {
                if (response.type() == Message::RESPONSE && 
                    response.correlation_id() == correlation_id) {
                    return response.data();
                }
                
                // Put message back in processing queue if it's not our response
                pending_messages_.push(response);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        throw std::runtime_error("Request timeout");
    }
    
    bool broadcast_message(const Message& message) {
        // Implementation would get all registered processes and send to each
        // Simplified version shown here
        return true;
    }
    
    // Service registration
    bool register_service(const std::string& service_name) {
        ProcessInfo info(process_id_, process_name_, getpid(), queue_name_);
        return service_discovery_.register_service(service_name, info);
    }
    
    bool unregister_service(const std::string& service_name) {
        return service_discovery_.unregister_service(service_name, process_id_);
    }
    
    std::string call_service(const std::string& service_name, const std::string& request_data,
                           std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        auto provider = service_discovery_.select_service_provider(service_name);
        return send_request(provider.process_id, request_data, timeout);
    }
    
    // Message handlers
    void set_request_handler(std::function<std::string(const std::string&)> handler) {
        request_handler_ = handler;
    }
    
    void set_notification_handler(std::function<void(const Message&)> handler) {
        notification_handler_ = handler;
    }
    
    // Shared memory access
    template<typename T>
    T* get_shared_object(const std::string& name) {
        return shm_manager_.find_object<T>(name);
    }
    
    template<typename T>
    T* create_shared_object(const std::string& name) {
        return shm_manager_.construct_object<T>(name);
    }
    
    template<typename T, typename... Args>
    T* create_shared_object(const std::string& name, Args&&... args) {
        return shm_manager_.construct_object<T>(name, std::forward<Args>(args)...);
    }
    
    // Statistics
    void print_statistics() {
        shm_manager_.print_memory_stats();
        std::cout << "Process ID: " << process_id_ << std::endl;
        std::cout << "Process Name: " << process_name_ << std::endl;
        std::cout << "Message Queue: " << queue_name_ << std::endl;
        std::cout << "Messages in queue: " << message_queue_->get_num_msg() << std::endl;
    }
    
private:
    std::string process_name_;
    std::string process_id_;
    std::string queue_name_;
    
    SharedMemoryManager shm_manager_;
    ProcessRegistry registry_;
    ServiceDiscovery service_discovery_;
    std::unique_ptr<MessageQueue> message_queue_;
    
    std::atomic<bool> running_;
    std::thread message_thread_;
    std::thread heartbeat_thread_;
    
    std::queue<Message> pending_messages_;
    std::mutex pending_messages_mutex_;
    
    std::function<std::string(const std::string&)> request_handler_;
    std::function<void(const Message&)> notification_handler_;
    
    std::string generate_process_id() {
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        return boost::uuids::to_string(uuid);
    }
    
    std::string generate_correlation_id() {
        boost::uuids::uuid uuid = boost::uuids::random_generator()();
        return boost::uuids::to_string(uuid);
    }
    
    void message_processing_loop() {
        while (running_) {
            Message message;
            unsigned int priority;
            
            // Check pending messages first
            {
                std::lock_guard<std::mutex> lock(pending_messages_mutex_);
                if (!pending_messages_.empty()) {
                    message = pending_messages_.front();
                    pending_messages_.pop();
                    process_message(message);
                    continue;
                }
            }
            
            // Check for new messages
            if (message_queue_->try_receive(message, priority)) {
                process_message(message);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }
    
    void process_message(const Message& message) {
        switch (message.type()) {
            case Message::REQUEST:
                handle_request(message);
                break;
            case Message::RESPONSE:
                // Responses are handled in send_request method
                break;
            case Message::NOTIFICATION:
                handle_notification(message);
                break;
            case Message::HEARTBEAT:
                // Heartbeats are processed automatically
                break;
            case Message::SHUTDOWN:
                handle_shutdown(message);
                break;
        }
    }
    
    void handle_request(const Message& request) {
        if (request_handler_) {
            try {
                std::string response_data = request_handler_(request.data());
                
                Message response(Message::RESPONSE, process_id_, response_data);
                response.set_correlation_id(request.correlation_id());
                
                // Send response back to requester
                if (!request.reply_to().empty()) {
                    try {
                        MessageQueue reply_queue(request.reply_to());
                        reply_queue.try_send(response);
                    } catch (const std::exception& e) {
                        std::cerr << "Error sending response: " << e.what() << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing request: " << e.what() << std::endl;
            }
        }
    }
    
    void handle_notification(const Message& notification) {
        if (notification_handler_) {
            notification_handler_(notification);
        }
    }
    
    void handle_shutdown(const Message& shutdown_message) {
        std::cout << "Shutdown message received" << std::endl;
        stop();
    }
    
    void heartbeat_loop() {
        while (running_) {
            registry_.update_heartbeat(process_id_);
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    
    int getpid() {
#ifdef _WIN32
        return _getpid();
#else
        return ::getpid();
#endif
    }
};
```

### Step 6: Example Applications

```cpp
// Example 1: Simple Request/Response Server
class CalculatorServer {
public:
    CalculatorServer() : ipc_("CalculatorServer") {
        ipc_.set_request_handler([this](const std::string& request) {
            return process_calculation(request);
        });
        
        ipc_.register_service("calculator");
        ipc_.start();
        
        std::cout << "Calculator server started" << std::endl;
    }
    
    void run() {
        std::cout << "Calculator server running. Press 'q' to quit." << std::endl;
        char input;
        while (std::cin >> input && input != 'q') {
            ipc_.print_statistics();
        }
    }
    
private:
    IPCManager ipc_;
    
    std::string process_calculation(const std::string& expression) {
        try {
            // Simple expression parser (just for demonstration)
            std::istringstream iss(expression);
            double a, b;
            char op;
            
            if (iss >> a >> op >> b) {
                double result;
                switch (op) {
                    case '+': result = a + b; break;
                    case '-': result = a - b; break;
                    case '*': result = a * b; break;
                    case '/': 
                        if (b == 0) throw std::runtime_error("Division by zero");
                        result = a / b; 
                        break;
                    default: throw std::runtime_error("Unknown operator");
                }
                return std::to_string(result);
            } else {
                throw std::runtime_error("Invalid expression format");
            }
        } catch (const std::exception& e) {
            return "ERROR: " + std::string(e.what());
        }
    }
};

// Example 2: Client Application
class CalculatorClient {
public:
    CalculatorClient() : ipc_("CalculatorClient") {
        ipc_.start();
        std::cout << "Calculator client started" << std::endl;
    }
    
    void run() {
        std::cout << "Enter expressions (e.g., '3.5 + 2.1') or 'quit' to exit:" << std::endl;
        
        std::string line;
        while (std::getline(std::cin, line) && line != "quit") {
            if (line.empty()) continue;
            
            try {
                std::string result = ipc_.call_service("calculator", line);
                std::cout << "Result: " << result << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
    }
    
private:
    IPCManager ipc_;
};

// Example 3: Shared Data Manager
class SharedDataManager {
public:
    SharedDataManager() : ipc_("SharedDataManager") {
        // Create shared data structures
        shared_counter_ = ipc_.create_shared_object<std::atomic<int>>("global_counter");
        if (shared_counter_) {
            shared_counter_->store(0);
        }
        
        shared_queue_ = std::make_unique<SharedQueue<int>>(
            ipc_.shm_manager_, "shared_int_queue", 1000);
        
        ipc_.set_request_handler([this](const std::string& request) {
            return handle_data_request(request);
        });
        
        ipc_.register_service("data_manager");
        ipc_.start();
    }
    
    void run() {
        std::cout << "Shared Data Manager running. Commands:" << std::endl;
        std::cout << "  inc - increment counter" << std::endl;
        std::cout << "  get - get counter value" << std::endl;
        std::cout << "  push <num> - push to queue" << std::endl;
        std::cout << "  pop - pop from queue" << std::endl;
        std::cout << "  quit - exit" << std::endl;
        
        std::string command;
        while (std::cin >> command && command != "quit") {
            if (command == "inc") {
                if (shared_counter_) {
                    int new_value = ++(*shared_counter_);
                    std::cout << "Counter incremented to: " << new_value << std::endl;
                }
            } else if (command == "get") {
                if (shared_counter_) {
                    std::cout << "Counter value: " << shared_counter_->load() << std::endl;
                }
            } else if (command == "push") {
                int value;
                if (std::cin >> value) {
                    if (shared_queue_->push(value)) {
                        std::cout << "Pushed " << value << " to queue" << std::endl;
                    } else {
                        std::cout << "Queue is full" << std::endl;
                    }
                }
            } else if (command == "pop") {
                int value;
                if (shared_queue_->pop(value)) {
                    std::cout << "Popped " << value << " from queue" << std::endl;
                } else {
                    std::cout << "Queue is empty" << std::endl;
                }
            }
        }
    }
    
private:
    IPCManager ipc_;
    std::atomic<int>* shared_counter_;
    std::unique_ptr<SharedQueue<int>> shared_queue_;
    
    std::string handle_data_request(const std::string& request) {
        // Handle remote data requests
        if (request == "get_counter") {
            return std::to_string(shared_counter_->load());
        } else if (request.substr(0, 4) == "inc_") {
            int increment = std::stoi(request.substr(4));
            return std::to_string(shared_counter_->fetch_add(increment) + increment);
        }
        return "Unknown request";
    }
};
```

### Step 7: Main Application Framework

```cpp
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <server|client|data_manager>" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "server") {
            CalculatorServer server;
            server.run();
        } else if (mode == "client") {
            CalculatorClient client;
            client.run();
        } else if (mode == "data_manager") {
            SharedDataManager manager;
            manager.run();
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Advanced Features

### Performance Monitoring

```cpp
class IPCMonitor {
public:
    struct Metrics {
        std::atomic<uint64_t> messages_sent{0};
        std::atomic<uint64_t> messages_received{0};
        std::atomic<uint64_t> bytes_transmitted{0};
        std::atomic<uint64_t> errors{0};
        std::chrono::steady_clock::time_point start_time;
        
        Metrics() : start_time(std::chrono::steady_clock::now()) {}
    };
    
    void record_message_sent(size_t bytes) {
        metrics_.messages_sent++;
        metrics_.bytes_transmitted += bytes;
    }
    
    void record_message_received(size_t bytes) {
        metrics_.messages_received++;
        metrics_.bytes_transmitted += bytes;
    }
    
    void record_error() {
        metrics_.errors++;
    }
    
    std::string get_metrics_report() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            now - metrics_.start_time);
        
        std::ostringstream report;
        report << "IPC Metrics Report:\n";
        report << "  Uptime: " << duration.count() << " seconds\n";
        report << "  Messages sent: " << metrics_.messages_sent.load() << "\n";
        report << "  Messages received: " << metrics_.messages_received.load() << "\n";
        report << "  Total bytes: " << metrics_.bytes_transmitted.load() << "\n";
        report << "  Errors: " << metrics_.errors.load() << "\n";
        
        if (duration.count() > 0) {
            report << "  Messages/sec: " << 
                (metrics_.messages_sent.load() + metrics_.messages_received.load()) / duration.count() << "\n";
            report << "  Bytes/sec: " << metrics_.bytes_transmitted.load() / duration.count() << "\n";
        }
        
        return report.str();
    }
    
private:
    Metrics metrics_;
};
```

## Testing Framework

### Integration Tests

```cpp
#define BOOST_TEST_MODULE IPCSystemTests
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(IPCTests)

BOOST_AUTO_TEST_CASE(TestBasicMessagePassing) {
    const std::string test_queue = "test_message_queue";
    
    // Cleanup any existing queue
    MessageQueue::remove_queue(test_queue);
    
    {
        MessageQueue sender(test_queue);
        Message msg(Message::NOTIFICATION, "test_sender", "Hello, World!");
        
        BOOST_CHECK(sender.send(msg));
    }
    
    {
        MessageQueue receiver(test_queue);
        Message received_msg;
        unsigned int priority;
        
        BOOST_CHECK(receiver.receive(received_msg, priority));
        BOOST_CHECK_EQUAL(received_msg.data(), "Hello, World!");
        BOOST_CHECK_EQUAL(received_msg.sender(), "test_sender");
    }
    
    MessageQueue::remove_queue(test_queue);
}

BOOST_AUTO_TEST_CASE(TestSharedMemoryOperations) {
    const std::string segment_name = "test_segment";
    
    SharedMemoryManager::remove_segment(segment_name);
    
    {
        SharedMemoryManager shm(segment_name, 1024 * 1024);
        
        // Test object construction
        int* shared_int = shm.construct_object<int>("test_int");
        BOOST_REQUIRE(shared_int != nullptr);
        *shared_int = 42;
        
        // Test object finding
        int* found_int = shm.find_object<int>("test_int");
        BOOST_REQUIRE(found_int != nullptr);
        BOOST_CHECK_EQUAL(*found_int, 42);
    }
    
    SharedMemoryManager::remove_segment(segment_name);
}

BOOST_AUTO_TEST_SUITE_END()
```

### Performance Benchmarks

```cpp
class IPCBenchmark {
public:
    struct BenchmarkResults {
        double messages_per_second;
        double bytes_per_second;
        std::chrono::milliseconds avg_latency;
        std::chrono::milliseconds max_latency;
    };
    
    BenchmarkResults benchmark_message_throughput(size_t num_messages, size_t message_size) {
        const std::string queue_name = "benchmark_queue";
        MessageQueue::remove_queue(queue_name);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Producer
        std::thread producer([&]() {
            MessageQueue queue(queue_name);
            std::string data(message_size, 'A');
            
            for (size_t i = 0; i < num_messages; ++i) {
                Message msg(Message::NOTIFICATION, "benchmark", data);
                queue.send(msg);
            }
        });
        
        // Consumer
        std::thread consumer([&]() {
            MessageQueue queue(queue_name);
            
            for (size_t i = 0; i < num_messages; ++i) {
                Message msg;
                unsigned int priority;
                queue.receive(msg, priority);
            }
        });
        
        producer.join();
        consumer.join();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BenchmarkResults results;
        results.messages_per_second = static_cast<double>(num_messages) / (duration.count() / 1000.0);
        results.bytes_per_second = static_cast<double>(num_messages * message_size) / (duration.count() / 1000.0);
        results.avg_latency = std::chrono::milliseconds(duration.count() / num_messages);
        results.max_latency = duration; // Simplified
        
        MessageQueue::remove_queue(queue_name);
        return results;
    }
};
```

## Deployment Guide

### Build Configuration

```cmake
cmake_minimum_required(VERSION 3.10)
project(IPCSystem)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS 
    system serialization thread unit_test_framework)

# Main executable
add_executable(ipc_demo
    src/main.cpp
    src/ipc_manager.cpp
    src/shared_memory_manager.cpp
    src/message_queue.cpp
    src/process_registry.cpp
)

target_link_libraries(ipc_demo 
    ${Boost_LIBRARIES}
    pthread
)

# Test executable
add_executable(ipc_tests
    tests/test_main.cpp
    tests/ipc_tests.cpp
    src/ipc_manager.cpp
    src/shared_memory_manager.cpp
)

target_link_libraries(ipc_tests 
    ${Boost_LIBRARIES}
    pthread
)

enable_testing()
add_test(NAME IPCTests COMMAND ipc_tests)
```

### Usage Examples

```bash
# Start the data manager
./ipc_demo data_manager &

# Start calculator server
./ipc_demo server &

# Run client
./ipc_demo client

# Run performance tests
./ipc_benchmark --messages 10000 --size 1024
```

## Assessment Criteria

- [ ] Implements comprehensive IPC mechanisms using Boost.Interprocess
- [ ] Demonstrates shared memory management with proper synchronization
- [ ] Provides reliable message passing with multiple communication patterns
- [ ] Includes process discovery and service registration
- [ ] Handles cross-platform compatibility considerations
- [ ] Implements fault tolerance and error recovery
- [ ] Provides performance monitoring and optimization
- [ ] Includes comprehensive testing suite
- [ ] Achieves performance benchmarks for throughput and latency
- [ ] Includes complete documentation and usage examples

## Deliverables

1. Complete IPC system implementation with all core features
2. Comprehensive test suite including unit and integration tests
3. Performance benchmarking framework and results
4. Cross-platform deployment guide
5. API documentation and usage examples
6. Performance tuning guide
7. Fault tolerance and recovery procedures documentation
