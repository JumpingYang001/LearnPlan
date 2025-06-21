# STL Projects

*Part of STL Learning Track - Practical Applications*

## Project 1: Generic Data Structure Library

### Description
Create a comprehensive library of generic data structures that are fully compatible with STL algorithms and conventions.

### Requirements
- Implement at least 5 different data structures
- Full STL compatibility (iterators, algorithms, etc.)
- Exception safety guarantees
- Performance benchmarks against STL equivalents
- Comprehensive unit tests

### Project Structure
```
STLDataStructures/
├── include/
│   ├── circular_buffer.hpp
│   ├── trie.hpp
│   ├── disjoint_set.hpp
│   ├── segment_tree.hpp
│   └── skip_list.hpp
├── src/
│   └── implementations/
├── tests/
│   ├── test_circular_buffer.cpp
│   ├── test_trie.cpp
│   └── benchmarks/
├── examples/
└── CMakeLists.txt
```

### Implementation Example: Circular Buffer
```cpp
#include <iterator>
#include <memory>
#include <stdexcept>
#include <algorithm>

template<typename T, typename Allocator = std::allocator<T>>
class CircularBuffer {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<Allocator>::pointer;
    using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
    
    class iterator {
    private:
        CircularBuffer* buffer_;
        size_type index_;
        
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        
        iterator(CircularBuffer* buffer, size_type index) 
            : buffer_(buffer), index_(index) {}
        
        reference operator*() const {
            return buffer_->data_[(buffer_->head_ + index_) % buffer_->capacity_];
        }
        
        pointer operator->() const {
            return &(operator*());
        }
        
        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++index_; return tmp; }
        iterator& operator--() { --index_; return *this; }
        iterator operator--(int) { iterator tmp = *this; --index_; return tmp; }
        
        iterator operator+(difference_type n) const {
            return iterator(buffer_, index_ + n);
        }
        
        iterator operator-(difference_type n) const {
            return iterator(buffer_, index_ - n);
        }
        
        iterator& operator+=(difference_type n) { index_ += n; return *this; }
        iterator& operator-=(difference_type n) { index_ -= n; return *this; }
        
        difference_type operator-(const iterator& other) const {
            return index_ - other.index_;
        }
        
        reference operator[](difference_type n) const {
            return buffer_->data_[(buffer_->head_ + index_ + n) % buffer_->capacity_];
        }
        
        bool operator==(const iterator& other) const { return index_ == other.index_; }
        bool operator!=(const iterator& other) const { return index_ != other.index_; }
        bool operator<(const iterator& other) const { return index_ < other.index_; }
        bool operator>(const iterator& other) const { return index_ > other.index_; }
        bool operator<=(const iterator& other) const { return index_ <= other.index_; }
        bool operator>=(const iterator& other) const { return index_ >= other.index_; }
    };
    
    using const_iterator = iterator; // Simplified for example
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    
private:
    allocator_type alloc_;
    pointer data_;
    size_type capacity_;
    size_type size_;
    size_type head_;
    
public:
    explicit CircularBuffer(size_type capacity, const Allocator& alloc = Allocator())
        : alloc_(alloc), capacity_(capacity), size_(0), head_(0) {
        data_ = std::allocator_traits<Allocator>::allocate(alloc_, capacity_);
    }
    
    ~CircularBuffer() {
        clear();
        std::allocator_traits<Allocator>::deallocate(alloc_, data_, capacity_);
    }
    
    // Element access
    reference operator[](size_type pos) {
        return data_[(head_ + pos) % capacity_];
    }
    
    const_reference operator[](size_type pos) const {
        return data_[(head_ + pos) % capacity_];
    }
    
    reference at(size_type pos) {
        if (pos >= size_) throw std::out_of_range("CircularBuffer::at");
        return (*this)[pos];
    }
    
    const_reference at(size_type pos) const {
        if (pos >= size_) throw std::out_of_range("CircularBuffer::at");
        return (*this)[pos];
    }
    
    reference front() { return (*this)[0]; }
    const_reference front() const { return (*this)[0]; }
    
    reference back() { return (*this)[size_ - 1]; }
    const_reference back() const { return (*this)[size_ - 1]; }
    
    // Iterators
    iterator begin() { return iterator(this, 0); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator cbegin() const { return begin(); }
    
    iterator end() { return iterator(this, size_); }
    const_iterator end() const { return const_iterator(this, size_); }
    const_iterator cend() const { return end(); }
    
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    
    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type max_size() const noexcept { return capacity_; }
    size_type capacity() const noexcept { return capacity_; }
    bool full() const noexcept { return size_ == capacity_; }
    
    // Modifiers
    void clear() noexcept {
        while (!empty()) {
            pop_back();
        }
    }
    
    void push_back(const T& value) {
        if (full()) {
            pop_front(); // Remove oldest element
        }
        
        size_type pos = (head_ + size_) % capacity_;
        std::allocator_traits<Allocator>::construct(alloc_, &data_[pos], value);
        if (size_ < capacity_) {
            ++size_;
        }
    }
    
    void push_back(T&& value) {
        if (full()) {
            pop_front(); // Remove oldest element
        }
        
        size_type pos = (head_ + size_) % capacity_;
        std::allocator_traits<Allocator>::construct(alloc_, &data_[pos], std::move(value));
        if (size_ < capacity_) {
            ++size_;
        }
    }
    
    template<typename... Args>
    reference emplace_back(Args&&... args) {
        if (full()) {
            pop_front(); // Remove oldest element
        }
        
        size_type pos = (head_ + size_) % capacity_;
        std::allocator_traits<Allocator>::construct(alloc_, &data_[pos], std::forward<Args>(args)...);
        if (size_ < capacity_) {
            ++size_;
        }
        return data_[pos];
    }
    
    void pop_front() {
        if (!empty()) {
            std::allocator_traits<Allocator>::destroy(alloc_, &data_[head_]);
            head_ = (head_ + 1) % capacity_;
            --size_;
        }
    }
    
    void pop_back() {
        if (!empty()) {
            size_type pos = (head_ + size_ - 1) % capacity_;
            std::allocator_traits<Allocator>::destroy(alloc_, &data_[pos]);
            --size_;
        }
    }
};

// Usage example
void circular_buffer_example() {
    CircularBuffer<int> buffer(5);
    
    // Fill buffer
    for (int i = 1; i <= 7; ++i) {
        buffer.push_back(i);
    }
    
    std::cout << "Buffer contents: ";
    std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    
    // Use STL algorithms
    auto it = std::find(buffer.begin(), buffer.end(), 5);
    if (it != buffer.end()) {
        std::cout << "Found 5 at position: " << std::distance(buffer.begin(), it) << std::endl;
    }
    
    std::sort(buffer.begin(), buffer.end());
    std::cout << "Sorted: ";
    std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}
```

## Project 2: Advanced Algorithm Library

### Description
Implement a collection of advanced algorithms not found in the standard library, with full STL compatibility.

### Algorithms to Implement
1. **Pattern Matching**: KMP, Boyer-Moore, Rabin-Karp
2. **Graph Algorithms**: Dijkstra, A*, Topological sort
3. **String Algorithms**: Suffix array, LCS, Edit distance
4. **Numerical Algorithms**: FFT, Matrix operations
5. **Parallel Algorithms**: Parallel sort, reduce, scan

### Implementation Example: KMP String Search
```cpp
#include <vector>
#include <iterator>
#include <algorithm>

template<typename ForwardIterator1, typename ForwardIterator2>
std::vector<size_t> compute_lps(ForwardIterator1 pattern_begin, ForwardIterator1 pattern_end) {
    size_t pattern_length = std::distance(pattern_begin, pattern_end);
    std::vector<size_t> lps(pattern_length, 0);
    
    size_t len = 0;
    size_t i = 1;
    
    auto pattern_it = pattern_begin;
    std::advance(pattern_it, 1);
    
    while (i < pattern_length) {
        auto current_it = pattern_begin;
        std::advance(current_it, len);
        
        auto pattern_i_it = pattern_begin;
        std::advance(pattern_i_it, i);
        
        if (*pattern_i_it == *current_it) {
            ++len;
            lps[i] = len;
            ++i;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                ++i;
            }
        }
    }
    
    return lps;
}

template<typename ForwardIterator1, typename ForwardIterator2>
std::vector<size_t> kmp_search(ForwardIterator1 text_begin, ForwardIterator1 text_end,
                               ForwardIterator2 pattern_begin, ForwardIterator2 pattern_end) {
    std::vector<size_t> matches;
    
    size_t text_length = std::distance(text_begin, text_end);
    size_t pattern_length = std::distance(pattern_begin, pattern_end);
    
    if (pattern_length == 0) return matches;
    if (text_length < pattern_length) return matches;
    
    auto lps = compute_lps(pattern_begin, pattern_end);
    
    size_t i = 0; // index for text
    size_t j = 0; // index for pattern
    
    auto text_it = text_begin;
    auto pattern_it = pattern_begin;
    
    while (i < text_length) {
        if (*text_it == *pattern_it) {
            ++text_it;
            ++pattern_it;
            ++i;
            ++j;
        }
        
        if (j == pattern_length) {
            matches.push_back(i - j);
            j = lps[j - 1];
            pattern_it = pattern_begin;
            std::advance(pattern_it, j);
        } else if (i < text_length && *text_it != *pattern_it) {
            if (j != 0) {
                j = lps[j - 1];
                pattern_it = pattern_begin;
                std::advance(pattern_it, j);
            } else {
                ++text_it;
                ++i;
            }
        }
    }
    
    return matches;
}

// Usage example
void kmp_example() {
    std::string text = "ABABDABACDABABCABCABCABCABC";
    std::string pattern = "ABABCABCABC";
    
    auto matches = kmp_search(text.begin(), text.end(), pattern.begin(), pattern.end());
    
    std::cout << "Pattern found at positions: ";
    for (size_t pos : matches) {
        std::cout << pos << " ";
    }
    std::cout << std::endl;
}
```

## Project 3: Memory Pool Allocator

### Description
Create a high-performance memory pool allocator system that integrates seamlessly with STL containers.

### Features
- Fixed-size block allocator
- Variable-size allocator with best-fit strategy
- Thread-safe allocator wrapper
- Memory debugging capabilities
- Statistics and profiling

### Implementation Example
```cpp
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <cassert>

template<typename T, size_t BlockSize = 4096>
class MemoryPool {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
    };
    
    struct Chunk {
        alignas(Block) char data[BlockSize];
        Chunk* next;
        
        Chunk() : next(nullptr) {}
    };
    
    Chunk* chunks_;
    Block* free_blocks_;
    std::atomic<size_t> allocated_count_;
    std::atomic<size_t> deallocated_count_;
    mutable std::mutex mutex_;
    
    void allocate_chunk() {
        Chunk* new_chunk = reinterpret_cast<Chunk*>(std::aligned_alloc(alignof(Chunk), sizeof(Chunk)));
        if (!new_chunk) {
            throw std::bad_alloc();
        }
        
        new(new_chunk) Chunk();
        new_chunk->next = chunks_;
        chunks_ = new_chunk;
        
        // Link all blocks in the chunk
        const size_t blocks_per_chunk = BlockSize / sizeof(Block);
        Block* block = reinterpret_cast<Block*>(new_chunk->data);
        
        for (size_t i = 0; i < blocks_per_chunk - 1; ++i) {
            block[i].next = &block[i + 1];
        }
        block[blocks_per_chunk - 1].next = free_blocks_;
        free_blocks_ = block;
    }
    
public:
    MemoryPool() : chunks_(nullptr), free_blocks_(nullptr), 
                   allocated_count_(0), deallocated_count_(0) {
        allocate_chunk();
    }
    
    ~MemoryPool() {
        // Clean up all chunks
        while (chunks_) {
            Chunk* next = chunks_->next;
            std::free(chunks_);
            chunks_ = next;
        }
    }
    
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!free_blocks_) {
            allocate_chunk();
        }
        
        Block* block = free_blocks_;
        free_blocks_ = free_blocks_->next;
        
        ++allocated_count_;
        return reinterpret_cast<T*>(block);
    }
    
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = free_blocks_;
        free_blocks_ = block;
        
        ++deallocated_count_;
    }
    
    size_t allocated_count() const { return allocated_count_.load(); }
    size_t deallocated_count() const { return deallocated_count_.load(); }
    size_t active_allocations() const { return allocated_count() - deallocated_count(); }
};

template<typename T>
class PoolAllocator {
private:
    static MemoryPool<T>& get_pool() {
        static MemoryPool<T> pool;
        return pool;
    }
    
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };
    
    PoolAllocator() = default;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>&) {}
    
    pointer allocate(size_type n) {
        if (n != 1) {
            // Fall back to standard allocator for non-single allocations
            return static_cast<pointer>(std::malloc(n * sizeof(T)));
        }
        return get_pool().allocate();
    }
    
    void deallocate(pointer p, size_type n) {
        if (n != 1) {
            std::free(p);
            return;
        }
        get_pool().deallocate(p);
    }
    
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }
    
    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
};

template<typename T, typename U>
bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) {
    return true;
}

template<typename T, typename U>
bool operator!=(const PoolAllocator<T>&, const PoolAllocator<U>&) {
    return false;
}

// Usage example
void memory_pool_example() {
    using PoolVector = std::vector<int, PoolAllocator<int>>;
    
    PoolVector vec;
    vec.reserve(1000);
    
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }
    
    std::cout << "Vector size: " << vec.size() << std::endl;
    std::cout << "Pool statistics:" << std::endl;
    // Note: Getting pool statistics would require exposing the pool instance
}
```

## Project 4: Functional Programming Library

### Description
Create a functional programming library that extends STL with functional programming concepts while maintaining STL compatibility.

### Features
- Lazy evaluation sequences
- Function composition utilities
- Monadic operations (Optional, Either)
- Immutable data structures
- Functional algorithms

### Implementation Example: Lazy Sequence
```cpp
#include <functional>
#include <optional>
#include <iterator>
#include <memory>

template<typename T>
class LazySequence {
private:
    std::function<std::optional<T>()> generator_;
    
public:
    class iterator {
    private:
        std::function<std::optional<T>()> generator_;
        std::optional<T> current_;
        bool is_end_;
        
        void advance() {
            current_ = generator_();
            is_end_ = !current_.has_value();
        }
        
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;
        
        iterator(std::function<std::optional<T>()> gen, bool is_end = false) 
            : generator_(gen), is_end_(is_end) {
            if (!is_end_) {
                advance();
            }
        }
        
        reference operator*() const {
            return current_.value();
        }
        
        pointer operator->() const {
            return &current_.value();
        }
        
        iterator& operator++() {
            advance();
            return *this;
        }
        
        iterator operator++(int) {
            iterator tmp = *this;
            advance();
            return tmp;
        }
        
        bool operator==(const iterator& other) const {
            return is_end_ == other.is_end_;
        }
        
        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };
    
    LazySequence(std::function<std::optional<T>()> gen) : generator_(gen) {}
    
    iterator begin() {
        return iterator(generator_);
    }
    
    iterator end() {
        return iterator(generator_, true);
    }
    
    template<typename Func>
    auto map(Func f) -> LazySequence<decltype(f(std::declval<T>()))> {
        using U = decltype(f(std::declval<T>()));
        
        return LazySequence<U>([gen = generator_, f]() -> std::optional<U> {
            auto value = gen();
            if (value) {
                return f(*value);
            }
            return std::nullopt;
        });
    }
    
    template<typename Predicate>
    LazySequence<T> filter(Predicate pred) {
        return LazySequence<T>([gen = generator_, pred]() -> std::optional<T> {
            while (true) {
                auto value = gen();
                if (!value) return std::nullopt;
                if (pred(*value)) return value;
            }
        });
    }
    
    LazySequence<T> take(size_t n) {
        auto counter = std::make_shared<size_t>(0);
        
        return LazySequence<T>([gen = generator_, counter, n]() -> std::optional<T> {
            if (*counter >= n) return std::nullopt;
            ++(*counter);
            return gen();
        });
    }
    
    std::vector<T> to_vector() {
        std::vector<T> result;
        for (const auto& item : *this) {
            result.push_back(item);
        }
        return result;
    }
};

// Factory functions
template<typename T>
LazySequence<T> range(T start, T end, T step = T(1)) {
    auto current = std::make_shared<T>(start);
    
    return LazySequence<T>([current, end, step]() -> std::optional<T> {
        if (*current >= end) return std::nullopt;
        T value = *current;
        *current += step;
        return value;
    });
}

template<typename T>
LazySequence<T> repeat(T value, size_t count) {
    auto counter = std::make_shared<size_t>(0);
    
    return LazySequence<T>([value, counter, count]() -> std::optional<T> {
        if (*counter >= count) return std::nullopt;
        ++(*counter);
        return value;
    });
}

// Usage example
void lazy_sequence_example() {
    auto numbers = range(1, 100)
        .filter([](int n) { return n % 2 == 0; })
        .map([](int n) { return n * n; })
        .take(5);
    
    std::cout << "First 5 squares of even numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Convert to vector
    auto vec = range(1, 10).to_vector();
    std::cout << "Range as vector: ";
    for (int n : vec) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
}
```

## Project 5: Concurrent Data Structures

### Description
Implement thread-safe, lock-free data structures that maintain STL compatibility where possible.

### Data Structures
1. Lock-free queue
2. Thread-safe hash table
3. Concurrent vector
4. Lock-free stack
5. Thread-safe priority queue

### Implementation Example: Lock-Free Queue
```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node;
        head_.store(dummy);
        tail_.store(dummy);
    }
    
    ~LockFreeQueue() {
        while (Node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }
    
    void enqueue(T item) {
        Node* new_node = new Node;
        T* data = new T(std::move(item));
        new_node->data.store(data);
        
        while (true) {
            Node* last = tail_.load();
            Node* next = last->next.load();
            
            if (last == tail_.load()) {
                if (next == nullptr) {
                    if (last->next.compare_exchange_weak(next, new_node)) {
                        tail_.compare_exchange_weak(last, new_node);
                        break;
                    }
                } else {
                    tail_.compare_exchange_weak(last, next);
                }
            }
        }
    }
    
    bool dequeue(T& result) {
        while (true) {
            Node* first = head_.load();
            Node* last = tail_.load();
            Node* next = first->next.load();
            
            if (first == head_.load()) {
                if (first == last) {
                    if (next == nullptr) {
                        return false; // Queue is empty
                    }
                    tail_.compare_exchange_weak(last, next);
                } else {
                    if (next == nullptr) {
                        continue;
                    }
                    
                    T* data = next->data.load();
                    if (data == nullptr) {
                        continue;
                    }
                    
                    if (head_.compare_exchange_weak(first, next)) {
                        result = *data;
                        delete data;
                        delete first;
                        return true;
                    }
                }
            }
        }
    }
    
    bool empty() const {
        Node* first = head_.load();
        Node* last = tail_.load();
        return (first == last) && (first->next.load() == nullptr);
    }
};

// Usage example
void lock_free_queue_example() {
    LockFreeQueue<int> queue;
    
    // Producer thread simulation
    for (int i = 0; i < 10; ++i) {
        queue.enqueue(i);
    }
    
    // Consumer thread simulation
    int value;
    while (queue.dequeue(value)) {
        std::cout << "Dequeued: " << value << std::endl;
    }
}
```

## Project Integration and Testing

### CMake Configuration
```cmake
cmake_minimum_required(VERSION 3.15)
project(STL_Projects)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Add subdirectories for each project
add_subdirectory(STLDataStructures)
add_subdirectory(AdvancedAlgorithms)
add_subdirectory(MemoryPoolAllocator)
add_subdirectory(FunctionalProgramming)
add_subdirectory(ConcurrentDataStructures)

# Benchmark target
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_subdirectory(benchmarks)
endif()

# Documentation target
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_subdirectory(docs)
endif()
```

### Unit Testing Framework
```cpp
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "circular_buffer.hpp"

class CircularBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer = std::make_unique<CircularBuffer<int>>(5);
    }
    
    std::unique_ptr<CircularBuffer<int>> buffer;
};

TEST_F(CircularBufferTest, BasicOperations) {
    EXPECT_TRUE(buffer->empty());
    EXPECT_EQ(buffer->size(), 0);
    EXPECT_EQ(buffer->capacity(), 5);
    
    buffer->push_back(1);
    EXPECT_FALSE(buffer->empty());
    EXPECT_EQ(buffer->size(), 1);
    EXPECT_EQ(buffer->front(), 1);
    EXPECT_EQ(buffer->back(), 1);
}

TEST_F(CircularBufferTest, STLCompatibility) {
    for (int i = 1; i <= 5; ++i) {
        buffer->push_back(i);
    }
    
    // Test with STL algorithms
    auto it = std::find(buffer->begin(), buffer->end(), 3);
    EXPECT_NE(it, buffer->end());
    EXPECT_EQ(*it, 3);
    
    int sum = std::accumulate(buffer->begin(), buffer->end(), 0);
    EXPECT_EQ(sum, 15);
    
    std::vector<int> vec(buffer->begin(), buffer->end());
    EXPECT_EQ(vec.size(), 5);
    EXPECT_EQ(vec[0], 1);
    EXPECT_EQ(vec[4], 5);
}

TEST_F(CircularBufferTest, OverwriteBehavior) {
    // Fill beyond capacity
    for (int i = 1; i <= 7; ++i) {
        buffer->push_back(i);
    }
    
    EXPECT_EQ(buffer->size(), 5);
    EXPECT_EQ(buffer->front(), 3); // First two elements were overwritten
    EXPECT_EQ(buffer->back(), 7);
}
```

### Performance Benchmarking
```cpp
#include <benchmark/benchmark.h>
#include <vector>
#include <deque>
#include "circular_buffer.hpp"

static void BM_CircularBuffer_PushBack(benchmark::State& state) {
    CircularBuffer<int> buffer(state.range(0));
    
    for (auto _ : state) {
        for (int i = 0; i < state.range(0); ++i) {
            buffer.push_back(i);
        }
        benchmark::DoNotOptimize(buffer);
    }
}

static void BM_Vector_PushBack(benchmark::State& state) {
    std::vector<int> vec;
    vec.reserve(state.range(0));
    
    for (auto _ : state) {
        vec.clear();
        for (int i = 0; i < state.range(0); ++i) {
            vec.push_back(i);
        }
        benchmark::DoNotOptimize(vec);
    }
}

static void BM_Deque_PushBack(benchmark::State& state) {
    std::deque<int> deque;
    
    for (auto _ : state) {
        deque.clear();
        for (int i = 0; i < state.range(0); ++i) {
            deque.push_back(i);
        }
        benchmark::DoNotOptimize(deque);
    }
}

BENCHMARK(BM_CircularBuffer_PushBack)->Range(8, 8<<10);
BENCHMARK(BM_Vector_PushBack)->Range(8, 8<<10);
BENCHMARK(BM_Deque_PushBack)->Range(8, 8<<10);

BENCHMARK_MAIN();
```

## Documentation and Deployment

### API Documentation (Doxygen)
```cpp
/**
 * @brief A circular buffer implementation with STL compatibility
 * 
 * @tparam T The type of elements stored in the buffer
 * @tparam Allocator The allocator type used for memory management
 * 
 * This class provides a fixed-size circular buffer that overwrites
 * the oldest elements when the buffer is full. It is fully compatible
 * with STL algorithms and provides the standard container interface.
 * 
 * @par Complexity Guarantees:
 * - Element access: O(1)
 * - Insertion/Removal: O(1)
 * - Iterator operations: O(1)
 * 
 * @par Thread Safety:
 * This class is not thread-safe. External synchronization is required
 * for concurrent access.
 * 
 * @example
 * @code
 * CircularBuffer<int> buffer(10);
 * buffer.push_back(42);
 * 
 * // Use with STL algorithms
 * std::sort(buffer.begin(), buffer.end());
 * auto it = std::find(buffer.begin(), buffer.end(), 42);
 * @endcode
 */
template<typename T, typename Allocator = std::allocator<T>>
class CircularBuffer {
    // Implementation...
};
```

### Project Deliverables

1. **Source Code**: Complete, documented, and tested implementations
2. **Documentation**: API documentation, design documents, usage guides
3. **Tests**: Unit tests, integration tests, performance benchmarks
4. **Examples**: Usage examples, tutorials, best practices
5. **Build System**: CMake configuration, CI/CD setup
6. **Performance Analysis**: Benchmark results, complexity analysis

### Success Criteria

1. **Functionality**: All data structures work correctly
2. **STL Compatibility**: Full compatibility with STL algorithms
3. **Performance**: Competitive with or better than standard alternatives
4. **Code Quality**: Clean, readable, well-documented code
5. **Testing**: Comprehensive test coverage (>90%)
6. **Documentation**: Complete API documentation and usage guides

## Extension Ideas

1. **Persistence**: Add support for serialization/deserialization
2. **Debugging**: Add debug modes with additional safety checks
3. **Metrics**: Built-in performance monitoring and statistics
4. **Adaptors**: Create adaptors for existing STL containers
5. **Specialized Versions**: SIMD-optimized versions for numeric types
6. **GPU Support**: CUDA-compatible versions of data structures
7. **Network Serialization**: Built-in support for network protocols
