# STL Extension Points

*Part of STL Learning Track - 1 week*

## Overview

STL extension points allow you to create custom components that integrate seamlessly with the STL ecosystem. This includes creating STL-compatible containers, iterators, algorithms, and customization points. Understanding these concepts enables you to extend the STL with your own types while maintaining compatibility with existing STL algorithms and functions.

## Creating STL-Compatible Containers

### Basic STL-Compatible Container
```cpp
#include <iterator>
#include <algorithm>
#include <iostream>
#include <initializer_list>

template<typename T>
class SimpleVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
    void reallocate(size_t new_capacity) {
        T* new_data = static_cast<T*>(std::malloc(new_capacity * sizeof(T)));
        
        // Move construct elements
        for (size_t i = 0; i < size_; ++i) {
            new(new_data + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        
        std::free(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }
    
public:
    // STL container typedefs
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    // Forward declarations for iterators
    class iterator;
    class const_iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    
    // Constructors
    SimpleVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit SimpleVector(size_type count) : SimpleVector() {
        resize(count);
    }
    
    SimpleVector(size_type count, const T& value) : SimpleVector() {
        resize(count, value);
    }
    
    SimpleVector(std::initializer_list<T> init) : SimpleVector() {
        reserve(init.size());
        for (const auto& item : init) {
            push_back(item);
        }
    }
    
    // Copy constructor
    SimpleVector(const SimpleVector& other) : SimpleVector() {
        reserve(other.capacity_);
        for (size_t i = 0; i < other.size_; ++i) {
            push_back(other.data_[i]);
        }
    }
    
    // Move constructor
    SimpleVector(SimpleVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Destructor
    ~SimpleVector() {
        clear();
        std::free(data_);
    }
    
    // Assignment operators
    SimpleVector& operator=(const SimpleVector& other) {
        if (this != &other) {
            clear();
            reserve(other.capacity_);
            for (size_t i = 0; i < other.size_; ++i) {
                push_back(other.data_[i]);
            }
        }
        return *this;
    }
    
    SimpleVector& operator=(SimpleVector&& other) noexcept {
        if (this != &other) {
            clear();
            std::free(data_);
            
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    // Element access
    reference operator[](size_type pos) { return data_[pos]; }
    const_reference operator[](size_type pos) const { return data_[pos]; }
    
    reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("SimpleVector::at");
        }
        return data_[pos];
    }
    
    const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("SimpleVector::at");
        }
        return data_[pos];
    }
    
    reference front() { return data_[0]; }
    const_reference front() const { return data_[0]; }
    
    reference back() { return data_[size_ - 1]; }
    const_reference back() const { return data_[size_ - 1]; }
    
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    
    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    
    void reserve(size_type new_cap) {
        if (new_cap > capacity_) {
            reallocate(new_cap);
        }
    }
    
    void shrink_to_fit() {
        if (size_ < capacity_) {
            reallocate(size_);
        }
    }
    
    // Modifiers
    void clear() noexcept {
        for (size_t i = 0; i < size_; ++i) {
            data_[i].~T();
        }
        size_ = 0;
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(value);
        ++size_;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(std::move(value));
        ++size_;
    }
    
    template<typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new(data_ + size_) T(std::forward<Args>(args)...);
        ++size_;
        return back();
    }
    
    void pop_back() {
        if (size_ > 0) {
            --size_;
            data_[size_].~T();
        }
    }
    
    void resize(size_type count) {
        if (count > size_) {
            reserve(count);
            for (size_t i = size_; i < count; ++i) {
                new(data_ + i) T();
            }
        } else {
            for (size_t i = count; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = count;
    }
    
    void resize(size_type count, const T& value) {
        if (count > size_) {
            reserve(count);
            for (size_t i = size_; i < count; ++i) {
                new(data_ + i) T(value);
            }
        } else {
            for (size_t i = count; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = count;
    }
    
    // Iterator classes
    class iterator {
    private:
        T* ptr_;
        
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        
        iterator(T* ptr) : ptr_(ptr) {}
        
        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        
        iterator& operator++() { ++ptr_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++ptr_; return tmp; }
        iterator& operator--() { --ptr_; return *this; }
        iterator operator--(int) { iterator tmp = *this; --ptr_; return tmp; }
        
        iterator operator+(difference_type n) const { return iterator(ptr_ + n); }
        iterator operator-(difference_type n) const { return iterator(ptr_ - n); }
        iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }
        
        difference_type operator-(const iterator& other) const { return ptr_ - other.ptr_; }
        
        reference operator[](difference_type n) const { return ptr_[n]; }
        
        bool operator==(const iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const iterator& other) const { return ptr_ < other.ptr_; }
        bool operator>(const iterator& other) const { return ptr_ > other.ptr_; }
        bool operator<=(const iterator& other) const { return ptr_ <= other.ptr_; }
        bool operator>=(const iterator& other) const { return ptr_ >= other.ptr_; }
    };
    
    class const_iterator {
    private:
        const T* ptr_;
        
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;
        
        const_iterator(const T* ptr) : ptr_(ptr) {}
        const_iterator(const iterator& it) : ptr_(&(*it)) {}
        
        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        
        const_iterator& operator++() { ++ptr_; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++ptr_; return tmp; }
        const_iterator& operator--() { --ptr_; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --ptr_; return tmp; }
        
        const_iterator operator+(difference_type n) const { return const_iterator(ptr_ + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(ptr_ - n); }
        const_iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        const_iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }
        
        difference_type operator-(const const_iterator& other) const { return ptr_ - other.ptr_; }
        
        reference operator[](difference_type n) const { return ptr_[n]; }
        
        bool operator==(const const_iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const const_iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const const_iterator& other) const { return ptr_ < other.ptr_; }
        bool operator>(const const_iterator& other) const { return ptr_ > other.ptr_; }
        bool operator<=(const const_iterator& other) const { return ptr_ <= other.ptr_; }
        bool operator>=(const const_iterator& other) const { return ptr_ >= other.ptr_; }
    };
    
    // Iterator methods
    iterator begin() { return iterator(data_); }
    const_iterator begin() const { return const_iterator(data_); }
    const_iterator cbegin() const { return const_iterator(data_); }
    
    iterator end() { return iterator(data_ + size_); }
    const_iterator end() const { return const_iterator(data_ + size_); }
    const_iterator cend() const { return const_iterator(data_ + size_); }
    
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
    
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }
};

void custom_container_example() {
    std::cout << "=== Custom STL-Compatible Container ===" << std::endl;
    
    SimpleVector<int> vec{1, 2, 3, 4, 5};
    
    std::cout << "Initial vector: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Use STL algorithms
    std::cout << "Using std::find: ";
    auto it = std::find(vec.begin(), vec.end(), 3);
    if (it != vec.end()) {
        std::cout << "Found " << *it << " at position " << std::distance(vec.begin(), it) << std::endl;
    }
    
    // Sort the vector
    std::sort(vec.rbegin(), vec.rend()); // Reverse sort
    std::cout << "After reverse sort: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Count elements
    int count = std::count_if(vec.begin(), vec.end(), [](int n) { return n > 3; });
    std::cout << "Elements > 3: " << count << std::endl;
    
    // Transform
    std::transform(vec.begin(), vec.end(), vec.begin(), [](int n) { return n * 2; });
    std::cout << "After doubling: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### Specialized Container (Ring Buffer)
```cpp
#include <iterator>
#include <algorithm>
#include <iostream>
#include <stdexcept>

template<typename T, size_t N>
class RingBuffer {
private:
    T data_[N];
    size_t head_;
    size_t tail_;
    size_t size_;
    
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    
    class iterator {
    private:
        RingBuffer* buffer_;
        size_t index_;
        
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        
        iterator(RingBuffer* buffer, size_t index) : buffer_(buffer), index_(index) {}
        
        reference operator*() const { 
            return buffer_->data_[(buffer_->head_ + index_) % N]; 
        }
        
        pointer operator->() const { 
            return &buffer_->data_[(buffer_->head_ + index_) % N]; 
        }
        
        iterator& operator++() { ++index_; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++index_; return tmp; }
        iterator& operator--() { --index_; return *this; }
        iterator operator--(int) { iterator tmp = *this; --index_; return tmp; }
        
        iterator operator+(difference_type n) const { return iterator(buffer_, index_ + n); }
        iterator operator-(difference_type n) const { return iterator(buffer_, index_ - n); }
        iterator& operator+=(difference_type n) { index_ += n; return *this; }
        iterator& operator-=(difference_type n) { index_ -= n; return *this; }
        
        difference_type operator-(const iterator& other) const { return index_ - other.index_; }
        
        reference operator[](difference_type n) const { 
            return buffer_->data_[(buffer_->head_ + index_ + n) % N]; 
        }
        
        bool operator==(const iterator& other) const { return index_ == other.index_; }
        bool operator!=(const iterator& other) const { return index_ != other.index_; }
        bool operator<(const iterator& other) const { return index_ < other.index_; }
        bool operator>(const iterator& other) const { return index_ > other.index_; }
        bool operator<=(const iterator& other) const { return index_ <= other.index_; }
        bool operator>=(const iterator& other) const { return index_ >= other.index_; }
    };
    
    RingBuffer() : head_(0), tail_(0), size_(0) {}
    
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == N; }
    size_type size() const { return size_; }
    constexpr size_type capacity() const { return N; }
    
    void push_back(const T& value) {
        data_[tail_] = value;
        tail_ = (tail_ + 1) % N;
        
        if (size_ < N) {
            ++size_;
        } else {
            head_ = (head_ + 1) % N; // Overwrite oldest
        }
    }
    
    void push_back(T&& value) {
        data_[tail_] = std::move(value);
        tail_ = (tail_ + 1) % N;
        
        if (size_ < N) {
            ++size_;
        } else {
            head_ = (head_ + 1) % N; // Overwrite oldest
        }
    }
    
    void pop_front() {
        if (!empty()) {
            head_ = (head_ + 1) % N;
            --size_;
        }
    }
    
    reference front() { return data_[head_]; }
    const_reference front() const { return data_[head_]; }
    
    reference back() { return data_[(tail_ - 1 + N) % N]; }
    const_reference back() const { return data_[(tail_ - 1 + N) % N]; }
    
    reference operator[](size_type pos) {
        return data_[(head_ + pos) % N];
    }
    
    const_reference operator[](size_type pos) const {
        return data_[(head_ + pos) % N];
    }
    
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size_); }
    
    // Const iterators would need similar implementation
};

void ring_buffer_example() {
    std::cout << "\n=== Ring Buffer Container ===" << std::endl;
    
    RingBuffer<int, 5> ring;
    
    // Fill the ring buffer
    for (int i = 1; i <= 7; ++i) {
        ring.push_back(i);
        std::cout << "After pushing " << i << ": ";
        for (const auto& item : ring) {
            std::cout << item << " ";
        }
        std::cout << "(size: " << ring.size() << ")" << std::endl;
    }
    
    // Use STL algorithms
    auto max_it = std::max_element(ring.begin(), ring.end());
    std::cout << "Max element: " << *max_it << std::endl;
    
    auto count = std::count_if(ring.begin(), ring.end(), [](int n) { return n > 5; });
    std::cout << "Elements > 5: " << count << std::endl;
}
```

## Creating STL-Compatible Iterators

### Custom Iterator for Filtered View
```cpp
#include <iterator>
#include <functional>
#include <vector>
#include <iostream>

template<typename Iterator, typename Predicate>
class FilterIterator {
private:
    Iterator current_;
    Iterator end_;
    Predicate predicate_;
    
    void advance_to_next_valid() {
        while (current_ != end_ && !predicate_(*current_)) {
            ++current_;
        }
    }
    
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = typename std::iterator_traits<Iterator>::pointer;
    using reference = typename std::iterator_traits<Iterator>::reference;
    
    FilterIterator(Iterator begin, Iterator end, Predicate pred)
        : current_(begin), end_(end), predicate_(pred) {
        advance_to_next_valid();
    }
    
    FilterIterator(Iterator end) : current_(end), end_(end) {}
    
    reference operator*() const { return *current_; }
    pointer operator->() const { return &(*current_); }
    
    FilterIterator& operator++() {
        ++current_;
        advance_to_next_valid();
        return *this;
    }
    
    FilterIterator operator++(int) {
        FilterIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    bool operator==(const FilterIterator& other) const {
        return current_ == other.current_;
    }
    
    bool operator!=(const FilterIterator& other) const {
        return !(*this == other);
    }
};

// Helper function to create filter iterators
template<typename Container, typename Predicate>
auto make_filter_range(Container& container, Predicate pred) {
    using Iterator = decltype(container.begin());
    
    struct FilterRange {
        FilterIterator<Iterator, Predicate> begin_it;
        FilterIterator<Iterator, Predicate> end_it;
        
        auto begin() { return begin_it; }
        auto end() { return end_it; }
    };
    
    return FilterRange{
        FilterIterator<Iterator, Predicate>(container.begin(), container.end(), pred),
        FilterIterator<Iterator, Predicate>(container.end())
    };
}

void filter_iterator_example() {
    std::cout << "\n=== Filter Iterator Example ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    std::cout << "Original numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Filter even numbers
    auto even_range = make_filter_range(numbers, [](int n) { return n % 2 == 0; });
    
    std::cout << "Even numbers: ";
    for (int n : even_range) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Filter numbers > 5
    auto greater_than_5 = make_filter_range(numbers, [](int n) { return n > 5; });
    
    std::cout << "Numbers > 5: ";
    for (int n : greater_than_5) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Use with STL algorithms
    auto count = std::distance(even_range.begin(), even_range.end());
    std::cout << "Count of even numbers: " << count << std::endl;
}
```

### Transform Iterator
```cpp
#include <iterator>
#include <functional>
#include <vector>
#include <iostream>
#include <string>

template<typename Iterator, typename Function>
class TransformIterator {
private:
    Iterator current_;
    Function function_;
    
public:
    using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
    using value_type = std::decay_t<decltype(function_(*current_))>;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = value_type*;
    using reference = value_type;
    
    TransformIterator(Iterator it, Function func) : current_(it), function_(func) {}
    
    reference operator*() const { return function_(*current_); }
    
    TransformIterator& operator++() {
        ++current_;
        return *this;
    }
    
    TransformIterator operator++(int) {
        TransformIterator tmp = *this;
        ++current_;
        return tmp;
    }
    
    TransformIterator& operator--() {
        --current_;
        return *this;
    }
    
    TransformIterator operator--(int) {
        TransformIterator tmp = *this;
        --current_;
        return tmp;
    }
    
    TransformIterator operator+(difference_type n) const {
        return TransformIterator(current_ + n, function_);
    }
    
    TransformIterator operator-(difference_type n) const {
        return TransformIterator(current_ - n, function_);
    }
    
    TransformIterator& operator+=(difference_type n) {
        current_ += n;
        return *this;
    }
    
    TransformIterator& operator-=(difference_type n) {
        current_ -= n;
        return *this;
    }
    
    difference_type operator-(const TransformIterator& other) const {
        return current_ - other.current_;
    }
    
    reference operator[](difference_type n) const {
        return function_(current_[n]);
    }
    
    bool operator==(const TransformIterator& other) const {
        return current_ == other.current_;
    }
    
    bool operator!=(const TransformIterator& other) const {
        return current_ != other.current_;
    }
    
    bool operator<(const TransformIterator& other) const {
        return current_ < other.current_;
    }
    
    bool operator>(const TransformIterator& other) const {
        return current_ > other.current_;
    }
    
    bool operator<=(const TransformIterator& other) const {
        return current_ <= other.current_;
    }
    
    bool operator>=(const TransformIterator& other) const {
        return current_ >= other.current_;
    }
};

// Helper function to create transform range
template<typename Container, typename Function>
auto make_transform_range(Container& container, Function func) {
    using Iterator = decltype(container.begin());
    
    struct TransformRange {
        TransformIterator<Iterator, Function> begin_it;
        TransformIterator<Iterator, Function> end_it;
        
        auto begin() { return begin_it; }
        auto end() { return end_it; }
    };
    
    return TransformRange{
        TransformIterator<Iterator, Function>(container.begin(), func),
        TransformIterator<Iterator, Function>(container.end(), func)
    };
}

void transform_iterator_example() {
    std::cout << "\n=== Transform Iterator Example ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    std::cout << "Original numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Transform to squares
    auto squared = make_transform_range(numbers, [](int n) { return n * n; });
    
    std::cout << "Squared: ";
    for (int n : squared) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Transform to strings
    auto as_strings = make_transform_range(numbers, [](int n) { return std::to_string(n); });
    
    std::cout << "As strings: ";
    for (const auto& s : as_strings) {
        std::cout << "'" << s << "' ";
    }
    std::cout << std::endl;
    
    // Use with STL algorithms
    auto max_squared = *std::max_element(squared.begin(), squared.end());
    std::cout << "Max squared value: " << max_squared << std::endl;
}
```

## Writing STL-Compatible Algorithms

### Custom Algorithm: find_all
```cpp
#include <iterator>
#include <vector>
#include <algorithm>
#include <iostream>

// Algorithm that finds all occurrences of a value
template<typename InputIterator, typename T>
std::vector<InputIterator> find_all(InputIterator first, InputIterator last, const T& value) {
    std::vector<InputIterator> results;
    
    while (first != last) {
        first = std::find(first, last, value);
        if (first != last) {
            results.push_back(first);
            ++first;
        }
    }
    
    return results;
}

// Algorithm that finds all occurrences matching a predicate
template<typename InputIterator, typename Predicate>
std::vector<InputIterator> find_all_if(InputIterator first, InputIterator last, Predicate pred) {
    std::vector<InputIterator> results;
    
    while (first != last) {
        first = std::find_if(first, last, pred);
        if (first != last) {
            results.push_back(first);
            ++first;
        }
    }
    
    return results;
}

// Algorithm that counts consecutive elements
template<typename InputIterator>
std::vector<std::pair<typename std::iterator_traits<InputIterator>::value_type, size_t>>
count_consecutive(InputIterator first, InputIterator last) {
    using value_type = typename std::iterator_traits<InputIterator>::value_type;
    std::vector<std::pair<value_type, size_t>> results;
    
    if (first == last) return results;
    
    value_type current = *first;
    size_t count = 1;
    ++first;
    
    while (first != last) {
        if (*first == current) {
            ++count;
        } else {
            results.emplace_back(current, count);
            current = *first;
            count = 1;
        }
        ++first;
    }
    
    results.emplace_back(current, count);
    return results;
}

void custom_algorithms_example() {
    std::cout << "\n=== Custom STL-Compatible Algorithms ===" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 2, 4, 2, 5, 6, 2};
    
    std::cout << "Numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Find all occurrences of 2
    auto twos = find_all(numbers.begin(), numbers.end(), 2);
    std::cout << "Found " << twos.size() << " occurrences of 2 at positions: ";
    for (auto it : twos) {
        std::cout << std::distance(numbers.begin(), it) << " ";
    }
    std::cout << std::endl;
    
    // Find all even numbers
    auto evens = find_all_if(numbers.begin(), numbers.end(), [](int n) { return n % 2 == 0; });
    std::cout << "Found " << evens.size() << " even numbers at positions: ";
    for (auto it : evens) {
        std::cout << std::distance(numbers.begin(), it) << " ";
    }
    std::cout << std::endl;
    
    // Count consecutive elements
    std::vector<int> consecutive = {1, 1, 1, 2, 2, 3, 1, 1, 4, 4, 4, 4};
    std::cout << "Consecutive: ";
    for (int n : consecutive) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    auto counts = count_consecutive(consecutive.begin(), consecutive.end());
    std::cout << "Consecutive counts: ";
    for (const auto& [value, count] : counts) {
        std::cout << value << "(" << count << ") ";
    }
    std::cout << std::endl;
}
```

### Custom Algorithm: parallel_transform
```cpp
#include <algorithm>
#include <vector>
#include <future>
#include <thread>
#include <iostream>

// Simple parallel transform (for demonstration)
template<typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator parallel_transform(InputIterator first, InputIterator last, 
                                  OutputIterator result, UnaryFunction func,
                                  size_t num_threads = std::thread::hardware_concurrency()) {
    
    size_t distance = std::distance(first, last);
    if (distance == 0) return result;
    
    if (distance < num_threads || num_threads == 1) {
        // Fall back to sequential for small ranges
        return std::transform(first, last, result, func);
    }
    
    std::vector<std::future<void>> futures;
    size_t chunk_size = distance / num_threads;
    
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_first = first;
        std::advance(chunk_first, i * chunk_size);
        auto chunk_last = chunk_first;
        std::advance(chunk_last, chunk_size);
        auto chunk_result = result;
        std::advance(chunk_result, i * chunk_size);
        
        futures.push_back(std::async(std::launch::async, [=]() {
            std::transform(chunk_first, chunk_last, chunk_result, func);
        }));
    }
    
    // Handle the last chunk (may be larger due to remainder)
    auto last_chunk_first = first;
    std::advance(last_chunk_first, (num_threads - 1) * chunk_size);
    auto last_chunk_result = result;
    std::advance(last_chunk_result, (num_threads - 1) * chunk_size);
    
    std::transform(last_chunk_first, last, last_chunk_result, func);
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    std::advance(result, distance);
    return result;
}

void parallel_algorithm_example() {
    std::cout << "\n=== Parallel Algorithm Example ===" << std::endl;
    
    const size_t size = 1000000;
    std::vector<int> input(size);
    std::vector<int> output1(size);
    std::vector<int> output2(size);
    
    // Initialize input
    std::iota(input.begin(), input.end(), 1);
    
    auto expensive_function = [](int x) {
        // Simulate expensive computation
        int result = x;
        for (int i = 0; i < 100; ++i) {
            result = (result * 7) % 1000000;
        }
        return result;
    };
    
    // Time sequential version
    auto start = std::chrono::high_resolution_clock::now();
    std::transform(input.begin(), input.end(), output1.begin(), expensive_function);
    auto end = std::chrono::high_resolution_clock::now();
    auto sequential_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Time parallel version
    start = std::chrono::high_resolution_clock::now();
    parallel_transform(input.begin(), input.end(), output2.begin(), expensive_function);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify results are the same
    bool results_match = std::equal(output1.begin(), output1.end(), output2.begin());
    
    std::cout << "Sequential time: " << sequential_time.count() << "ms" << std::endl;
    std::cout << "Parallel time: " << parallel_time.count() << "ms" << std::endl;
    std::cout << "Results match: " << results_match << std::endl;
    
    if (parallel_time.count() > 0) {
        double speedup = static_cast<double>(sequential_time.count()) / parallel_time.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }
}
```

## Customization Points

### std::swap Specialization
```cpp
#include <algorithm>
#include <iostream>
#include <vector>

class BigObject {
private:
    std::vector<int> data_;
    std::string name_;
    
public:
    BigObject(const std::string& name, size_t size) 
        : data_(size, 42), name_(name) {
        std::cout << "BigObject " << name_ << " created" << std::endl;
    }
    
    // Copy constructor (expensive)
    BigObject(const BigObject& other) 
        : data_(other.data_), name_(other.name_ + "_copy") {
        std::cout << "BigObject " << name_ << " copied (expensive!)" << std::endl;
    }
    
    // Move constructor (cheap)
    BigObject(BigObject&& other) noexcept
        : data_(std::move(other.data_)), name_(std::move(other.name_)) {
        std::cout << "BigObject " << name_ << " moved (cheap!)" << std::endl;
    }
    
    // Assignment operators
    BigObject& operator=(const BigObject& other) {
        if (this != &other) {
            data_ = other.data_;
            name_ = other.name_ + "_assigned";
            std::cout << "BigObject " << name_ << " copy assigned" << std::endl;
        }
        return *this;
    }
    
    BigObject& operator=(BigObject&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            name_ = std::move(other.name_);
            std::cout << "BigObject " << name_ << " move assigned" << std::endl;
        }
        return *this;
    }
    
    const std::string& name() const { return name_; }
    size_t size() const { return data_.size(); }
    
    // Efficient swap member function
    void swap(BigObject& other) noexcept {
        data_.swap(other.data_);
        name_.swap(other.name_);
        std::cout << "Efficient swap performed" << std::endl;
    }
};

// Specialize std::swap for BigObject
namespace std {
    template<>
    void swap<BigObject>(BigObject& a, BigObject& b) noexcept {
        a.swap(b);
    }
}

void customization_points_example() {
    std::cout << "\n=== Customization Points Example ===" << std::endl;
    
    BigObject obj1("Object1", 1000);
    BigObject obj2("Object2", 2000);
    
    std::cout << "\nBefore swap:" << std::endl;
    std::cout << "obj1: " << obj1.name() << " (size: " << obj1.size() << ")" << std::endl;
    std::cout << "obj2: " << obj2.name() << " (size: " << obj2.size() << ")" << std::endl;
    
    // This will use our specialized swap
    std::cout << "\nPerforming swap:" << std::endl;
    std::swap(obj1, obj2);
    
    std::cout << "\nAfter swap:" << std::endl;
    std::cout << "obj1: " << obj1.name() << " (size: " << obj1.size() << ")" << std::endl;
    std::cout << "obj2: " << obj2.name() << " (size: " << obj2.size() << ")" << std::endl;
    
    // Test with STL algorithms that use swap
    std::vector<BigObject> objects;
    objects.emplace_back("A", 100);
    objects.emplace_back("B", 200);
    objects.emplace_back("C", 300);
    
    std::cout << "\nReversing vector (uses swap internally):" << std::endl;
    std::reverse(objects.begin(), objects.end());
}
```

### Hash Function Specialization
```cpp
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <string>

struct Point {
    int x, y;
    
    Point(int x_, int y_) : x(x_), y(y_) {}
    
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

// Specialize std::hash for Point
namespace std {
    template<>
    struct hash<Point> {
        size_t operator()(const Point& p) const noexcept {
            size_t h1 = std::hash<int>{}(p.x);
            size_t h2 = std::hash<int>{}(p.y);
            return h1 ^ (h2 << 1); // Simple hash combination
        }
    };
}

// Custom hash function class
struct PointHash {
    size_t operator()(const Point& p) const noexcept {
        // Better hash combination using prime numbers
        return std::hash<int>{}(p.x) * 31 + std::hash<int>{}(p.y);
    }
};

void hash_specialization_example() {
    std::cout << "\n=== Hash Function Specialization ===" << std::endl;
    
    // Using std::hash specialization
    std::unordered_set<Point> point_set;
    point_set.insert({1, 2});
    point_set.insert({3, 4});
    point_set.insert({1, 2}); // Duplicate, won't be inserted
    
    std::cout << "Point set size: " << point_set.size() << std::endl;
    
    // Using custom hash function
    std::unordered_set<Point, PointHash> custom_point_set;
    custom_point_set.insert({5, 6});
    custom_point_set.insert({7, 8});
    
    std::cout << "Custom point set size: " << custom_point_set.size() << std::endl;
    
    // Using with unordered_map
    std::unordered_map<Point, std::string> point_names;
    point_names[{0, 0}] = "Origin";
    point_names[{1, 0}] = "X-axis";
    point_names[{0, 1}] = "Y-axis";
    
    std::cout << "\nPoint names:" << std::endl;
    for (const auto& [point, name] : point_names) {
        std::cout << "(" << point.x << ", " << point.y << ") -> " << name << std::endl;
    }
    
    // Test hash quality
    std::cout << "\nHash values:" << std::endl;
    std::hash<Point> hasher;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            Point p(x, y);
            std::cout << "(" << x << ", " << y << ") -> " << hasher(p) << std::endl;
        }
    }
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

int main() {
    std::cout << "=== STL Extension Points Examples ===" << std::endl;
    
    std::cout << "\n--- Custom Containers ---" << std::endl;
    custom_container_example();
    ring_buffer_example();
    
    std::cout << "\n--- Custom Iterators ---" << std::endl;
    filter_iterator_example();
    transform_iterator_example();
    
    std::cout << "\n--- Custom Algorithms ---" << std::endl;
    custom_algorithms_example();
    parallel_algorithm_example();
    
    std::cout << "\n--- Customization Points ---" << std::endl;
    customization_points_example();
    hash_specialization_example();
    
    return 0;
}
```

## Best Practices for STL Extension

### 1. Follow STL Conventions
- Use standard typedefs (`value_type`, `iterator`, etc.)
- Implement standard member functions (`begin()`, `end()`, `size()`, etc.)
- Follow naming conventions
- Provide appropriate iterator categories

### 2. Exception Safety
- Provide strong exception safety guarantees where possible
- Use RAII for resource management
- Consider `noexcept` specifications

### 3. Performance Considerations
- Avoid unnecessary copies
- Consider move semantics
- Optimize for common use cases
- Profile custom implementations

### 4. Documentation and Testing
- Document iterator categories and complexity guarantees
- Test with various STL algorithms
- Provide usage examples
- Consider edge cases

## Key Concepts Summary

1. **Container Requirements**: Implement standard typedefs and member functions
2. **Iterator Requirements**: Follow iterator category requirements and provide proper operators
3. **Algorithm Design**: Use iterator-based interfaces for maximum flexibility
4. **Customization Points**: Specialize standard functions like `std::swap` and `std::hash`
5. **STL Compatibility**: Ensure your types work with existing STL algorithms

## Common Pitfalls

1. **Incomplete Iterator Interface**: Missing required operators or typedefs
2. **Invalid Iterator Categories**: Claiming higher category than actually supported
3. **Exception Safety**: Not providing proper exception guarantees
4. **Performance Issues**: Inefficient implementations that don't meet complexity requirements
5. **Thread Safety**: Not considering concurrent access patterns

## Exercises

1. Implement a doubly-linked list container with full STL compatibility
2. Create a binary tree iterator that supports in-order traversal
3. Write a parallel version of `std::for_each`
4. Implement a memory-mapped file container with STL interface
5. Create a compressed string container that works with string algorithms
6. Build a priority queue with STL-compatible iterators
7. Implement a custom allocator and integrate it with STL containers
8. Create a thread-safe container wrapper that maintains STL interface
