# STL Iterators

*Part of STL Learning Track - 1 week*

## Overview

Iterators are objects that act as a bridge between containers and algorithms. They provide a way to access elements in a container sequentially without exposing the underlying representation of the container.

## Iterator Categories

STL defines several categories of iterators, each with different capabilities:

### 1. Input Iterators
```cpp
#include <iostream>
#include <iterator>
#include <sstream>

void input_iterator_examples() {
    // std::istream_iterator is an input iterator
    std::istringstream iss("1 2 3 4 5");
    std::istream_iterator<int> input_it(iss);
    std::istream_iterator<int> end_it;
    
    std::cout << "Reading from stream: ";
    while (input_it != end_it) {
        std::cout << *input_it << " ";
        ++input_it;
    }
    std::cout << std::endl;
    
    // Input iterators are single-pass
    // Cannot be used multiple times
}
```

### 2. Output Iterators
```cpp
#include <iostream>
#include <iterator>
#include <vector>

void output_iterator_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // std::ostream_iterator is an output iterator
    std::ostream_iterator<int> output_it(std::cout, " ");
    
    std::cout << "Output iterator: ";
    for (const auto& item : vec) {
        *output_it = item;
        ++output_it;
    }
    std::cout << std::endl;
    
    // Using with algorithms
    std::copy(vec.begin(), vec.end(), 
              std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;
}
```

### 3. Forward Iterators
```cpp
#include <forward_list>
#include <iostream>

void forward_iterator_examples() {
    std::forward_list<int> flist = {1, 2, 3, 4, 5};
    
    // Forward iterators can be used multiple times
    auto it = flist.begin();
    auto saved_it = it; // Copy iterator
    
    std::cout << "First pass: ";
    while (it != flist.end()) {
        std::cout << *it << " ";
        ++it;
    }
    std::cout << std::endl;
    
    std::cout << "Second pass from saved position: ";
    while (saved_it != flist.end()) {
        std::cout << *saved_it << " ";
        ++saved_it;
    }
    std::cout << std::endl;
}
```

### 4. Bidirectional Iterators
```cpp
#include <list>
#include <set>
#include <iostream>

void bidirectional_iterator_examples() {
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    // Forward iteration
    std::cout << "Forward: ";
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // Backward iteration
    std::cout << "Backward: ";
    for (auto it = lst.rbegin(); it != lst.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // Manual backward iteration
    std::cout << "Manual backward: ";
    auto it = lst.end();
    while (it != lst.begin()) {
        --it;
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // Set iterators are also bidirectional
    std::set<int> s = {3, 1, 4, 1, 5, 9, 2, 6};
    std::cout << "Set forward: ";
    for (auto it = s.begin(); it != s.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}
```

### 5. Random Access Iterators
```cpp
#include <vector>
#include <deque>
#include <algorithm>
#include <iostream>

void random_access_iterator_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto it = vec.begin();
    
    // Arithmetic operations
    std::cout << "Element at index 0: " << *it << std::endl;
    std::cout << "Element at index 3: " << *(it + 3) << std::endl;
    std::cout << "Element at index 7: " << *(it + 7) << std::endl;
    
    // Iterator arithmetic
    auto it2 = vec.end() - 1;
    std::cout << "Last element: " << *it2 << std::endl;
    
    // Distance between iterators
    std::cout << "Distance: " << std::distance(it, it2) << std::endl;
    
    // Comparison operations
    if (it < it2) {
        std::cout << "it is before it2" << std::endl;
    }
    
    // Subscript operator
    std::cout << "it[5] = " << it[5] << std::endl;
    
    // Random access enables efficient algorithms
    std::sort(vec.begin(), vec.end(), std::greater<int>());
    std::cout << "Sorted (descending): ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### 6. Contiguous Iterators (C++17)
```cpp
#include <vector>
#include <array>
#include <iostream>

void contiguous_iterator_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Contiguous iterators guarantee that elements are stored contiguously
    auto it = vec.begin();
    
    // Can use pointer arithmetic
    int* ptr = &(*it);
    std::cout << "Using pointer arithmetic: ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << ptr[i] << " ";
    }
    std::cout << std::endl;
    
    // std::array also has contiguous iterators
    std::array<int, 5> arr = {10, 20, 30, 40, 50};
    auto arr_it = arr.begin();
    int* arr_ptr = &(*arr_it);
    
    std::cout << "Array elements: ";
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr_ptr[i] << " ";
    }
    std::cout << std::endl;
}
```

## Iterator Traits

Iterator traits provide information about iterator types at compile time:

```cpp
#include <iterator>
#include <vector>
#include <list>
#include <iostream>
#include <type_traits>

template<typename Iterator>
void analyze_iterator() {
    using traits = std::iterator_traits<Iterator>;
    
    std::cout << "Iterator analysis:" << std::endl;
    std::cout << "Value type size: " << sizeof(typename traits::value_type) << std::endl;
    
    // Check iterator category
    if (std::is_same_v<typename traits::iterator_category, std::random_access_iterator_tag>) {
        std::cout << "Random access iterator" << std::endl;
    } else if (std::is_same_v<typename traits::iterator_category, std::bidirectional_iterator_tag>) {
        std::cout << "Bidirectional iterator" << std::endl;
    } else if (std::is_same_v<typename traits::iterator_category, std::forward_iterator_tag>) {
        std::cout << "Forward iterator" << std::endl;
    } else if (std::is_same_v<typename traits::iterator_category, std::input_iterator_tag>) {
        std::cout << "Input iterator" << std::endl;
    } else if (std::is_same_v<typename traits::iterator_category, std::output_iterator_tag>) {
        std::cout << "Output iterator" << std::endl;
    }
    std::cout << std::endl;
}

void iterator_traits_examples() {
    std::vector<int> vec = {1, 2, 3};
    std::list<double> lst = {1.1, 2.2, 3.3};
    
    analyze_iterator<std::vector<int>::iterator>();
    analyze_iterator<std::list<double>::iterator>();
}
```

## Custom Iterator Implementation

```cpp
#include <iterator>
#include <iostream>

template<typename T>
class SimpleVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;

public:
    SimpleVector(size_t capacity = 10) 
        : size_(0), capacity_(capacity) {
        data_ = new T[capacity_];
    }
    
    ~SimpleVector() {
        delete[] data_;
    }
    
    void push_back(const T& value) {
        if (size_ < capacity_) {
            data_[size_++] = value;
        }
    }
    
    size_t size() const { return size_; }
    
    // Custom iterator class
    class Iterator {
    private:
        T* ptr_;
        
    public:
        // Iterator traits
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        
        Iterator(T* ptr) : ptr_(ptr) {}
        
        // Dereference
        reference operator*() const { return *ptr_; }
        pointer operator->() const { return ptr_; }
        
        // Increment/Decrement
        Iterator& operator++() { ++ptr_; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++ptr_; return tmp; }
        Iterator& operator--() { --ptr_; return *this; }
        Iterator operator--(int) { Iterator tmp = *this; --ptr_; return tmp; }
        
        // Arithmetic
        Iterator operator+(difference_type n) const { return Iterator(ptr_ + n); }
        Iterator operator-(difference_type n) const { return Iterator(ptr_ - n); }
        Iterator& operator+=(difference_type n) { ptr_ += n; return *this; }
        Iterator& operator-=(difference_type n) { ptr_ -= n; return *this; }
        
        difference_type operator-(const Iterator& other) const { return ptr_ - other.ptr_; }
        
        // Subscript
        reference operator[](difference_type n) const { return ptr_[n]; }
        
        // Comparison
        bool operator==(const Iterator& other) const { return ptr_ == other.ptr_; }
        bool operator!=(const Iterator& other) const { return ptr_ != other.ptr_; }
        bool operator<(const Iterator& other) const { return ptr_ < other.ptr_; }
        bool operator>(const Iterator& other) const { return ptr_ > other.ptr_; }
        bool operator<=(const Iterator& other) const { return ptr_ <= other.ptr_; }
        bool operator>=(const Iterator& other) const { return ptr_ >= other.ptr_; }
    };
    
    Iterator begin() { return Iterator(data_); }
    Iterator end() { return Iterator(data_ + size_); }
};

void custom_iterator_examples() {
    SimpleVector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.push_back(4);
    vec.push_back(5);
    
    std::cout << "Custom vector contents: ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // Range-based for loop works too
    std::cout << "Range-based loop: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // STL algorithms work with our iterator
    auto it = std::find(vec.begin(), vec.end(), 3);
    if (it != vec.end()) {
        std::cout << "Found 3 at position: " << std::distance(vec.begin(), it) << std::endl;
    }
}
```

## Iterator Adapters

### Reverse Iterators
```cpp
#include <vector>
#include <iterator>
#include <iostream>

void reverse_iterator_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    std::cout << "Forward: ";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Reverse: ";
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // Manual reverse iterator
    std::cout << "Manual reverse: ";
    std::reverse_iterator<std::vector<int>::iterator> rev_it(vec.end());
    std::reverse_iterator<std::vector<int>::iterator> rev_end(vec.begin());
    
    for (; rev_it != rev_end; ++rev_it) {
        std::cout << *rev_it << " ";
    }
    std::cout << std::endl;
}
```

### Insert Iterators
```cpp
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>
#include <iostream>

void insert_iterator_examples() {
    std::vector<int> source = {1, 2, 3, 4, 5};
    std::vector<int> dest;
    std::list<int> dest_list;
    
    // Back insert iterator
    std::copy(source.begin(), source.end(), 
              std::back_inserter(dest));
    
    std::cout << "Back insert: ";
    for (const auto& item : dest) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Front insert iterator (for containers that support it)
    std::copy(source.begin(), source.end(), 
              std::front_inserter(dest_list));
    
    std::cout << "Front insert: ";
    for (const auto& item : dest_list) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Insert iterator at specific position
    std::vector<int> dest2 = {10, 20, 30};
    auto insert_pos = dest2.begin() + 1;
    std::copy(source.begin(), source.begin() + 2, 
              std::inserter(dest2, insert_pos));
    
    std::cout << "Insert at position: ";
    for (const auto& item : dest2) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Iterator Invalidation Rules

```cpp
#include <vector>
#include <list>
#include <iostream>

void iterator_invalidation_examples() {
    std::cout << "Vector iterator invalidation:" << std::endl;
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin() + 2; // Points to element 3
    
    std::cout << "Before modification: " << *it << std::endl;
    
    // This may invalidate iterators if reallocation occurs
    vec.push_back(6);
    // it might be invalid now - don't use it!
    
    // Safe approach: recalculate iterator
    it = vec.begin() + 2;
    std::cout << "After push_back: " << *it << std::endl;
    
    std::cout << "\nList iterator invalidation:" << std::endl;
    
    std::list<int> lst = {1, 2, 3, 4, 5};
    auto lst_it = lst.begin();
    std::advance(lst_it, 2); // Points to element 3
    
    std::cout << "Before modification: " << *lst_it << std::endl;
    
    // List iterators remain valid after insertion/deletion
    // (except for the deleted element itself)
    lst.push_back(6);
    lst.push_front(0);
    
    std::cout << "After modifications: " << *lst_it << std::endl; // Still valid
    
    // Demonstrate invalidation when element is erased
    auto to_erase = lst.begin();
    std::advance(to_erase, 1); // Points to element 1
    
    lst.erase(to_erase); // to_erase is now invalid
    std::cout << "After erase: " << *lst_it << std::endl; // lst_it still valid
}
```

## Iterator Utilities

### std::advance
```cpp
#include <iterator>
#include <vector>
#include <list>
#include <iostream>

void advance_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::list<int> lst = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // std::advance moves iterator by n positions
    auto vec_it = vec.begin();
    std::advance(vec_it, 5); // O(1) for random access iterators
    std::cout << "Vector element at position 5: " << *vec_it << std::endl;
    
    auto lst_it = lst.begin();
    std::advance(lst_it, 5); // O(n) for bidirectional iterators
    std::cout << "List element at position 5: " << *lst_it << std::endl;
    
    // Negative advance for bidirectional/random access iterators
    std::advance(vec_it, -2);
    std::cout << "Vector element after moving back 2: " << *vec_it << std::endl;
}
```

### std::distance
```cpp
#include <iterator>
#include <vector>
#include <list>
#include <iostream>

void distance_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::list<int> lst = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // std::distance calculates distance between iterators
    auto vec_begin = vec.begin();
    auto vec_end = vec.end();
    auto vec_middle = vec.begin() + 5;
    
    std::cout << "Vector size: " << std::distance(vec_begin, vec_end) << std::endl;
    std::cout << "Distance to middle: " << std::distance(vec_begin, vec_middle) << std::endl;
    
    auto lst_begin = lst.begin();
    auto lst_end = lst.end();
    auto lst_middle = lst.begin();
    std::advance(lst_middle, 5);
    
    std::cout << "List size: " << std::distance(lst_begin, lst_end) << std::endl;
    std::cout << "Distance to middle: " << std::distance(lst_begin, lst_middle) << std::endl;
}
```

### std::next and std::prev
```cpp
#include <iterator>
#include <vector>
#include <list>
#include <iostream>

void next_prev_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    auto vec_it = vec.begin();
    auto lst_it = lst.begin();
    
    // std::next returns iterator advanced by n positions (doesn't modify original)
    auto vec_next = std::next(vec_it, 3);
    std::cout << "Vector element 3 positions ahead: " << *vec_next << std::endl;
    std::cout << "Original iterator still points to: " << *vec_it << std::endl;
    
    auto lst_next = std::next(lst_it, 2);
    std::cout << "List element 2 positions ahead: " << *lst_next << std::endl;
    
    // std::prev returns iterator moved back by n positions
    auto vec_end = vec.end();
    auto vec_prev = std::prev(vec_end, 2);
    std::cout << "Vector element 2 positions before end: " << *vec_prev << std::endl;
    
    auto lst_end = lst.end();
    auto lst_prev = std::prev(lst_end, 1);
    std::cout << "List last element: " << *lst_prev << std::endl;
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <iterator>
#include <algorithm>

int main() {
    std::cout << "=== Iterator Examples ===" << std::endl;
    
    input_iterator_examples();
    output_iterator_examples();
    forward_iterator_examples();
    bidirectional_iterator_examples();
    random_access_iterator_examples();
    contiguous_iterator_examples();
    
    std::cout << "\n=== Iterator Traits ===" << std::endl;
    iterator_traits_examples();
    
    std::cout << "\n=== Custom Iterator ===" << std::endl;
    custom_iterator_examples();
    
    std::cout << "\n=== Iterator Adapters ===" << std::endl;
    reverse_iterator_examples();
    insert_iterator_examples();
    
    std::cout << "\n=== Iterator Invalidation ===" << std::endl;
    iterator_invalidation_examples();
    
    std::cout << "\n=== Iterator Utilities ===" << std::endl;
    advance_examples();
    distance_examples();
    next_prev_examples();
    
    return 0;
}
```

## Key Concepts Summary

1. **Iterator Categories**: Each category builds upon the previous one's capabilities
2. **Iterator Traits**: Provide compile-time information about iterators
3. **Custom Iterators**: Must implement required operations for their category
4. **Iterator Invalidation**: Different containers have different invalidation rules
5. **Iterator Utilities**: Functions like advance, distance, next, prev make iterator manipulation easier

## Best Practices

1. Use iterator traits for generic programming
2. Prefer range-based for loops when possible
3. Be aware of iterator invalidation rules
4. Use appropriate iterator utilities (advance, distance, etc.)
5. Implement all required operations when creating custom iterators
6. Test custom iterators with STL algorithms

## Exercises

1. Implement a custom iterator for a binary tree (in-order traversal)
2. Create a filter iterator that skips elements not matching a predicate
3. Implement a transform iterator that applies a function to elements on access
4. Write a generic function that works with any iterator category
5. Create a circular iterator that wraps around at container boundaries
