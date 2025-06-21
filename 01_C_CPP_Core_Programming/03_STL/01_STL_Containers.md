# STL Containers

*Part of STL Learning Track - 2 weeks*

## Overview

STL containers are objects that store other objects and provide an interface to access and manipulate the stored data. They manage memory automatically and provide different performance characteristics based on their internal structure.

## Sequence Containers

### std::vector
```cpp
#include <vector>
#include <iostream>

void vector_examples() {
    // Basic operations
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Access elements
    std::cout << "First element: " << vec[0] << std::endl;
    std::cout << "Last element: " << vec.back() << std::endl;
    
    // Add elements
    vec.push_back(6);
    vec.emplace_back(7);
    
    // Iterate
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Size and capacity
    std::cout << "Size: " << vec.size() << std::endl;
    std::cout << "Capacity: " << vec.capacity() << std::endl;
    
    // Reserve memory
    vec.reserve(100);
    
    // Insert at position
    vec.insert(vec.begin() + 2, 99);
    
    // Erase elements
    vec.erase(vec.begin() + 2);
}
```

### std::array
```cpp
#include <array>
#include <algorithm>

void array_examples() {
    // Fixed-size array
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    
    // Access elements
    std::cout << "Size: " << arr.size() << std::endl;
    std::cout << "Max size: " << arr.max_size() << std::endl;
    
    // Fill with value
    std::array<int, 10> arr2;
    arr2.fill(42);
    
    // Sort
    std::sort(arr.begin(), arr.end());
    
    // Range-based loop
    for (const auto& item : arr) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::deque
```cpp
#include <deque>

void deque_examples() {
    std::deque<int> dq = {3, 4, 5};
    
    // Add to front and back
    dq.push_front(2);
    dq.push_back(6);
    dq.push_front(1);
    
    // Access elements
    std::cout << "Front: " << dq.front() << std::endl;
    std::cout << "Back: " << dq.back() << std::endl;
    
    // Remove from front and back
    dq.pop_front();
    dq.pop_back();
    
    // Random access
    std::cout << "Element at index 2: " << dq[2] << std::endl;
    
    // Print all elements
    for (const auto& item : dq) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::list
```cpp
#include <list>

void list_examples() {
    std::list<int> lst = {1, 3, 5, 7, 9};
    
    // Insert at specific position
    auto it = lst.begin();
    std::advance(it, 2);
    lst.insert(it, 4);
    
    // Add to front and back
    lst.push_front(0);
    lst.push_back(10);
    
    // Sort
    lst.sort();
    
    // Remove duplicates
    lst.unique();
    
    // Remove specific value
    lst.remove(5);
    
    // Splice (move elements from another list)
    std::list<int> other = {100, 200};
    lst.splice(lst.end(), other);
    
    // Print all elements
    for (const auto& item : lst) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::forward_list
```cpp
#include <forward_list>

void forward_list_examples() {
    std::forward_list<int> flist = {1, 2, 3, 4, 5};
    
    // Add to front
    flist.push_front(0);
    
    // Insert after position
    auto it = flist.begin();
    flist.insert_after(it, 99);
    
    // Erase after position
    flist.erase_after(flist.begin());
    
    // Sort
    flist.sort();
    
    // Print all elements
    for (const auto& item : flist) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Associative Containers

### std::set and std::multiset
```cpp
#include <set>

void set_examples() {
    // std::set - unique elements, sorted
    std::set<int> s = {5, 2, 8, 1, 9, 2}; // 2 will appear only once
    
    // Insert elements
    s.insert(3);
    s.insert(7);
    
    // Check if element exists
    if (s.find(5) != s.end()) {
        std::cout << "5 found in set" << std::endl;
    }
    
    // Count elements (0 or 1 for set)
    std::cout << "Count of 5: " << s.count(5) << std::endl;
    
    // Range operations
    auto lower = s.lower_bound(3);
    auto upper = s.upper_bound(7);
    
    // Print elements in range
    for (auto it = lower; it != upper; ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // multiset - allows duplicates
    std::multiset<int> ms = {1, 2, 2, 3, 3, 3};
    std::cout << "Count of 3 in multiset: " << ms.count(3) << std::endl;
}
```

### std::map and std::multimap
```cpp
#include <map>

void map_examples() {
    // std::map - key-value pairs, unique keys
    std::map<std::string, int> m = {
        {"apple", 5},
        {"banana", 3},
        {"orange", 8}
    };
    
    // Insert elements
    m["grape"] = 12;
    m.insert(std::make_pair("mango", 7));
    m.emplace("cherry", 4);
    
    // Access elements
    std::cout << "Apples: " << m["apple"] << std::endl;
    
    // Find element
    auto it = m.find("banana");
    if (it != m.end()) {
        std::cout << "Found " << it->first << " with value " << it->second << std::endl;
    }
    
    // Iterate through map
    for (const auto& pair : m) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // multimap - allows duplicate keys
    std::multimap<std::string, int> mm = {
        {"color", 1},
        {"color", 2},
        {"color", 3}
    };
    
    auto range = mm.equal_range("color");
    for (auto it = range.first; it != range.second; ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }
}
```

## Unordered Containers

### std::unordered_set and std::unordered_multiset
```cpp
#include <unordered_set>

void unordered_set_examples() {
    // Hash-based set for O(1) average access
    std::unordered_set<int> us = {1, 2, 3, 4, 5};
    
    // Insert
    us.insert(6);
    us.emplace(7);
    
    // Find
    if (us.find(3) != us.end()) {
        std::cout << "3 found in unordered_set" << std::endl;
    }
    
    // Bucket information
    std::cout << "Bucket count: " << us.bucket_count() << std::endl;
    std::cout << "Load factor: " << us.load_factor() << std::endl;
    
    // Reserve buckets
    us.reserve(100);
    
    // unordered_multiset allows duplicates
    std::unordered_multiset<int> ums = {1, 1, 2, 2, 3, 3};
    std::cout << "Count of 2: " << ums.count(2) << std::endl;
}
```

### std::unordered_map and std::unordered_multimap
```cpp
#include <unordered_map>

void unordered_map_examples() {
    // Hash-based map for O(1) average access
    std::unordered_map<std::string, int> um = {
        {"red", 1},
        {"green", 2},
        {"blue", 3}
    };
    
    // Insert
    um["yellow"] = 4;
    um.emplace("purple", 5);
    
    // Access
    std::cout << "Red value: " << um["red"] << std::endl;
    
    // Custom hash function example
    struct Point {
        int x, y;
        bool operator==(const Point& other) const {
            return x == other.x && y == other.y;
        }
    };
    
    struct PointHash {
        std::size_t operator()(const Point& p) const {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };
    
    std::unordered_map<Point, std::string, PointHash> point_map;
    point_map[{1, 2}] = "Point A";
    point_map[{3, 4}] = "Point B";
}
```

## Container Adaptors

### std::stack
```cpp
#include <stack>

void stack_examples() {
    std::stack<int> stk;
    
    // Push elements
    stk.push(1);
    stk.push(2);
    stk.push(3);
    
    // Access top element
    std::cout << "Top: " << stk.top() << std::endl;
    
    // Pop elements
    while (!stk.empty()) {
        std::cout << stk.top() << " ";
        stk.pop();
    }
    std::cout << std::endl;
    
    // Stack with custom underlying container
    std::stack<int, std::deque<int>> deque_stack;
    std::stack<int, std::list<int>> list_stack;
}
```

### std::queue
```cpp
#include <queue>

void queue_examples() {
    std::queue<int> q;
    
    // Push elements
    q.push(1);
    q.push(2);
    q.push(3);
    
    // Access front and back
    std::cout << "Front: " << q.front() << std::endl;
    std::cout << "Back: " << q.back() << std::endl;
    
    // Pop elements
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    std::cout << std::endl;
}
```

### std::priority_queue
```cpp
#include <queue>
#include <vector>
#include <functional>

void priority_queue_examples() {
    // Max heap by default
    std::priority_queue<int> pq;
    
    pq.push(3);
    pq.push(1);
    pq.push(4);
    pq.push(2);
    
    std::cout << "Max heap: ";
    while (!pq.empty()) {
        std::cout << pq.top() << " ";
        pq.pop();
    }
    std::cout << std::endl;
    
    // Min heap
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    
    min_pq.push(3);
    min_pq.push(1);
    min_pq.push(4);
    min_pq.push(2);
    
    std::cout << "Min heap: ";
    while (!min_pq.empty()) {
        std::cout << min_pq.top() << " ";
        min_pq.pop();
    }
    std::cout << std::endl;
    
    // Custom comparator
    struct Person {
        std::string name;
        int age;
    };
    
    auto cmp = [](const Person& a, const Person& b) {
        return a.age > b.age; // Min heap by age
    };
    
    std::priority_queue<Person, std::vector<Person>, decltype(cmp)> person_pq(cmp);
    
    person_pq.push({"Alice", 30});
    person_pq.push({"Bob", 25});
    person_pq.push({"Charlie", 35});
    
    while (!person_pq.empty()) {
        const auto& p = person_pq.top();
        std::cout << p.name << " (" << p.age << ") ";
        person_pq.pop();
    }
    std::cout << std::endl;
}
```

## Container Selection Guidelines

### When to use each container:

1. **std::vector**: Default choice for sequence containers. Good cache locality, random access.
2. **std::array**: When size is known at compile time and you need stack allocation.
3. **std::deque**: When you need efficient insertion/deletion at both ends.
4. **std::list**: When you need frequent insertion/deletion in the middle.
5. **std::forward_list**: When memory is critical and you only need forward iteration.
6. **std::set/std::map**: When you need sorted, unique elements with logarithmic search.
7. **std::unordered_set/std::unordered_map**: When you need fast average-case lookup (O(1)).
8. **std::multiset/std::multimap**: When you need sorted containers with duplicate keys.
9. **std::stack**: LIFO operations.
10. **std::queue**: FIFO operations.
11. **std::priority_queue**: When you need a heap-based priority system.

## Performance Characteristics

| Container | Access | Insert/Delete (end) | Insert/Delete (middle) | Find |
|-----------|--------|-------------------|----------------------|------|
| vector | O(1) | O(1) amortized | O(n) | O(n) |
| array | O(1) | N/A | N/A | O(n) |
| deque | O(1) | O(1) | O(n) | O(n) |
| list | O(n) | O(1) | O(1) | O(n) |
| forward_list | O(n) | O(1) | O(1) | O(n) |
| set/map | O(log n) | O(log n) | O(log n) | O(log n) |
| unordered_set/map | O(1) avg | O(1) avg | O(1) avg | O(1) avg |

## Complete Example Program

```cpp
#include <iostream>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>

int main() {
    // Demonstrate container usage
    vector_examples();
    array_examples();
    deque_examples();
    list_examples();
    forward_list_examples();
    set_examples();
    map_examples();
    unordered_set_examples();
    unordered_map_examples();
    stack_examples();
    queue_examples();
    priority_queue_examples();
    
    return 0;
}
```

## Exercises

1. Implement a function that takes a vector and returns the most frequent element.
2. Create a program that uses std::map to count word frequencies in a text.
3. Implement a LRU cache using std::list and std::unordered_map.
4. Compare performance of std::set vs std::unordered_set for different operations.
5. Implement a task scheduler using std::priority_queue.
