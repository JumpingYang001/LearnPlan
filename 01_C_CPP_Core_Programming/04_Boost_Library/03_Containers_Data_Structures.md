# Containers and Data Structures

*Duration: 2 weeks*

## Overview

This section covers Boost's advanced container libraries that provide specialized data structures not available in the standard library.

## Learning Topics

### Boost.Container
- flat_map, flat_set - Sorted vector-based containers
- small_vector - Small object optimization
- static_vector - Fixed capacity vector
- stable_vector - Vector with stable iterators

### Boost.MultiIndex
- Multi-indexed containers for complex data access patterns
- Index types: ordered, hashed, sequenced, random access
- Custom indices and composite keys
- Query operations and iteration

### Boost.Bimap
- Bidirectional maps for one-to-one relationships
- Different relation types
- Custom relations and constraints
- Efficient bidirectional lookups

### Boost.CircularBuffer
- Fixed-size circular buffer implementation
- Overwrite policies and capacity management
- Applications in streaming data and ring buffers

## Code Examples

### Boost.Container - flat_map Example
```cpp
#include <boost/container/flat_map.hpp>
#include <iostream>
#include <string>

void demonstrate_flat_map() {
    boost::container::flat_map<int, std::string> fmap;
    
    // Insert elements
    fmap[1] = "One";
    fmap[3] = "Three";
    fmap[2] = "Two";
    fmap[5] = "Five";
    fmap[4] = "Four";
    
    std::cout << "flat_map contents:\n";
    for (const auto& pair : fmap) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }
    
    // Demonstrate memory layout benefits
    std::cout << "Memory is contiguous, cache-friendly\n";
    
    // Reserve capacity for better performance
    boost::container::flat_map<int, std::string> optimized_fmap;
    optimized_fmap.reserve(100);
}
```

### small_vector Example
```cpp
#include <boost/container/small_vector.hpp>
#include <iostream>

void demonstrate_small_vector() {
    // small_vector with inline storage for 8 elements
    boost::container::small_vector<int, 8> svec;
    
    std::cout << "Initial capacity: " << svec.capacity() << "\n";
    
    // Add elements that fit in inline storage
    for (int i = 0; i < 8; ++i) {
        svec.push_back(i);
    }
    
    std::cout << "After 8 elements - capacity: " << svec.capacity() << "\n";
    std::cout << "No heap allocation yet!\n";
    
    // Add one more element - triggers heap allocation
    svec.push_back(8);
    std::cout << "After 9 elements - capacity: " << svec.capacity() << "\n";
    std::cout << "Now using heap allocation\n";
    
    // Print elements
    for (int val : svec) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}
```

### Boost.MultiIndex Example
```cpp
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <iostream>
#include <string>

struct Employee {
    int id;
    std::string name;
    std::string department;
    double salary;
    
    Employee(int id, const std::string& name, 
             const std::string& dept, double salary)
        : id(id), name(name), department(dept), salary(salary) {}
};

// Define multi-index container
namespace bmi = boost::multi_index;

typedef bmi::multi_index_container<
    Employee,
    bmi::indexed_by<
        // Index by ID (unique)
        bmi::ordered_unique<
            bmi::member<Employee, int, &Employee::id>
        >,
        // Index by name (non-unique)
        bmi::ordered_non_unique<
            bmi::member<Employee, std::string, &Employee::name>
        >,
        // Hash index by department
        bmi::hashed_non_unique<
            bmi::member<Employee, std::string, &Employee::department>
        >,
        // Composite index by department and salary
        bmi::ordered_non_unique<
            bmi::composite_key<
                Employee,
                bmi::member<Employee, std::string, &Employee::department>,
                bmi::member<Employee, double, &Employee::salary>
            >
        >
    >
> EmployeeContainer;

void demonstrate_multi_index() {
    EmployeeContainer employees;
    
    // Add employees
    employees.insert(Employee(1, "Alice", "Engineering", 80000));
    employees.insert(Employee(2, "Bob", "Marketing", 60000));
    employees.insert(Employee(3, "Charlie", "Engineering", 90000));
    employees.insert(Employee(4, "Diana", "HR", 55000));
    
    // Access by ID (index 0)
    auto& id_index = employees.get<0>();
    auto it = id_index.find(2);
    if (it != id_index.end()) {
        std::cout << "Employee 2: " << it->name << "\n";
    }
    
    // Access by name (index 1)
    auto& name_index = employees.get<1>();
    auto name_range = name_index.equal_range("Alice");
    for (auto it = name_range.first; it != name_range.second; ++it) {
        std::cout << "Found Alice: ID " << it->id << "\n";
    }
    
    // Access by department (index 2)
    auto& dept_index = employees.get<2>();
    auto dept_range = dept_index.equal_range("Engineering");
    std::cout << "Engineering employees:\n";
    for (auto it = dept_range.first; it != dept_range.second; ++it) {
        std::cout << "  " << it->name << " - $" << it->salary << "\n";
    }
    
    // Access by composite key (index 3)
    auto& composite_index = employees.get<3>();
    auto comp_it = composite_index.lower_bound(
        boost::make_tuple("Engineering", 85000.0)
    );
    std::cout << "Engineering employees with salary >= $85000:\n";
    for (; comp_it != composite_index.end() && 
           comp_it->department == "Engineering"; ++comp_it) {
        std::cout << "  " << comp_it->name << " - $" << comp_it->salary << "\n";
    }
}
```

### Boost.Bimap Example
```cpp
#include <boost/bimap.hpp>
#include <iostream>
#include <string>

void demonstrate_bimap() {
    typedef boost::bimap<int, std::string> employee_bimap;
    employee_bimap employees;
    
    // Insert mappings
    employees.insert({1, "Alice"});
    employees.insert({2, "Bob"});
    employees.insert({3, "Charlie"});
    
    // Forward lookup (ID -> Name)
    std::cout << "Employee 2: " << employees.left.at(2) << "\n";
    
    // Reverse lookup (Name -> ID)
    std::cout << "Alice's ID: " << employees.right.at("Alice") << "\n";
    
    // Iterate through left view (ID -> Name)
    std::cout << "All employees (by ID):\n";
    for (const auto& pair : employees.left) {
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
    }
    
    // Iterate through right view (Name -> ID)
    std::cout << "All employees (by name):\n";
    for (const auto& pair : employees.right) {
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
    }
    
    // Check if mapping exists
    if (employees.left.count(4) == 0) {
        std::cout << "Employee 4 not found\n";
    }
}
```

### Boost.CircularBuffer Example
```cpp
#include <boost/circular_buffer.hpp>
#include <iostream>
#include <numeric>

void demonstrate_circular_buffer() {
    // Create circular buffer with capacity of 5
    boost::circular_buffer<int> cb(5);
    
    // Fill the buffer
    for (int i = 1; i <= 7; ++i) {
        cb.push_back(i);
        std::cout << "Added " << i << ", buffer: ";
        for (int val : cb) {
            std::cout << val << " ";
        }
        std::cout << "(size: " << cb.size() << ")\n";
    }
    
    std::cout << "\nBuffer is full: " << std::boolalpha << cb.full() << "\n";
    
    // Calculate running average
    double avg = std::accumulate(cb.begin(), cb.end(), 0.0) / cb.size();
    std::cout << "Average of last " << cb.size() << " values: " << avg << "\n";
    
    // Demonstrate as a sliding window
    std::cout << "\nSliding window demo:\n";
    boost::circular_buffer<double> window(3);
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    for (double value : data) {
        window.push_back(value);
        if (window.full()) {
            double window_avg = std::accumulate(window.begin(), window.end(), 0.0) / window.size();
            std::cout << "Value: " << value << ", 3-point avg: " << window_avg << "\n";
        }
    }
}
```

### Advanced MultiIndex with Custom Predicates
```cpp
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <iostream>
#include <string>

class Product {
public:
    Product(int id, const std::string& name, double price, const std::string& category)
        : id_(id), name_(name), price_(price), category_(category) {}
    
    int getId() const { return id_; }
    const std::string& getName() const { return name_; }
    double getPrice() const { return price_; }
    const std::string& getCategory() const { return category_; }
    
private:
    int id_;
    std::string name_;
    double price_;
    std::string category_;
};

// Custom comparator for price ranges
struct PriceRange {
    bool operator()(const Product& p1, const Product& p2) const {
        return getPriceRange(p1.getPrice()) < getPriceRange(p2.getPrice());
    }
    
private:
    int getPriceRange(double price) const {
        if (price < 10.0) return 0;      // Budget
        if (price < 50.0) return 1;      // Mid-range
        return 2;                        // Premium
    }
};

typedef bmi::multi_index_container<
    Product,
    bmi::indexed_by<
        bmi::ordered_unique<bmi::const_mem_fun<Product, int, &Product::getId>>,
        bmi::ordered_non_unique<bmi::const_mem_fun<Product, double, &Product::getPrice>>,
        bmi::ordered_non_unique<PriceRange>
    >
> ProductCatalog;

void demonstrate_advanced_multi_index() {
    ProductCatalog catalog;
    
    catalog.insert(Product(1, "Widget A", 5.99, "Tools"));
    catalog.insert(Product(2, "Widget B", 25.50, "Tools"));
    catalog.insert(Product(3, "Gadget X", 75.00, "Electronics"));
    catalog.insert(Product(4, "Gadget Y", 8.25, "Electronics"));
    
    // Query by price range using custom comparator (index 2)
    auto& price_range_index = catalog.get<2>();
    std::cout << "Products by price range:\n";
    
    for (const auto& product : price_range_index) {
        std::string range;
        double price = product.getPrice();
        if (price < 10.0) range = "Budget";
        else if (price < 50.0) range = "Mid-range";
        else range = "Premium";
        
        std::cout << "  " << product.getName() << " ($" << price 
                  << ") - " << range << "\n";
    }
}
```

## Practical Exercises

1. **Performance Comparison**
   - Compare flat_map vs std::map performance
   - Measure small_vector vs std::vector for small sizes
   - Benchmark multi_index vs multiple separate containers

2. **Database Simulation**
   - Create a student database with multi_index
   - Support queries by ID, name, GPA, and major
   - Implement range queries and complex filters

3. **Streaming Data Buffer**
   - Implement a real-time data processor using circular_buffer
   - Calculate moving averages and detect trends
   - Handle different buffer sizes and update frequencies

4. **Bidirectional Mapping System**
   - Create a translation system using bimap
   - Support multiple language pairs
   - Implement efficient reverse lookups

## Performance Considerations

### flat_map vs std::map
- Better cache locality for flat_map
- Slower insertion/deletion for flat_map
- Choose based on read/write ratio

### small_vector Optimization
- Eliminate heap allocations for small collections
- Choose inline capacity based on typical usage
- Consider alignment and padding overhead

### MultiIndex Trade-offs
- Memory overhead of multiple indices
- Update cost when modifying indexed fields
- Query performance benefits

## Best Practices

1. **Container Selection**
   - Choose containers based on access patterns
   - Consider memory layout and cache effects
   - Profile actual usage scenarios

2. **Index Design**
   - Design indices to match query patterns
   - Avoid over-indexing for write-heavy workloads
   - Use composite keys for complex queries

3. **Memory Management**
   - Reserve capacity when size is predictable
   - Use small_vector for frequent small collections
   - Consider stable_vector for iterator stability

## Assessment

- Can select appropriate containers for specific use cases
- Understands performance trade-offs between container types
- Can design efficient multi-index schemas
- Implements circular buffers for streaming data scenarios

## Next Steps

Move on to [String Processing and Text Handling](04_String_Processing_Text_Handling.md) to explore Boost's text processing capabilities.
