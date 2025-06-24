# Containers and Data Structures

*Duration: 2 weeks*

## Overview

This section covers Boost's advanced container libraries that provide specialized data structures not available in the standard library. These containers offer unique performance characteristics, specialized use cases, and advanced features that complement the STL containers.

### Why Boost Containers?

While the C++ Standard Library provides essential containers like `vector`, `map`, and `set`, Boost containers fill specific gaps:

- **Performance optimizations** for specific use cases
- **Memory-efficient alternatives** for small collections
- **Specialized data structures** for complex access patterns
- **Enhanced functionality** beyond standard containers

### Container Categories

| Category | Containers | Primary Use Case |
|----------|------------|------------------|
| **Optimized Variants** | `flat_map`, `flat_set`, `small_vector` | Better performance for specific scenarios |
| **Specialized Structures** | `circular_buffer`, `stable_vector` | Unique data access patterns |
| **Multi-Access** | `multi_index_container` | Complex querying requirements |
| **Bidirectional** | `bimap` | Two-way lookups |

### Memory Layout Comparison

```
Standard Containers:
std::map<int, string>
┌─────┐    ┌─────┐    ┌─────┐
│Node1│───>│Node2│───>│Node3│  (scattered in memory)
└─────┘    └─────┘    └─────┘

Boost Containers:
flat_map<int, string>
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ k1  │ v1  │ k2  │ v2  │ k3  │ v3  │  (contiguous memory)
└─────┴─────┴─────┴─────┴─────┴─────┘
```

## Learning Topics

### Boost.Container

#### flat_map and flat_set - Sorted Vector-Based Containers

**Concept**: Instead of using tree-based storage (like `std::map`), these containers store elements in sorted vectors, providing better cache locality and memory efficiency.

**Key Advantages:**
- **Cache-friendly**: Contiguous memory layout
- **Memory efficient**: No node overhead
- **Fast iteration**: Linear memory access
- **Better for read-heavy workloads**

**Trade-offs:**
- **Slower insertion/deletion**: O(n) vs O(log n)
- **Higher memory requirements during insertion**

```cpp
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <map>
#include <chrono>
#include <random>
#include <iostream>

// Performance comparison demonstration
void compare_map_performance() {
    const int NUM_ELEMENTS = 10000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100000);
    
    // Generate test data
    std::vector<std::pair<int, std::string>> test_data;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        test_data.emplace_back(dis(gen), "value_" + std::to_string(i));
    }
    
    // Test std::map
    auto start = std::chrono::high_resolution_clock::now();
    std::map<int, std::string> std_map;
    for (const auto& pair : test_data) {
        std_map.insert(pair);
    }
    
    // Benchmark lookups
    int lookup_count = 0;
    for (int i = 0; i < 1000; ++i) {
        auto it = std_map.find(dis(gen));
        if (it != std_map.end()) lookup_count++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto std_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test boost::container::flat_map
    start = std::chrono::high_resolution_clock::now();
    boost::container::flat_map<int, std::string> flat_map;
    flat_map.reserve(NUM_ELEMENTS); // Important optimization!
    
    for (const auto& pair : test_data) {
        flat_map.insert(pair);
    }
    
    // Benchmark lookups
    lookup_count = 0;
    for (int i = 0; i < 1000; ++i) {
        auto it = flat_map.find(dis(gen));
        if (it != flat_map.end()) lookup_count++;
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto flat_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "std::map time: " << std_time.count() << " μs\n";
    std::cout << "flat_map time: " << flat_time.count() << " μs\n";
    std::cout << "Memory usage - std::map: ~" << (std_map.size() * (sizeof(std::pair<int, std::string>) + 3 * sizeof(void*))) << " bytes\n";
    std::cout << "Memory usage - flat_map: ~" << (flat_map.size() * sizeof(std::pair<int, std::string>)) << " bytes\n";
}

// Practical use case: Configuration cache
class ConfigCache {
private:
    boost::container::flat_map<std::string, std::string> cache_;
    
public:
    ConfigCache() {
        cache_.reserve(100); // Expected configuration size
    }
    
    void load_config(const std::string& filename) {
        // Simulate loading configuration
        cache_["database.host"] = "localhost";
        cache_["database.port"] = "5432";
        cache_["cache.size"] = "1000";
        cache_["log.level"] = "INFO";
        // ... more config entries
        
        // Sort once after loading for optimal lookup performance
        // flat_map automatically maintains sorted order
    }
    
    std::string get_config(const std::string& key) const {
        auto it = cache_.find(key);
        return (it != cache_.end()) ? it->second : "";
    }
    
    void print_all_configs() const {
        std::cout << "Configuration entries:\n";
        for (const auto& [key, value] : cache_) {
            std::cout << "  " << key << " = " << value << "\n";
        }
    }
};
```

#### small_vector - Small Object Optimization

**Concept**: Stores small number of elements inline (on stack) to avoid heap allocation, falling back to heap allocation for larger sizes.

**When to use:**
- Collections that are usually small (< 10-20 elements)
- Frequent creation/destruction of small collections
- Memory allocation is expensive
- Cache locality is critical

```cpp
#include <boost/container/small_vector.hpp>
#include <vector>
#include <chrono>
#include <iostream>

// Demonstrate small_vector benefits
template<typename VectorType>
void benchmark_small_collections(const std::string& name) {
    const int ITERATIONS = 100000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        VectorType vec;
        
        // Typical small collection operations
        for (int i = 0; i < 5; ++i) {
            vec.push_back(i * iter);
        }
        
        // Some processing
        int sum = 0;
        for (const auto& val : vec) {
            sum += val;
        }
        
        // vec automatically destroyed here
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << " time: " << duration.count() << " μs\n";
}

void demonstrate_small_vector_optimization() {
    std::cout << "Benchmarking small collections:\n";
    
    // Compare different vector types
    benchmark_small_collections<std::vector<int>>("std::vector");
    benchmark_small_collections<boost::container::small_vector<int, 8>>("small_vector<8>");
    benchmark_small_collections<boost::container::small_vector<int, 16>>("small_vector<16>");
}

// Real-world example: Function call argument lists
class FunctionCall {
private:
    std::string function_name_;
    boost::container::small_vector<std::string, 4> arguments_; // Most functions have <= 4 args
    
public:
    FunctionCall(const std::string& name) : function_name_(name) {}
    
    void add_argument(const std::string& arg) {
        arguments_.push_back(arg);
    }
    
    void execute() {
        std::cout << "Calling " << function_name_ << "(";
        for (size_t i = 0; i < arguments_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << arguments_[i];
        }
        std::cout << ")\n";
        
        // Arguments stored inline for typical case (no heap allocation)
        std::cout << "Arguments capacity: " << arguments_.capacity() << "\n";
        std::cout << "Using heap: " << (arguments_.size() > 4 ? "Yes" : "No") << "\n";
    }
};
```

#### static_vector - Fixed Capacity Vector

**Concept**: Vector with compile-time fixed capacity, never allocates memory dynamically.

**Use cases:**
- Embedded systems with strict memory constraints
- Real-time systems where allocation is forbidden
- Stack-based containers with known maximum size

```cpp
#include <boost/container/static_vector.hpp>
#include <iostream>
#include <stdexcept>

// Embedded system buffer example
template<size_t MaxSize>
class RingBuffer {
private:
    boost::container::static_vector<int, MaxSize> buffer_;
    size_t head_ = 0;
    
public:
    bool push(int value) {
        if (buffer_.size() < MaxSize) {
            buffer_.push_back(value);
            return true;
        } else {
            // Overwrite oldest element
            buffer_[head_] = value;
            head_ = (head_ + 1) % MaxSize;
            return false; // Indicate overwrite occurred
        }
    }
    
    void print_buffer() const {
        std::cout << "Buffer contents: ";
        for (const auto& val : buffer_) {
            std::cout << val << " ";
        }
        std::cout << "(size: " << buffer_.size() << "/" << MaxSize << ")\n";
    }
    
    // Guaranteed no heap allocation - safe for real-time systems
    constexpr size_t max_size() const noexcept { return MaxSize; }
    size_t size() const noexcept { return buffer_.size(); }
    bool empty() const noexcept { return buffer_.empty(); }
};

void demonstrate_static_vector() {
    RingBuffer<5> buffer;
    
    std::cout << "Adding elements to static ring buffer:\n";
    for (int i = 1; i <= 8; ++i) {
        bool no_overwrite = buffer.push(i * 10);
        std::cout << "Added " << (i * 10) << (no_overwrite ? "" : " (overwrote)") << "\n";
        buffer.print_buffer();
    }
}
```

#### stable_vector - Vector with Stable Iterators

**Concept**: Provides vector-like interface but with stable iterators that don't invalidate on insertion/deletion.

**Key feature**: Elements are stored in separate chunks, so iterators remain valid even after modifications.

```cpp
#include <boost/container/stable_vector.hpp>
#include <iostream>
#include <vector>

// Demonstrate iterator stability
void demonstrate_stable_vector() {
    boost::container::stable_vector<std::string> stable_vec;
    std::vector<std::string> regular_vec;
    
    // Add initial elements
    stable_vec.push_back("First");
    stable_vec.push_back("Second");
    stable_vec.push_back("Third");
    
    regular_vec.push_back("First");
    regular_vec.push_back("Second");
    regular_vec.push_back("Third");
    
    // Get iterators to second element
    auto stable_it = stable_vec.begin() + 1;
    auto regular_it = regular_vec.begin() + 1;
    
    std::cout << "Before insertion:\n";
    std::cout << "Stable vector element: " << *stable_it << "\n";
    std::cout << "Regular vector element: " << *regular_it << "\n";
    
    // Insert at beginning - this may invalidate regular vector iterator
    stable_vec.insert(stable_vec.begin(), "New First");
    regular_vec.insert(regular_vec.begin(), "New First");
    
    std::cout << "\nAfter insertion at beginning:\n";
    std::cout << "Stable vector element: " << *stable_it << " (still valid!)\n";
    // std::cout << "Regular vector element: " << *regular_it << " (UNDEFINED BEHAVIOR!)\n";
    
    // Safe to use stable_it, but regular_it is invalidated
    std::cout << "Stable vector contents: ";
    for (const auto& elem : stable_vec) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

// Use case: Observer pattern with stable references
class EventManager {
private:
    struct EventHandler {
        std::string name;
        std::function<void()> handler;
        
        EventHandler(const std::string& n, std::function<void()> h) 
            : name(n), handler(h) {}
    };
    
    boost::container::stable_vector<EventHandler> handlers_;
    
public:
    // Returns stable iterator that can be stored for later removal
    auto register_handler(const std::string& name, std::function<void()> handler) {
        handlers_.emplace_back(name, handler);
        return handlers_.end() - 1; // This iterator remains valid!
    }
    
    void remove_handler(decltype(handlers_.begin()) it) {
        handlers_.erase(it); // Other iterators remain valid
    }
    
    void trigger_event() {
        std::cout << "Triggering event to " << handlers_.size() << " handlers:\n";
        for (const auto& handler : handlers_) {
            std::cout << "  Calling " << handler.name << "\n";
            handler.handler();
        }
    }
};
```

### Boost.MultiIndex

**Concept**: A single container that maintains multiple indices simultaneously, allowing efficient access through different keys and access patterns.

**Think of it as**: A database table with multiple indexes - you can quickly find records by ID, name, date, or any combination of fields.

#### Index Types Available

| Index Type | Description | Access Pattern | Use Case |
|------------|-------------|----------------|----------|
| **ordered_unique** | Sorted, unique keys | `O(log n)` search | Primary keys, unique identifiers |
| **ordered_non_unique** | Sorted, duplicate keys allowed | `O(log n)` search | Secondary indices, categories |
| **hashed_unique** | Hash table, unique keys | `O(1)` average search | Fast unique lookups |
| **hashed_non_unique** | Hash table, duplicates allowed | `O(1)` average search | Fast non-unique lookups |
| **sequenced** | Insertion order maintained | Linear access | Timeline, history |
| **random_access** | Vector-like access | `O(1)` by position | Indexed access |

#### Multi-indexed Container Architecture

```
Employee Container with Multiple Indices:

Index 0 (by ID - ordered_unique):
   1 → Alice   2 → Bob   3 → Charlie
   
Index 1 (by Name - ordered_non_unique):
   Alice → Emp1   Bob → Emp2   Charlie → Emp3
   
Index 2 (by Dept - hashed_non_unique):
   "Engineering" → {Emp1, Emp3}
   "Marketing" → {Emp2}
   
Index 3 (by Dept+Salary - composite):
   ("Engineering", 80000) → Emp1
   ("Engineering", 90000) → Emp3
   ("Marketing", 60000) → Emp2
```

#### Comprehensive MultiIndex Example

```cpp
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <iostream>
#include <string>
#include <chrono>

// Enhanced Employee class with more realistic data
class Employee {
public:
    Employee(int id, const std::string& name, const std::string& dept, 
             double salary, const std::string& email, 
             const std::chrono::system_clock::time_point& hire_date)
        : id_(id), name_(name), department_(dept), salary_(salary), 
          email_(email), hire_date_(hire_date) {}
    
    // Accessors
    int getId() const { return id_; }
    const std::string& getName() const { return name_; }
    const std::string& getDepartment() const { return department_; }
    double getSalary() const { return salary_; }
    const std::string& getEmail() const { return email_; }
    const std::chrono::system_clock::time_point& getHireDate() const { return hire_date_; }
    
    // Computed properties
    int getYearsOfService() const {
        auto now = std::chrono::system_clock::now();
        auto duration = now - hire_date_;
        return std::chrono::duration_cast<std::chrono::hours>(duration).count() / (24 * 365);
    }
    
    std::string getSalaryBand() const {
        if (salary_ < 50000) return "Junior";
        if (salary_ < 80000) return "Mid";
        if (salary_ < 120000) return "Senior";
        return "Executive";
    }
    
    void print() const {
        std::cout << "ID: " << id_ << ", Name: " << name_ 
                  << ", Dept: " << department_ << ", Salary: $" << salary_ 
                  << ", Band: " << getSalaryBand() << "\n";
    }

private:
    int id_;
    std::string name_;
    std::string department_;
    double salary_;
    std::string email_;
    std::chrono::system_clock::time_point hire_date_;
};

// Global function for salary band index
std::string get_salary_band(const Employee& emp) {
    return emp.getSalaryBand();
}

// Define the multi-index container with comprehensive indices
namespace bmi = boost::multi_index;

typedef bmi::multi_index_container<
    Employee,
    bmi::indexed_by<
        // Index 0: Primary key (ID) - ordered, unique
        bmi::ordered_unique<
            bmi::tag<struct by_id>,
            bmi::const_mem_fun<Employee, int, &Employee::getId>
        >,
        
        // Index 1: By name - ordered, non-unique (handle duplicates)
        bmi::ordered_non_unique<
            bmi::tag<struct by_name>,
            bmi::const_mem_fun<Employee, const std::string&, &Employee::getName>
        >,
        
        // Index 2: By department - hashed, non-unique (fast department queries)
        bmi::hashed_non_unique<
            bmi::tag<struct by_department>,
            bmi::const_mem_fun<Employee, const std::string&, &Employee::getDepartment>
        >,
        
        // Index 3: By salary - ordered, non-unique (salary range queries)
        bmi::ordered_non_unique<
            bmi::tag<struct by_salary>,
            bmi::const_mem_fun<Employee, double, &Employee::getSalary>
        >,
        
        // Index 4: Composite key (department + salary) for complex queries
        bmi::ordered_non_unique<
            bmi::tag<struct by_dept_salary>,
            bmi::composite_key<
                Employee,
                bmi::const_mem_fun<Employee, const std::string&, &Employee::getDepartment>,
                bmi::const_mem_fun<Employee, double, &Employee::getSalary>
            >
        >,
        
        // Index 5: By salary band using global function
        bmi::ordered_non_unique<
            bmi::tag<struct by_salary_band>,
            bmi::global_fun<const Employee&, std::string, get_salary_band>
        >,
        
        // Index 6: Insertion order (for audit trails, history)
        bmi::sequenced<bmi::tag<struct by_insertion_order>>,
        
        // Index 7: Random access by position (for pagination)
        bmi::random_access<bmi::tag<struct by_position>>
    >
> EmployeeDatabase;

class HRSystem {
private:
    EmployeeDatabase employees_;
    
public:
    void addEmployee(const Employee& emp) {
        auto result = employees_.insert(emp);
        if (result.second) {
            std::cout << "Added employee: " << emp.getName() << "\n";
        } else {
            std::cout << "Employee with ID " << emp.getId() << " already exists!\n";
        }
    }
    
    // Query by ID (fastest - O(log n) with ordered index)
    void findById(int id) {
        auto& id_index = employees_.get<by_id>();
        auto it = id_index.find(id);
        
        if (it != id_index.end()) {
            std::cout << "Found employee by ID:\n";
            it->print();
        } else {
            std::cout << "Employee with ID " << id << " not found.\n";
        }
    }
    
    // Query by department (fast - O(1) average with hash index)
    void findByDepartment(const std::string& dept) {
        auto& dept_index = employees_.get<by_department>();
        auto range = dept_index.equal_range(dept);
        
        std::cout << "Employees in " << dept << " department:\n";
        for (auto it = range.first; it != range.second; ++it) {
            std::cout << "  ";
            it->print();
        }
    }
    
    // Salary range queries (efficient with ordered salary index)
    void findBySalaryRange(double min_salary, double max_salary) {
        auto& salary_index = employees_.get<by_salary>();
        auto lower = salary_index.lower_bound(min_salary);
        auto upper = salary_index.upper_bound(max_salary);
        
        std::cout << "Employees with salary between $" << min_salary 
                  << " and $" << max_salary << ":\n";
        for (auto it = lower; it != upper; ++it) {
            std::cout << "  ";
            it->print();
        }
    }
    
    // Complex query using composite index
    void findByDeptAndMinSalary(const std::string& dept, double min_salary) {
        auto& composite_index = employees_.get<by_dept_salary>();
        
        // Find all employees in department with salary >= min_salary
        auto lower = composite_index.lower_bound(boost::make_tuple(dept, min_salary));
        auto upper = composite_index.upper_bound(boost::make_tuple(dept, 
                                                std::numeric_limits<double>::max()));
        
        std::cout << "Employees in " << dept << " with salary >= $" << min_salary << ":\n";
        for (auto it = lower; it != upper; ++it) {
            std::cout << "  ";
            it->print();
        }
    }
    
    // Query by salary band
    void findBySalaryBand(const std::string& band) {
        auto& band_index = employees_.get<by_salary_band>();
        auto range = band_index.equal_range(band);
        
        std::cout << "Employees in " << band << " salary band:\n";
        for (auto it = range.first; it != range.second; ++it) {
            std::cout << "  ";
            it->print();
        }
    }
    
    // Show hiring timeline (using sequenced index)
    void showHiringTimeline() {
        auto& timeline_index = employees_.get<by_insertion_order>();
        
        std::cout << "Employee hiring timeline:\n";
        int position = 1;
        for (const auto& emp : timeline_index) {
            std::cout << position++ << ". " << emp.getName() 
                      << " (" << emp.getDepartment() << ")\n";
        }
    }
    
    // Pagination using random access index
    void showPage(size_t page_number, size_t page_size) {
        auto& pos_index = employees_.get<by_position>();
        
        size_t start = page_number * page_size;
        size_t end = std::min(start + page_size, pos_index.size());
        
        if (start >= pos_index.size()) {
            std::cout << "Page " << page_number << " is out of range.\n";
            return;
        }
        
        std::cout << "Page " << page_number + 1 << " (employees " 
                  << start + 1 << "-" << end << "):\n";
        
        for (size_t i = start; i < end; ++i) {
            std::cout << "  " << (i + 1) << ". ";
            pos_index[i].print();
        }
    }
    
    // Update employee (complex operation - affects multiple indices)
    void updateSalary(int id, double new_salary) {
        auto& id_index = employees_.get<by_id>();
        auto it = id_index.find(id);
        
        if (it != id_index.end()) {
            // Multi-index allows modification through modify() function
            bool success = id_index.modify(it, [new_salary](Employee& emp) {
                // This lambda modifies the employee
                // Multi-index will automatically update all affected indices
                const_cast<double&>(emp.getSalary()) = new_salary; // Hack for demo
            });
            
            if (success) {
                std::cout << "Updated salary for " << it->getName() 
                          << " to $" << new_salary << "\n";
            } else {
                std::cout << "Failed to update salary (index consistency error)\n";
            }
        } else {
            std::cout << "Employee with ID " << id << " not found.\n";
        }
    }
    
    void printStatistics() {
        std::cout << "\n=== HR System Statistics ===\n";
        std::cout << "Total employees: " << employees_.size() << "\n";
        
        // Department statistics using hashed index
        auto& dept_index = employees_.get<by_department>();
        std::cout << "Departments: ";
        std::set<std::string> unique_depts;
        for (const auto& emp : dept_index) {
            unique_depts.insert(emp.getDepartment());
        }
        for (const auto& dept : unique_depts) {
            auto range = dept_index.equal_range(dept);
            std::cout << dept << "(" << std::distance(range.first, range.second) << ") ";
        }
        std::cout << "\n";
        
        // Salary statistics using ordered salary index
        auto& salary_index = employees_.get<by_salary>();
        if (!salary_index.empty()) {
            std::cout << "Salary range: $" << salary_index.begin()->getSalary() 
                      << " - $" << salary_index.rbegin()->getSalary() << "\n";
        }
        
        // Salary band distribution
        auto& band_index = employees_.get<by_salary_band>();
        std::cout << "Salary bands: ";
        std::set<std::string> unique_bands;
        for (const auto& emp : band_index) {
            unique_bands.insert(emp.getSalaryBand());
        }
        for (const auto& band : unique_bands) {
            auto range = band_index.equal_range(band);
            std::cout << band << "(" << std::distance(range.first, range.second) << ") ";
        }
        std::cout << "\n";
    }
};

void demonstrate_comprehensive_multi_index() {
    HRSystem hr;
    
    // Add sample employees
    auto now = std::chrono::system_clock::now();
    auto days_ago = [&](int days) {
        return now - std::chrono::hours(24 * days);
    };
    
    hr.addEmployee(Employee(1, "Alice Johnson", "Engineering", 85000, "alice@company.com", days_ago(1200)));
    hr.addEmployee(Employee(2, "Bob Smith", "Marketing", 62000, "bob@company.com", days_ago(800)));
    hr.addEmployee(Employee(3, "Charlie Brown", "Engineering", 95000, "charlie@company.com", days_ago(600)));
    hr.addEmployee(Employee(4, "Diana Prince", "HR", 58000, "diana@company.com", days_ago(400)));
    hr.addEmployee(Employee(5, "Eve Davis", "Engineering", 78000, "eve@company.com", days_ago(200)));
    hr.addEmployee(Employee(6, "Frank Wilson", "Sales", 125000, "frank@company.com", days_ago(100)));
    
    // Demonstrate different query patterns
    std::cout << "\n=== Query Demonstrations ===\n";
    
    hr.findById(3);
    std::cout << "\n";
    
    hr.findByDepartment("Engineering");
    std::cout << "\n";
    
    hr.findBySalaryRange(70000, 90000);
    std::cout << "\n";
    
    hr.findByDeptAndMinSalary("Engineering", 80000);
    std::cout << "\n";
    
    hr.findBySalaryBand("Senior");
    std::cout << "\n";
    
    hr.showHiringTimeline();
    std::cout << "\n";
    
    hr.showPage(0, 3); // First page
    std::cout << "\n";
    
    hr.showPage(1, 3); // Second page
    std::cout << "\n";
    
    hr.printStatistics();
}
```

#### Custom Indices and Composite Keys

```cpp
// Advanced custom index example
#include <boost/multi_index/key_extractors.hpp>

// Custom key extractor for complex logic
struct DepartmentSizeKey {
    typedef std::string result_type;
    
    std::string operator()(const Employee& emp) const {
        // Custom logic: categorize departments by typical size
        const std::string& dept = emp.getDepartment();
        if (dept == "Engineering" || dept == "Sales") return "Large";
        if (dept == "Marketing" || dept == "HR") return "Medium";
        return "Small";
    }
};

// Multi-level composite key
struct EmployeeHierarchyKey {
    typedef boost::tuple<std::string, std::string, double> result_type;
    
    result_type operator()(const Employee& emp) const {
        return boost::make_tuple(
            emp.getDepartment(),     // Level 1: Department
            emp.getSalaryBand(),     // Level 2: Salary Band
            emp.getSalary()          // Level 3: Exact Salary
        );
    }
};

// Performance monitoring for queries
template<typename IndexType>
void benchmark_query(const std::string& name, IndexType& index, 
                     std::function<void(IndexType&)> query_func) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run query multiple times for better measurement
    for (int i = 0; i < 1000; ++i) {
        query_func(index);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << " query time: " << duration.count() / 1000.0 << " μs average\n";
}
```

### Boost.Bimap

**Concept**: A bidirectional map that maintains both forward (left→right) and reverse (right→left) mappings simultaneously, allowing efficient lookups in both directions.

**Think of it as**: Two maps in one - you can lookup by key to get value, or by value to get key.

#### Bimap Relation Types

| Relation Type | Left Side | Right Side | Use Case |
|---------------|-----------|------------|----------|
| **set_of, set_of** | Unique, ordered | Unique, ordered | One-to-one mapping |
| **multiset_of, multiset_of** | Duplicates allowed | Duplicates allowed | Many-to-many mapping |
| **unordered_set_of** | Unique, hashed | Unique, hashed | Fast one-to-one lookups |
| **list_of** | Insertion order | Insertion order | Ordered associations |

#### Bimap Architecture

```
Traditional Approach (Two Maps):
std::map<int, string> id_to_name;     // ID → Name
std::map<string, int> name_to_id;     // Name → ID
- Duplicate data storage
- Manual synchronization required
- Inconsistency risk

Boost.Bimap Approach:
boost::bimap<int, string> employees;
- Single data storage
- Automatic synchronization
- Guaranteed consistency
- Efficient both-way lookups
```

#### Comprehensive Bimap Examples

```cpp
#include <boost/bimap.hpp>
#include <boost/bimap/set_of.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/bimap/unordered_set_of.hpp>
#include <boost/bimap/list_of.hpp>
#include <iostream>
#include <string>

// Basic one-to-one mapping
void demonstrate_basic_bimap() {
    std::cout << "=== Basic Bimap (One-to-One) ===\n";
    
    // Employee ID ↔ Employee Name mapping
    boost::bimap<int, std::string> employee_map;
    
    // Insert mappings
    employee_map.insert({1, "Alice Johnson"});
    employee_map.insert({2, "Bob Smith"});
    employee_map.insert({3, "Charlie Brown"});
    
    // Forward lookup (ID → Name)
    std::cout << "Employee 2: " << employee_map.left.at(2) << "\n";
    
    // Reverse lookup (Name → ID)
    std::cout << "Alice's ID: " << employee_map.right.at("Alice Johnson") << "\n";
    
    // Check existence before access
    if (employee_map.left.count(4) == 0) {
        std::cout << "Employee 4 not found\n";
    }
    
    // Iterate through left view (ID → Name)
    std::cout << "All employees (by ID):\n";
    for (const auto& pair : employee_map.left) {
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
    }
    
    // Iterate through right view (Name → ID)
    std::cout << "All employees (by name):\n";
    for (const auto& pair : employee_map.right) {
        std::cout << "  " << pair.first << ": " << pair.second << "\n";
    }
}

// Many-to-many mapping using multiset
void demonstrate_many_to_many_bimap() {
    std::cout << "\n=== Many-to-Many Bimap ===\n";
    
    // Student ↔ Course mapping (students can take multiple courses)
    boost::bimap<
        boost::bimaps::multiset_of<std::string>,  // Student names (duplicates allowed)
        boost::bimaps::multiset_of<std::string>   // Course names (duplicates allowed)
    > student_courses;
    
    // Add student-course relationships
    student_courses.insert({"Alice", "Math"});
    student_courses.insert({"Alice", "Physics"});
    student_courses.insert({"Alice", "Chemistry"});
    student_courses.insert({"Bob", "Math"});
    student_courses.insert({"Bob", "Computer Science"});
    student_courses.insert({"Charlie", "Physics"});
    student_courses.insert({"Charlie", "Chemistry"});
    
    // Find all courses for a student
    std::cout << "Alice's courses:\n";
    auto alice_courses = student_courses.left.equal_range("Alice");
    for (auto it = alice_courses.first; it != alice_courses.second; ++it) {
        std::cout << "  " << it->second << "\n";
    }
    
    // Find all students in a course
    std::cout << "Students in Physics:\n";
    auto physics_students = student_courses.right.equal_range("Physics");
    for (auto it = physics_students.first; it != physics_students.second; ++it) {
        std::cout << "  " << it->second << "\n";
    }
    
    // Count relationships
    std::cout << "Alice is taking " << student_courses.left.count("Alice") << " courses\n";
    std::cout << "Physics has " << student_courses.right.count("Physics") << " students\n";
}

// High-performance hash-based bimap
void demonstrate_hash_bimap() {
    std::cout << "\n=== Hash-based Bimap (High Performance) ===\n";
    
    // IP Address ↔ Hostname mapping for fast network lookups
    boost::bimap<
        boost::bimaps::unordered_set_of<std::string>,  // IP addresses
        boost::bimaps::unordered_set_of<std::string>   // Hostnames
    > dns_cache;
    
    // Populate DNS cache
    dns_cache.insert({"192.168.1.1", "router.local"});
    dns_cache.insert({"192.168.1.100", "server.local"});
    dns_cache.insert({"192.168.1.200", "workstation.local"});
    dns_cache.insert({"8.8.8.8", "dns.google"});
    
    // Fast forward resolution (IP → Hostname)
    auto resolve_ip = [&](const std::string& ip) {
        auto it = dns_cache.left.find(ip);
        return (it != dns_cache.left.end()) ? it->second : "Unknown";
    };
    
    // Fast reverse resolution (Hostname → IP)
    auto resolve_hostname = [&](const std::string& hostname) {
        auto it = dns_cache.right.find(hostname);
        return (it != dns_cache.right.end()) ? it->second : "Unknown";
    };
    
    // Test lookups
    std::cout << "IP 192.168.1.1 resolves to: " << resolve_ip("192.168.1.1") << "\n";
    std::cout << "Hostname server.local resolves to: " << resolve_hostname("server.local") << "\n";
    std::cout << "Unknown IP 1.2.3.4 resolves to: " << resolve_ip("1.2.3.4") << "\n";
    
    // Both lookups are O(1) average time complexity!
}

// Real-world use case: Translation system
class TranslationSystem {
private:
    boost::bimap<std::string, std::string> translations_;
    
public:
    void add_translation(const std::string& english, const std::string& spanish) {
        translations_.insert({english, spanish});
    }
    
    std::string translate_to_spanish(const std::string& english) {
        auto it = translations_.left.find(english);
        return (it != translations_.left.end()) ? it->second : "[Translation not found]";
    }
    
    std::string translate_to_english(const std::string& spanish) {
        auto it = translations_.right.find(spanish);
        return (it != translations_.right.end()) ? it->second : "[Translation not found]";
    }
    
    void load_dictionary() {
        add_translation("hello", "hola");
        add_translation("goodbye", "adiós");
        add_translation("thank you", "gracias");
        add_translation("please", "por favor");
        add_translation("water", "agua");
        add_translation("food", "comida");
    }
    
    void interactive_translation() {
        std::string input;
        std::cout << "Enter English or Spanish word (or 'quit' to exit):\n";
        
        while (std::getline(std::cin, input) && input != "quit") {
            if (input.empty()) continue;
            
            // Try English → Spanish
            std::string spanish = translate_to_spanish(input);
            if (spanish != "[Translation not found]") {
                std::cout << "English → Spanish: " << input << " → " << spanish << "\n";
            } else {
                // Try Spanish → English
                std::string english = translate_to_english(input);
                if (english != "[Translation not found]") {
                    std::cout << "Spanish → English: " << input << " → " << english << "\n";
                } else {
                    std::cout << "Translation not found for: " << input << "\n";
                }
            }
            
            std::cout << "Enter another word (or 'quit' to exit):\n";
        }
    }
    
    void print_dictionary() {
        std::cout << "English → Spanish Dictionary:\n";
        for (const auto& pair : translations_.left) {
            std::cout << "  " << pair.first << " → " << pair.second << "\n";
        }
    }
};

// Custom relation types and constraints
void demonstrate_custom_bimap() {
    std::cout << "\n=== Custom Bimap Relations ===\n";
    
    // User ID ↔ Session Token with custom constraints
    typedef boost::bimap<
        boost::bimaps::set_of<int>,                    // User IDs (unique, ordered)
        boost::bimaps::unordered_set_of<std::string>   // Session tokens (unique, hashed)
    > SessionMap;
    
    SessionMap active_sessions;
    
    // Session management functions
    auto create_session = [&](int user_id, const std::string& token) {
        // Check if user already has a session
        auto existing = active_sessions.left.find(user_id);
        if (existing != active_sessions.left.end()) {
            std::cout << "User " << user_id << " already has session: " << existing->second << "\n";
            return false;
        }
        
        // Create new session
        active_sessions.insert({user_id, token});
        std::cout << "Created session for user " << user_id << ": " << token << "\n";
        return true;
    };
    
    auto validate_session = [&](const std::string& token) -> int {
        auto it = active_sessions.right.find(token);
        if (it != active_sessions.right.end()) {
            std::cout << "Valid session " << token << " for user " << it->second << "\n";
            return it->second;
        } else {
            std::cout << "Invalid session token: " << token << "\n";
            return -1;
        }
    };
    
    auto logout_user = [&](int user_id) {
        auto it = active_sessions.left.find(user_id);
        if (it != active_sessions.left.end()) {
            std::cout << "Logging out user " << user_id << " (token: " << it->second << ")\n";
            active_sessions.left.erase(it);
            return true;
        } else {
            std::cout << "User " << user_id << " not logged in\n";
            return false;
        }
    };
    
    // Test session management
    create_session(1001, "abc123def456");
    create_session(1002, "xyz789uvw012");
    create_session(1001, "duplicate_attempt"); // Should fail
    
    validate_session("abc123def456");
    validate_session("invalid_token");
    
    logout_user(1001);
    validate_session("abc123def456"); // Should now be invalid
    
    std::cout << "Active sessions: " << active_sessions.size() << "\n";
}

void demonstrate_comprehensive_bimap() {
    demonstrate_basic_bimap();
    demonstrate_many_to_many_bimap();
    demonstrate_hash_bimap();
    demonstrate_custom_bimap();
    
    // Translation system demo
    std::cout << "\n=== Translation System Demo ===\n";
    TranslationSystem translator;
    translator.load_dictionary();
    translator.print_dictionary();
    
    // Example translations
    std::cout << "\nExample translations:\n";
    std::cout << "hello → " << translator.translate_to_spanish("hello") << "\n";
    std::cout << "gracias → " << translator.translate_to_english("gracias") << "\n";
    std::cout << "unknown → " << translator.translate_to_spanish("unknown") << "\n";
}
```

#### Performance Comparison: Bimap vs Two Maps

```cpp
// Benchmark: Bimap vs manual two-map approach
void benchmark_bimap_vs_two_maps() {
    const int NUM_OPERATIONS = 100000;
    
    // Setup data
    std::vector<std::pair<int, std::string>> test_data;
    for (int i = 0; i < NUM_OPERATIONS; ++i) {
        test_data.emplace_back(i, "Value_" + std::to_string(i));
    }
    
    // Benchmark Two Maps approach
    auto start = std::chrono::high_resolution_clock::now();
    
    std::map<int, std::string> forward_map;
    std::map<std::string, int> reverse_map;
    
    // Insert operations
    for (const auto& [key, value] : test_data) {
        forward_map[key] = value;
        reverse_map[value] = key;  // Manual synchronization
    }
    
    // Lookup operations
    for (int i = 0; i < 1000; ++i) {
        auto it1 = forward_map.find(i);
        auto it2 = reverse_map.find("Value_" + std::to_string(i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto two_maps_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Benchmark Bimap approach
    start = std::chrono::high_resolution_clock::now();
    
    boost::bimap<int, std::string> bimap;
    
    // Insert operations
    for (const auto& [key, value] : test_data) {
        bimap.insert({key, value});  // Single insertion
    }
    
    // Lookup operations
    for (int i = 0; i < 1000; ++i) {
        auto it1 = bimap.left.find(i);
        auto it2 = bimap.right.find("Value_" + std::to_string(i));
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto bimap_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Performance Comparison:\n";
    std::cout << "Two Maps: " << two_maps_time.count() << " ms\n";
    std::cout << "Bimap: " << bimap_time.count() << " ms\n";
    std::cout << "Memory usage - Two Maps: ~" << (forward_map.size() * 2 * sizeof(std::pair<int, std::string>)) << " bytes\n";
    std::cout << "Memory usage - Bimap: ~" << (bimap.size() * sizeof(std::pair<int, std::string>)) << " bytes\n";
}
```

### Boost.CircularBuffer

**Concept**: A fixed-size container that maintains a circular buffer where new elements overwrite the oldest elements once capacity is reached. Perfect for sliding windows, streaming data, and fixed-size caches.

**Think of it as**: A ring buffer that automatically manages memory and provides STL-compatible interface.

#### CircularBuffer Characteristics

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Fixed Capacity** | Size never exceeds specified limit | Predictable memory usage |
| **Automatic Overwrite** | Oldest elements removed automatically | No manual memory management |
| **STL Compatible** | Standard iterator interface | Easy integration |
| **Efficient Operations** | O(1) push/pop operations | High performance |

#### CircularBuffer States

```
Empty Buffer (capacity = 5):
┌─┬─┬─┬─┬─┐
│ │ │ │ │ │
└─┴─┴─┴─┴─┘
head=0, tail=0, size=0

Partially Filled:
┌─┬─┬─┬─┬─┐
│A│B│C│ │ │
└─┴─┴─┴─┴─┘
head=0, tail=3, size=3

Full Buffer:
┌─┬─┬─┬─┬─┐
│A│B│C│D│E│
└─┴─┴─┴─┴─┘
head=0, tail=0, size=5

After Overwrite (added F):
┌─┬─┬─┬─┬─┐
│F│B│C│D│E│
└─┴─┴─┴─┴─┘
head=1, tail=1, size=5
```

#### Comprehensive CircularBuffer Examples

```cpp
#include <boost/circular_buffer.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Basic circular buffer operations
void demonstrate_basic_circular_buffer() {
    std::cout << "=== Basic Circular Buffer Operations ===\n";
    
    // Create circular buffer with capacity of 5
    boost::circular_buffer<int> cb(5);
    
    std::cout << "Initial state:\n";
    std::cout << "  Capacity: " << cb.capacity() << "\n";
    std::cout << "  Size: " << cb.size() << "\n";
    std::cout << "  Empty: " << std::boolalpha << cb.empty() << "\n";
    std::cout << "  Full: " << cb.full() << "\n\n";
    
    // Fill the buffer beyond capacity
    std::cout << "Adding elements 1-7:\n";
    for (int i = 1; i <= 7; ++i) {
        cb.push_back(i);
        
        std::cout << "Added " << i << " → Buffer: [";
        for (auto it = cb.begin(); it != cb.end(); ++it) {
            if (it != cb.begin()) std::cout << ", ";
            std::cout << *it;
        }
        std::cout << "] (size: " << cb.size() << ", full: " << cb.full() << ")\n";
    }
    
    // Demonstrate front/back access
    std::cout << "\nBuffer access:\n";
    std::cout << "  Front: " << cb.front() << "\n";
    std::cout << "  Back: " << cb.back() << "\n";
    std::cout << "  Element [2]: " << cb[2] << "\n";
    
    // Remove elements
    std::cout << "\nRemoving elements:\n";
    while (!cb.empty()) {
        std::cout << "Removed " << cb.front() << " → ";
        cb.pop_front();
        std::cout << "Size: " << cb.size() << "\n";
    }
}

// Real-world use case: Moving average calculator
class MovingAverageCalculator {
private:
    boost::circular_buffer<double> values_;
    double sum_;
    
public:
    explicit MovingAverageCalculator(size_t window_size) 
        : values_(window_size), sum_(0.0) {}
    
    void add_value(double value) {
        if (values_.full()) {
            // Remove oldest value from sum
            sum_ -= values_.front();
        }
        
        values_.push_back(value);
        sum_ += value;
    }
    
    double get_average() const {
        return values_.empty() ? 0.0 : sum_ / values_.size();
    }
    
    size_t get_count() const { return values_.size(); }
    size_t get_window_size() const { return values_.capacity(); }
    
    void print_window() const {
        std::cout << "Window: [";
        for (auto it = values_.begin(); it != values_.end(); ++it) {
            if (it != values_.begin()) std::cout << ", ";
            std::cout << *it;
        }
        std::cout << "] avg=" << get_average() << "\n";
    }
    
    // Calculate other statistics
    double get_min() const {
        return values_.empty() ? 0.0 : *std::min_element(values_.begin(), values_.end());
    }
    
    double get_max() const {
        return values_.empty() ? 0.0 : *std::max_element(values_.begin(), values_.end());
    }
    
    double get_variance() const {
        if (values_.size() < 2) return 0.0;
        
        double avg = get_average();
        double sq_sum = 0.0;
        for (double val : values_) {
            sq_sum += (val - avg) * (val - avg);
        }
        return sq_sum / values_.size();
    }
};

void demonstrate_moving_average() {
    std::cout << "=== Moving Average Calculator ===\n";
    
    MovingAverageCalculator calc(5); // 5-point moving average
    
    // Simulate stock prices or sensor data
    std::vector<double> data = {100.0, 102.5, 101.0, 103.2, 99.8, 98.5, 101.1, 104.0, 102.3, 100.9};
    
    std::cout << "Processing data stream:\n";
    for (double value : data) {
        calc.add_value(value);
        std::cout << "New value: " << value << " → ";
        calc.print_window();
        
        if (calc.get_count() >= 3) { // Show statistics when we have enough data
            std::cout << "  Stats: min=" << calc.get_min() 
                      << ", max=" << calc.get_max() 
                      << ", variance=" << calc.get_variance() << "\n";
        }
        std::cout << "\n";
    }
}

// Performance monitoring system
class PerformanceMonitor {
private:
    boost::circular_buffer<std::chrono::milliseconds> response_times_;
    boost::circular_buffer<double> cpu_usage_;
    boost::circular_buffer<size_t> memory_usage_;
    
public:
    PerformanceMonitor(size_t history_size = 100) 
        : response_times_(history_size), cpu_usage_(history_size), memory_usage_(history_size) {}
    
    void record_response_time(std::chrono::milliseconds time) {
        response_times_.push_back(time);
    }
    
    void record_cpu_usage(double percentage) {
        cpu_usage_.push_back(percentage);
    }
    
    void record_memory_usage(size_t bytes) {
        memory_usage_.push_back(bytes);
    }
    
    // Performance analysis
    struct PerformanceStats {
        double avg_response_time_ms;
        double avg_cpu_percentage;
        double avg_memory_mb;
        size_t sample_count;
        std::chrono::milliseconds max_response_time;
        double max_cpu_percentage;
        size_t max_memory_bytes;
    };
    
    PerformanceStats get_stats() const {
        PerformanceStats stats = {};
        
        if (!response_times_.empty()) {
            double sum_ms = 0.0;
            std::chrono::milliseconds max_time{0};
            
            for (const auto& time : response_times_) {
                sum_ms += time.count();
                max_time = std::max(max_time, time);
            }
            
            stats.avg_response_time_ms = sum_ms / response_times_.size();
            stats.max_response_time = max_time;
        }
        
        if (!cpu_usage_.empty()) {
            double sum_cpu = std::accumulate(cpu_usage_.begin(), cpu_usage_.end(), 0.0);
            stats.avg_cpu_percentage = sum_cpu / cpu_usage_.size();
            stats.max_cpu_percentage = *std::max_element(cpu_usage_.begin(), cpu_usage_.end());
        }
        
        if (!memory_usage_.empty()) {
            double sum_mem = std::accumulate(memory_usage_.begin(), memory_usage_.end(), 0.0);
            stats.avg_memory_mb = (sum_mem / memory_usage_.size()) / (1024 * 1024);
            stats.max_memory_bytes = *std::max_element(memory_usage_.begin(), memory_usage_.end());
        }
        
        stats.sample_count = std::min({response_times_.size(), cpu_usage_.size(), memory_usage_.size()});
        return stats;
    }
    
    void print_dashboard() const {
        auto stats = get_stats();
        
        std::cout << "=== Performance Dashboard ===\n";
        std::cout << "Sample Count: " << stats.sample_count << "\n";
        std::cout << "Response Time: avg=" << stats.avg_response_time_ms << "ms"
                  << ", max=" << stats.max_response_time.count() << "ms\n";
        std::cout << "CPU Usage: avg=" << stats.avg_cpu_percentage << "%"
                  << ", max=" << stats.max_cpu_percentage << "%\n";
        std::cout << "Memory Usage: avg=" << stats.avg_memory_mb << "MB"
                  << ", max=" << (stats.max_memory_bytes / (1024.0 * 1024.0)) << "MB\n";
        
        // Trend analysis (simple)
        if (response_times_.size() >= 10) {
            auto recent_avg = std::accumulate(response_times_.end() - 5, response_times_.end(), 
                                            std::chrono::milliseconds{0}).count() / 5.0;
            auto older_avg = std::accumulate(response_times_.end() - 10, response_times_.end() - 5, 
                                           std::chrono::milliseconds{0}).count() / 5.0;
            
            std::cout << "Response Time Trend: ";
            if (recent_avg > older_avg * 1.1) {
                std::cout << "DEGRADING (" << recent_avg << "ms vs " << older_avg << "ms)\n";
            } else if (recent_avg < older_avg * 0.9) {
                std::cout << "IMPROVING (" << recent_avg << "ms vs " << older_avg << "ms)\n";
            } else {
                std::cout << "STABLE (" << recent_avg << "ms vs " << older_avg << "ms)\n";
            }
        }
    }
};

void demonstrate_performance_monitoring() {
    std::cout << "=== Performance Monitoring System ===\n";
    
    PerformanceMonitor monitor(20); // Keep last 20 measurements
    
    // Simulate performance data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> response_dist(50, 200);  // 50-200ms response times
    std::uniform_real_distribution<> cpu_dist(10.0, 80.0);  // 10-80% CPU
    std::uniform_int_distribution<> mem_dist(100, 500);     // 100-500MB memory
    
    std::cout << "Collecting performance data...\n\n";
    
    for (int i = 0; i < 25; ++i) {
        // Simulate some performance degradation over time
        int response_base = response_dist(gen) + (i / 5) * 10; // Gradual slowdown
        double cpu_base = cpu_dist(gen) + (i / 10) * 5.0;      // Gradual CPU increase
        int mem_base = mem_dist(gen) + i * 5;                  // Gradual memory increase
        
        monitor.record_response_time(std::chrono::milliseconds(response_base));
        monitor.record_cpu_usage(cpu_base);
        monitor.record_memory_usage(mem_base * 1024 * 1024); // Convert to bytes
        
        // Print dashboard every 5 samples
        if ((i + 1) % 5 == 0) {
            std::cout << "After " << (i + 1) << " samples:\n";
            monitor.print_dashboard();
            std::cout << "\n";
        }
    }
}

// Streaming data buffer with different overwrite policies
template<typename T>
class StreamingBuffer {
private:
    boost::circular_buffer<T> buffer_;
    size_t overwrite_count_;
    
public:
    explicit StreamingBuffer(size_t capacity) 
        : buffer_(capacity), overwrite_count_(0) {}
    
    void push(const T& value) {
        if (buffer_.full()) {
            overwrite_count_++;
        }
        buffer_.push_back(value);
    }
    
    // Get latest N elements
    std::vector<T> get_latest(size_t n) const {
        std::vector<T> result;
        size_t start = (buffer_.size() > n) ? buffer_.size() - n : 0;
        
        for (size_t i = start; i < buffer_.size(); ++i) {
            result.push_back(buffer_[i]);
        }
        
        return result;
    }
    
    // Get elements by time window (assuming time-ordered data)
    template<typename TimeType>
    std::vector<T> get_by_time_window(TimeType current_time, TimeType window_duration,
                                     std::function<TimeType(const T&)> time_extractor) const {
        std::vector<T> result;
        TimeType cutoff_time = current_time - window_duration;
        
        for (const auto& item : buffer_) {
            if (time_extractor(item) >= cutoff_time) {
                result.push_back(item);
            }
        }
        
        return result;
    }
    
    size_t size() const { return buffer_.size(); }
    size_t capacity() const { return buffer_.capacity(); }
    size_t overwrite_count() const { return overwrite_count_; }
    bool full() const { return buffer_.full(); }
};

// Log entry for streaming example
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    std::string level;
    std::string message;
    
    LogEntry(const std::string& lvl, const std::string& msg) 
        : timestamp(std::chrono::system_clock::now()), level(lvl), message(msg) {}
};

void demonstrate_streaming_buffer() {
    std::cout << "=== Streaming Data Buffer ===\n";
    
    StreamingBuffer<LogEntry> log_buffer(10); // Keep last 10 log entries
    
    // Simulate log entries
    std::vector<std::pair<std::string, std::string>> log_data = {
        {"INFO", "Application started"},
        {"DEBUG", "Loading configuration"},
        {"INFO", "Database connected"},
        {"WARN", "High memory usage detected"},
        {"ERROR", "Failed to process request"},
        {"INFO", "Request processed successfully"},
        {"DEBUG", "Cache cleared"},
        {"WARN", "Disk space low"},
        {"ERROR", "Database connection lost"},
        {"INFO", "Database reconnected"},
        {"DEBUG", "Garbage collection triggered"},
        {"WARN", "Response time high"},
        {"ERROR", "Critical system error"},
        {"INFO", "System recovered"}
    };
    
    for (const auto& [level, message] : log_data) {
        log_buffer.push(LogEntry(level, message));
        
        std::cout << "Added: [" << level << "] " << message << "\n";
        std::cout << "Buffer status: " << log_buffer.size() << "/" << log_buffer.capacity()
                  << ", overwrites: " << log_buffer.overwrite_count() << "\n";
        
        // Show latest 3 entries
        auto latest = log_buffer.get_latest(3);
        std::cout << "Latest 3 entries:\n";
        for (const auto& entry : latest) {
            std::cout << "  [" << entry.level << "] " << entry.message << "\n";
        }
        std::cout << "\n";
    }
}

void demonstrate_comprehensive_circular_buffer() {
    demonstrate_basic_circular_buffer();
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    
    demonstrate_moving_average();
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    
    demonstrate_performance_monitoring();
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    
    demonstrate_streaming_buffer();
}
```

#### Advanced CircularBuffer Techniques

```cpp
// Custom allocator support
void demonstrate_custom_allocator() {
    // Circular buffer with custom allocator for specific memory pools
    std::cout << "=== Custom Allocator Support ===\n";
    
    // Example: Pool allocator for high-performance scenarios
    // boost::circular_buffer<int, PoolAllocator<int>> pool_buffer(100);
    
    // For now, demonstrate standard allocator awareness
    boost::circular_buffer<std::string> string_buffer(5);
    
    // The circular buffer properly constructs/destructs elements
    string_buffer.push_back("First");
    string_buffer.push_back("Second");
    string_buffer.push_back(std::string(1000, 'X')); // Large string
    
    std::cout << "Buffer handles complex objects properly\n";
    std::cout << "Size: " << string_buffer.size() << "\n";
    
    // Elements are properly destroyed when overwritten
    for (int i = 0; i < 10; ++i) {
        string_buffer.push_back("New_" + std::to_string(i));
    }
    
    std::cout << "After overwrites, size: " << string_buffer.size() << "\n";
}

// Thread-safe wrapper (basic example)
template<typename T>
class ThreadSafeCircularBuffer {
private:
    mutable std::mutex mutex_;
    boost::circular_buffer<T> buffer_;
    
public:
    explicit ThreadSafeCircularBuffer(size_t capacity) : buffer_(capacity) {}
    
    void push_back(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(item);
    }
    
    bool try_pop_front(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) {
            return false;
        }
        item = buffer_.front();
        buffer_.pop_front();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }
    
    std::vector<T> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<T>(buffer_.begin(), buffer_.end());
    }
};
```

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

### Exercise 1: Container Performance Analysis
**Objective**: Compare performance characteristics of different container types.

```cpp
// TODO: Implement comprehensive benchmark
class ContainerBenchmark {
public:
    // Compare flat_map vs std::map for different workloads
    void benchmark_map_types(size_t data_size, double read_write_ratio);
    
    // Compare small_vector vs std::vector for different sizes
    void benchmark_small_collections(size_t max_size);
    
    // Compare multi_index vs separate containers
    void benchmark_multi_access_patterns();
    
    // Measure memory usage and cache effects
    void analyze_memory_patterns();
};

// YOUR TASK:
// 1. Implement each benchmark method
// 2. Test with different data sizes (100, 1K, 10K, 100K elements)
// 3. Vary read/write ratios (90/10, 70/30, 50/50)
// 4. Document performance characteristics
// 5. Create recommendations for container selection
```

### Exercise 2: Multi-Index Database System
**Objective**: Design a comprehensive student management system using multi-index containers.

```cpp
// TODO: Complete the student database implementation
class StudentDatabase {
private:
    // Define multi-index container with indices for:
    // - Student ID (unique)
    // - Name (non-unique)
    // - GPA (ordered for range queries)
    // - Major (hashed for fast lookups)
    // - Enrollment date (for temporal queries)
    // - Composite index (Major + GPA)
    
public:
    // IMPLEMENT THESE METHODS:
    void enroll_student(const Student& student);
    void update_gpa(int student_id, double new_gpa);
    std::vector<Student> find_by_major(const std::string& major);
    std::vector<Student> find_by_gpa_range(double min_gpa, double max_gpa);
    std::vector<Student> find_top_students_by_major(const std::string& major, size_t count);
    std::vector<Student> find_students_enrolled_after(const std::chrono::system_clock::time_point& date);
    void generate_transcript(int student_id);
    void print_statistics();
    
    // ADVANCED FEATURES:
    void bulk_import_students(const std::vector<Student>& students);
    void export_to_csv(const std::string& filename);
    void create_backup_snapshot();
    void restore_from_snapshot();
};

// Test your implementation with:
// - 10,000+ student records
// - Complex queries combining multiple criteria
// - Bulk operations
// - Error handling and edge cases
```

### Exercise 3: Real-Time Data Processing System
**Objective**: Build a streaming data processor using circular buffers.

```cpp
// TODO: Implement real-time data processing pipeline
class RealTimeProcessor {
private:
    // Use circular buffers for different data streams
    boost::circular_buffer<SensorReading> temperature_buffer_;
    boost::circular_buffer<SensorReading> humidity_buffer_;
    boost::circular_buffer<SensorReading> pressure_buffer_;
    
public:
    // IMPLEMENT THESE METHODS:
    void process_sensor_data(const SensorReading& reading);
    MovingStatistics calculate_moving_statistics(SensorType type, size_t window_size);
    std::vector<Anomaly> detect_anomalies();
    void trigger_alerts(const std::vector<Anomaly>& anomalies);
    void export_historical_data(const std::string& filename);
    
    // ADVANCED FEATURES:
    void implement_sliding_window_correlation();
    void detect_trends_and_patterns();
    void implement_predictive_analytics();
    void handle_missing_data_points();
};

// Requirements:
// - Handle 1000+ data points per second
// - Maintain 1-hour rolling window
// - Detect anomalies in real-time
// - Calculate correlations between different sensors
// - Generate alerts for critical conditions
```

### Exercise 4: Bidirectional Translation System
**Objective**: Create a comprehensive translation and localization system.

```cpp
// TODO: Implement advanced translation system
class TranslationSystem {
private:
    // Use bimaps for different language pairs
    boost::bimap<std::string, std::string> en_es_translations_;
    boost::bimap<std::string, std::string> en_fr_translations_;
    boost::bimap<std::string, std::string> en_de_translations_;
    
    // Additional features
    boost::bimap<std::string, std::string> phrase_translations_;
    boost::circular_buffer<std::string> translation_history_;
    
public:
    // IMPLEMENT THESE METHODS:
    void load_dictionary(const std::string& language_pair, const std::string& filename);
    std::string translate(const std::string& text, const std::string& from_lang, const std::string& to_lang);
    std::vector<std::string> get_translation_suggestions(const std::string& partial_text);
    void add_custom_translation(const std::string& from_text, const std::string& to_text, const std::string& language_pair);
    std::vector<std::string> get_translation_history();
    void export_custom_dictionary(const std::string& filename);
    
    // ADVANCED FEATURES:
    void implement_fuzzy_matching();
    void handle_context_dependent_translations();
    void implement_translation_confidence_scoring();
    void support_batch_translation();
};

// Test scenarios:
// - Large dictionaries (100K+ entries)
// - Real-time translation requests
// - Fuzzy matching for typos
// - Context-aware translations
// - Performance under load
```

### Exercise 5: Memory-Efficient Game Engine Components
**Objective**: Design game engine systems using appropriate Boost containers.

```cpp
// TODO: Implement game engine systems
class GameEngine {
private:
    // Use small_vector for typically small collections
    boost::container::small_vector<Component*, 8> entity_components_;
    
    // Use stable_vector for objects that need stable references
    boost::container::stable_vector<GameObject> game_objects_;
    
    // Use flat_map for configuration and assets
    boost::container::flat_map<std::string, std::string> game_config_;
    boost::container::flat_map<std::string, Asset*> asset_cache_;
    
    // Use circular_buffer for frame timing and input history
    boost::circular_buffer<double> frame_times_;
    boost::circular_buffer<InputEvent> input_history_;
    
public:
    // IMPLEMENT THESE SYSTEMS:
    void update_game_objects(double delta_time);
    void render_scene();
    void handle_input(const InputEvent& event);
    void load_game_assets();
    void manage_memory_pools();
    
    // PERFORMANCE REQUIREMENTS:
    // - Maintain 60 FPS with 10,000+ game objects
    // - Minimize memory allocations during gameplay
    // - Provide stable references for game objects
    // - Efficient asset lookup and caching
    // - Real-time input processing
};

// Challenges:
// - Optimize for cache locality
// - Minimize dynamic allocations
// - Handle object lifecycle properly
// - Implement efficient spatial partitioning
// - Manage resource cleanup
```

## Performance Considerations

### Memory Layout and Cache Performance

#### Cache-Friendly Containers
```cpp
// Demonstrate cache performance differences
#include <chrono>
#include <random>

void benchmark_cache_performance() {
    const size_t NUM_ELEMENTS = 100000;
    const size_t NUM_LOOKUPS = 10000;
    
    // Generate random keys for lookup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, NUM_ELEMENTS - 1);
    
    std::vector<int> lookup_keys;
    for (size_t i = 0; i < NUM_LOOKUPS; ++i) {
        lookup_keys.push_back(dis(gen));
    }
    
    // Test std::map (tree-based, non-contiguous memory)
    std::map<int, int> tree_map;
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        tree_map[i] = i * 2;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    volatile int sum1 = 0;
    for (int key : lookup_keys) {
        sum1 += tree_map[key];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto tree_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test boost::container::flat_map (vector-based, contiguous memory)
    boost::container::flat_map<int, int> flat_map;
    flat_map.reserve(NUM_ELEMENTS);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        flat_map[i] = i * 2;
    }
    
    start = std::chrono::high_resolution_clock::now();
    volatile int sum2 = 0;
    for (int key : lookup_keys) {
        sum2 += flat_map[key];
    }
    end = std::chrono::high_resolution_clock::now();
    auto flat_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Cache Performance Comparison:\n";
    std::cout << "std::map lookups: " << tree_time.count() << " μs\n";
    std::cout << "flat_map lookups: " << flat_time.count() << " μs\n";
    std::cout << "Performance ratio: " << (double)tree_time.count() / flat_time.count() << "x\n";
    
    // Memory usage comparison
    size_t tree_memory = NUM_ELEMENTS * (sizeof(std::pair<int, int>) + 3 * sizeof(void*) + sizeof(char)); // Node overhead
    size_t flat_memory = NUM_ELEMENTS * sizeof(std::pair<int, int>);
    
    std::cout << "Memory usage - std::map: " << tree_memory << " bytes\n";
    std::cout << "Memory usage - flat_map: " << flat_memory << " bytes\n";
    std::cout << "Memory ratio: " << (double)tree_memory / flat_memory << "x\n";
}
```

#### Small Object Optimization Analysis
```cpp
// Analyze when small_vector provides benefits
template<size_t InlineCapacity>
void benchmark_small_vector(const std::string& name) {
    const int ITERATIONS = 1000000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; ++i) {
        boost::container::small_vector<int, InlineCapacity> vec;
        
        // Fill with elements that fit in inline storage
        for (int j = 0; j < InlineCapacity; ++j) {
            vec.push_back(j);
        }
        
        // Process elements
        volatile int sum = 0;
        for (int val : vec) {
            sum += val;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << " time: " << duration.count() << " μs\n";
}

void analyze_small_vector_optimization() {
    std::cout << "Small Vector Optimization Analysis:\n";
    
    benchmark_small_vector<0>("std::vector equivalent");
    benchmark_small_vector<4>("small_vector<4>");
    benchmark_small_vector<8>("small_vector<8>");
    benchmark_small_vector<16>("small_vector<16>");
    benchmark_small_vector<32>("small_vector<32>");
    
    std::cout << "\nOptimal inline capacity depends on:\n";
    std::cout << "- Typical collection size in your application\n";
    std::cout << "- Frequency of collection creation/destruction\n";
    std::cout << "- Memory allocation cost in your environment\n";
}
```

### Multi-Index Performance Trade-offs

#### Index Maintenance Overhead
```cpp
// Demonstrate multi-index update costs
void benchmark_multi_index_updates() {
    const size_t NUM_OPERATIONS = 10000;
    
    // Single index container (std::map)
    std::map<int, Employee> single_index;
    
    // Multi-index container
    EmployeeDatabase multi_index;
    
    // Benchmark insertions
    std::vector<Employee> test_employees;
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        test_employees.emplace_back(i, "Employee_" + std::to_string(i), 
                                   "Dept_" + std::to_string(i % 10), 
                                   50000 + (i % 50000));
    }
    
    // Single index insertion
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& emp : test_employees) {
        single_index[emp.getId()] = emp;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto single_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Multi-index insertion
    start = std::chrono::high_resolution_clock::now();
    for (const auto& emp : test_employees) {
        multi_index.addEmployee(emp);
    }
    end = std::chrono::high_resolution_clock::now();
    auto multi_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Insertion Performance:\n";
    std::cout << "Single index: " << single_time.count() << " ms\n";
    std::cout << "Multi-index: " << multi_time.count() << " ms\n";
    std::cout << "Overhead ratio: " << (double)multi_time.count() / single_time.count() << "x\n";
    
    // But multi-index provides multiple efficient query paths!
    std::cout << "\nQuery capabilities:\n";
    std::cout << "Single index: O(log n) by ID only\n";
    std::cout << "Multi-index: O(log n) by ID, name, salary; O(1) by department\n";
}
```

### Circular Buffer Performance Characteristics

#### Memory Allocation Patterns
```cpp
// Compare circular buffer vs dynamic containers
void benchmark_streaming_performance() {
    const size_t NUM_OPERATIONS = 1000000;
    const size_t BUFFER_SIZE = 1000;
    
    // Dynamic container (vector with periodic cleanup)
    std::vector<int> dynamic_buffer;
    dynamic_buffer.reserve(BUFFER_SIZE * 2);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        dynamic_buffer.push_back(i);
        
        // Periodic cleanup to maintain size
        if (dynamic_buffer.size() > BUFFER_SIZE) {
            dynamic_buffer.erase(dynamic_buffer.begin(), 
                               dynamic_buffer.begin() + (dynamic_buffer.size() - BUFFER_SIZE));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto dynamic_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Circular buffer
    boost::circular_buffer<int> circular_buffer(BUFFER_SIZE);
    
    start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
        circular_buffer.push_back(i);
        // No manual cleanup needed!
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto circular_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Streaming Performance Comparison:\n";
    std::cout << "Dynamic buffer: " << dynamic_time.count() << " ms\n";
    std::cout << "Circular buffer: " << circular_time.count() << " ms\n";
    std::cout << "Performance improvement: " << (double)dynamic_time.count() / circular_time.count() << "x\n";
    
    std::cout << "\nMemory characteristics:\n";
    std::cout << "Dynamic buffer: Variable size, frequent allocations\n";
    std::cout << "Circular buffer: Fixed size, no allocations after initialization\n";
}
```

### Container Selection Guidelines

#### Decision Matrix
```cpp
// Container selection helper
class ContainerSelector {
public:
    enum class AccessPattern {
        SEQUENTIAL,
        RANDOM,
        KEY_VALUE,
        MULTI_KEY,
        BIDIRECTIONAL,
        STREAMING
    };
    
    enum class SizePattern {
        SMALL_FIXED,      // < 16 elements, known size
        SMALL_VARIABLE,   // < 100 elements, unknown size
        MEDIUM,           // 100-10K elements
        LARGE,            // > 10K elements
        STREAMING         // Continuous data flow
    };
    
    enum class UpdatePattern {
        READ_ONLY,        // No modifications after creation
        READ_HEAVY,       // 90%+ reads
        BALANCED,         // Mixed read/write
        WRITE_HEAVY,      // Frequent modifications
        APPEND_ONLY       // Only additions
    };
    
    std::string recommend_container(AccessPattern access, SizePattern size, UpdatePattern update) {
        if (access == AccessPattern::STREAMING) {
            return "boost::circular_buffer - optimal for streaming data";
        }
        
        if (access == AccessPattern::BIDIRECTIONAL) {
            return "boost::bimap - efficient bidirectional lookups";
        }
        
        if (access == AccessPattern::MULTI_KEY) {
            return "boost::multi_index_container - multiple efficient access paths";
        }
        
        if (access == AccessPattern::KEY_VALUE) {
            if (update == UpdatePattern::READ_HEAVY || update == UpdatePattern::READ_ONLY) {
                return "boost::container::flat_map - cache-friendly for read-heavy workloads";
            } else {
                return "std::map - better for frequent modifications";
            }
        }
        
        if (access == AccessPattern::SEQUENTIAL || access == AccessPattern::RANDOM) {
            if (size == SizePattern::SMALL_FIXED) {
                return "boost::container::static_vector - no dynamic allocation";
            } else if (size == SizePattern::SMALL_VARIABLE) {
                return "boost::container::small_vector - optimized for small sizes";
            } else {
                return "std::vector - general purpose dynamic array";
            }
        }
        
        return "std::vector - default recommendation";
    }
    
    void print_selection_guide() {
        std::cout << "Container Selection Guide:\n\n";
        
        std::cout << "For small collections (< 16 elements):\n";
        std::cout << "  - Fixed size: static_vector\n";
        std::cout << "  - Variable size: small_vector\n\n";
        
        std::cout << "For key-value storage:\n";
        std::cout << "  - Read-heavy: flat_map\n";
        std::cout << "  - Write-heavy: std::map\n";
        std::cout << "  - Bidirectional: bimap\n";
        std::cout << "  - Multiple access patterns: multi_index\n\n";
        
        std::cout << "For streaming data:\n";
        std::cout << "  - Fixed-size window: circular_buffer\n";
        std::cout << "  - Growing data: std::deque\n\n";
        
        std::cout << "For stable references:\n";
        std::cout << "  - stable_vector (iterator stability)\n";
        std::cout << "  - std::list (node-based)\n\n";
    }
};

void demonstrate_container_selection() {
    ContainerSelector selector;
    selector.print_selection_guide();
    
    // Example recommendations
    std::cout << "Example recommendations:\n";
    std::cout << "Game entity components: " 
              << selector.recommend_container(
                  ContainerSelector::AccessPattern::SEQUENTIAL,
                  ContainerSelector::SizePattern::SMALL_VARIABLE,
                  ContainerSelector::UpdatePattern::READ_HEAVY) << "\n";
    
    std::cout << "Configuration cache: " 
              << selector.recommend_container(
                  ContainerSelector::AccessPattern::KEY_VALUE,
                  ContainerSelector::SizePattern::MEDIUM,
                  ContainerSelector::UpdatePattern::READ_ONLY) << "\n";
    
    std::cout << "Real-time sensor data: " 
              << selector.recommend_container(
                  ContainerSelector::AccessPattern::STREAMING,
                  ContainerSelector::SizePattern::STREAMING,
                  ContainerSelector::UpdatePattern::APPEND_ONLY) << "\n";
}
```

## Best Practices

### Container Selection Strategy

#### 1. Analyze Access Patterns First
```cpp
// Before choosing a container, understand your access patterns
class AccessPatternAnalyzer {
public:
    struct AccessMetrics {
        size_t insertions = 0;
        size_t deletions = 0;
        size_t lookups = 0;
        size_t iterations = 0;
        size_t random_access = 0;
        
        double get_read_write_ratio() const {
            return (double)(lookups + iterations + random_access) / (insertions + deletions + 1);
        }
    };
    
    // Profile your actual usage patterns
    void profile_usage() {
        std::cout << "Profile your container usage:\n";
        std::cout << "1. How often do you insert/delete? (per second/minute/hour)\n";
        std::cout << "2. How often do you lookup elements?\n";
        std::cout << "3. Do you iterate through all elements?\n";
        std::cout << "4. Do you need random access by index?\n";
        std::cout << "5. What's the typical size range?\n";
        std::cout << "6. Do you need multiple access methods?\n";
    }
};
```

#### 2. Consider Memory Constraints
```cpp
// Memory-aware container selection
class MemoryConstraintAnalyzer {
public:
    void analyze_memory_requirements() {
        std::cout << "Memory Analysis Guidelines:\n\n";
        
        std::cout << "Embedded/Real-time Systems:\n";
        std::cout << "  - Prefer static_vector (no dynamic allocation)\n";
        std::cout << "  - Use small_vector for small collections\n";
        std::cout << "  - Avoid multi_index (high memory overhead)\n\n";
        
        std::cout << "Memory-Constrained Applications:\n";
        std::cout << "  - Use flat_map/flat_set (less memory overhead)\n";
        std::cout << "  - Consider circular_buffer for streaming data\n";
        std::cout << "  - Profile actual memory usage\n\n";
        
        std::cout << "High-Performance Applications:\n";
        std::cout << "  - Optimize for cache locality (flat containers)\n";
        std::cout << "  - Use multi_index for complex queries\n";
        std::cout << "  - Consider custom allocators\n";
    }
    
    // Memory usage estimation
    template<typename Container>
    size_t estimate_memory_usage(size_t element_count, size_t element_size) {
        // This is a simplified estimation
        if constexpr (std::is_same_v<Container, std::vector<int>>) {
            return element_count * element_size;
        } else if constexpr (std::is_same_v<Container, std::map<int, int>>) {
            return element_count * (element_size + 3 * sizeof(void*) + sizeof(char));
        } else if constexpr (std::is_same_v<Container, boost::container::flat_map<int, int>>) {
            return element_count * element_size;
        }
        return element_count * element_size; // Default estimation
    }
};
```

#### 3. Performance Optimization Guidelines
```cpp
// Performance optimization best practices
class PerformanceOptimizer {
public:
    void optimize_for_insertions() {
        std::cout << "Insertion Optimization:\n";
        std::cout << "- Reserve capacity for vectors and flat containers\n";
        std::cout << "- Use emplace_back instead of push_back when possible\n";
        std::cout << "- Batch insertions when possible\n";
        std::cout << "- Consider stable_vector if iterator stability is needed\n\n";
        
        // Example: Optimal insertion pattern
        boost::container::flat_map<int, std::string> optimized_map;
        optimized_map.reserve(1000); // Pre-allocate capacity
        
        // Batch insert
        std::vector<std::pair<int, std::string>> data;
        // ... populate data ...
        optimized_map.insert(data.begin(), data.end());
    }
    
    void optimize_for_lookups() {
        std::cout << "Lookup Optimization:\n";
        std::cout << "- Use flat_map for read-heavy workloads\n";
        std::cout << "- Use unordered containers for O(1) lookups\n";
        std::cout << "- Consider multi_index for multiple lookup paths\n";
        std::cout << "- Use bimap for bidirectional lookups\n\n";
    }
    
    void optimize_for_memory() {
        std::cout << "Memory Optimization:\n";
        std::cout << "- Use small_vector for small collections\n";
        std::cout << "- Use static_vector for fixed-size collections\n";
        std::cout << "- Shrink containers after bulk deletions\n";
        std::cout << "- Consider circular_buffer for streaming data\n\n";
    }
};
```

### Index Design for Multi-Index Containers

#### 4. Efficient Index Selection
```cpp
// Multi-index design best practices
class MultiIndexDesigner {
public:
    void design_efficient_indices() {
        std::cout << "Multi-Index Design Guidelines:\n\n";
        
        std::cout << "Index Type Selection:\n";
        std::cout << "- ordered_unique: Primary keys, unique identifiers\n";
        std::cout << "- ordered_non_unique: Range queries, sorting\n";
        std::cout << "- hashed_unique: Fast unique lookups\n";
        std::cout << "- hashed_non_unique: Fast lookups with duplicates\n";
        std::cout << "- sequenced: Insertion order, history\n";
        std::cout << "- random_access: Position-based access\n\n";
        
        std::cout << "Composite Key Guidelines:\n";
        std::cout << "- Order keys by selectivity (most selective first)\n";
        std::cout << "- Use composite keys for range queries\n";
        std::cout << "- Consider memory overhead of complex keys\n\n";
    }
    
    // Example: Well-designed multi-index
    struct OptimalEmployeeIndex {
        typedef bmi::multi_index_container<
            Employee,
            bmi::indexed_by<
                // Primary key - most selective
                bmi::ordered_unique<
                    bmi::tag<struct by_id>,
                    bmi::member<Employee, int, &Employee::id>
                >,
                
                // Fast department queries (common use case)
                bmi::hashed_non_unique<
                    bmi::tag<struct by_department>,
                    bmi::member<Employee, std::string, &Employee::department>
                >,
                
                // Salary range queries
                bmi::ordered_non_unique<
                    bmi::tag<struct by_salary>,
                    bmi::member<Employee, double, &Employee::salary>
                >,
                
                // Complex queries (department + salary)
                bmi::ordered_non_unique<
                    bmi::tag<struct by_dept_salary>,
                    bmi::composite_key<
                        Employee,
                        bmi::member<Employee, std::string, &Employee::department>,
                        bmi::member<Employee, double, &Employee::salary>
                    >
                >
            >
        > type;
    };
};
```

### Memory Management Best Practices

#### 5. Efficient Memory Usage
```cpp
// Memory management guidelines
class MemoryManager {
public:
    void demonstrate_memory_best_practices() {
        std::cout << "Memory Management Best Practices:\n\n";
        
        // 1. Reserve capacity for growing containers
        std::cout << "1. Reserve Capacity:\n";
        boost::container::flat_map<int, std::string> config_cache;
        config_cache.reserve(100); // Avoid multiple reallocations
        
        // 2. Use appropriate container for size
        std::cout << "2. Size-Appropriate Containers:\n";
        boost::container::small_vector<int, 8> small_list; // For typically small collections
        boost::container::static_vector<int, 5> fixed_list; // For known fixed size
        
        // 3. Shrink containers after bulk operations
        std::cout << "3. Shrink After Bulk Operations:\n";
        std::vector<int> bulk_data;
        // ... add lots of data ...
        // ... remove most data ...
        bulk_data.shrink_to_fit(); // Reclaim unused memory
        
        // 4. Use circular buffer for streaming data
        std::cout << "4. Streaming Data Management:\n";
        boost::circular_buffer<double> sensor_data(1000); // Fixed memory footprint
        
        // 5. Consider memory pools for frequent allocations
        std::cout << "5. Memory Pools (concept):\n";
        std::cout << "   - Use boost::pool for frequent same-size allocations\n";
        std::cout << "   - Consider custom allocators for containers\n";
    }
    
    void analyze_memory_patterns() {
        std::cout << "Memory Pattern Analysis:\n";
        
        // Analyze different container memory patterns
        std::cout << "Container Memory Patterns:\n";
        std::cout << "- std::vector: Contiguous, may over-allocate\n";
        std::cout << "- std::map: Scattered nodes, per-element overhead\n";
        std::cout << "- flat_map: Contiguous, cache-friendly\n";
        std::cout << "- multi_index: Multiple data structures, higher overhead\n";
        std::cout << "- circular_buffer: Fixed size, predictable usage\n";
    }
};
```

### Error Handling and Robustness

#### 6. Safe Container Operations
```cpp
// Error handling best practices
class SafeContainerOperations {
public:
    void demonstrate_safe_operations() {
        std::cout << "Safe Container Operations:\n\n";
        
        // 1. Check container state before operations
        boost::container::flat_map<int, std::string> map;
        
        // Safe access
        auto safe_find = [&](int key) -> std::string {
            auto it = map.find(key);
            return (it != map.end()) ? it->second : "";
        };
        
        // 2. Handle capacity limits
        boost::container::static_vector<int, 10> fixed_vec;
        
        auto safe_push = [&](int value) -> bool {
            if (fixed_vec.size() < fixed_vec.capacity()) {
                fixed_vec.push_back(value);
                return true;
            }
            return false; // Capacity exceeded
        };
        
        // 3. Exception safety
        try {
            // Multi-index operations can throw
            EmployeeDatabase db;
            db.addEmployee(Employee(1, "Test", "Dept", -1000)); // Invalid salary?
        } catch (const std::exception& e) {
            std::cout << "Multi-index error: " << e.what() << "\n";
        }
        
        // 4. Iterator safety
        boost::container::stable_vector<int> stable_vec;
        stable_vec.push_back(1);
        auto it = stable_vec.begin();
        
        // Safe with stable_vector
        stable_vec.insert(stable_vec.begin(), 0);
        std::cout << "Iterator still valid: " << *it << "\n"; // Still safe
    }
    
    void demonstrate_error_recovery() {
        std::cout << "Error Recovery Patterns:\n";
        
        // 1. Backup and restore for critical operations
        class BackupManager {
            std::vector<Employee> backup_;
            
        public:
            void create_backup(const EmployeeDatabase& db) {
                // Create backup before risky operations
                backup_.clear();
                // ... copy data ...
            }
            
            void restore_backup(EmployeeDatabase& db) {
                // Restore from backup if operation fails
                // ... restore data ...
            }
        };
        
        // 2. Validation before insertion
        auto validate_employee = [](const Employee& emp) -> bool {
            return emp.getId() > 0 && 
                   !emp.getName().empty() && 
                   emp.getSalary() >= 0;
        };
        
        // 3. Graceful degradation
        std::cout << "Implement graceful degradation for resource constraints\n";
    }
};
```

### Thread Safety Considerations

#### 7. Concurrent Access Patterns
```cpp
// Thread safety guidelines
class ThreadSafetyGuidelines {
public:
    void analyze_thread_safety() {
        std::cout << "Thread Safety Analysis:\n\n";
        
        std::cout << "Boost Containers Thread Safety:\n";
        std::cout << "- NOT thread-safe by default\n";
        std::cout << "- Multiple readers: Generally safe\n";
        std::cout << "- Any writer: Requires synchronization\n";
        std::cout << "- Iterator invalidation: Check container-specific rules\n\n";
        
        std::cout << "Synchronization Strategies:\n";
        std::cout << "1. Mutex protection for shared containers\n";
        std::cout << "2. Reader-writer locks for read-heavy workloads\n";
        std::cout << "3. Lock-free alternatives for high-performance scenarios\n";
        std::cout << "4. Thread-local containers when possible\n";
    }
    
    // Example: Thread-safe wrapper
    template<typename Container>
    class ThreadSafeContainer {
    private:
        mutable std::shared_mutex mutex_;
        Container container_;
        
    public:
        // Read operations (shared lock)
        template<typename Key>
        auto find(const Key& key) const {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return container_.find(key);
        }
        
        size_t size() const {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return container_.size();
        }
        
        // Write operations (exclusive lock)
        template<typename... Args>
        auto insert(Args&&... args) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            return container_.insert(std::forward<Args>(args)...);
        }
        
        template<typename Key>
        auto erase(const Key& key) {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            return container_.erase(key);
        }
    };
};
```

### Summary of Best Practices

```cpp
// Complete best practices checklist
class BestPracticesChecklist {
public:
    void print_checklist() {
        std::cout << "Boost Containers Best Practices Checklist:\n\n";
        
        std::cout << "□ Container Selection:\n";
        std::cout << "  □ Analyzed access patterns\n";
        std::cout << "  □ Considered memory constraints\n";
        std::cout << "  □ Evaluated performance requirements\n";
        std::cout << "  □ Profiled actual usage scenarios\n\n";
        
        std::cout << "□ Performance Optimization:\n";
        std::cout << "  □ Reserved capacity for growing containers\n";
        std::cout << "  □ Used appropriate container for data size\n";
        std::cout << "  □ Optimized for cache locality when needed\n";
        std::cout << "  □ Minimized index overhead in multi-index\n\n";
        
        std::cout << "□ Memory Management:\n";
        std::cout << "  □ Used small_vector for small collections\n";
        std::cout << "  □ Used static_vector for fixed-size collections\n";
        std::cout << "  □ Used circular_buffer for streaming data\n";
        std::cout << "  □ Shrunk containers after bulk operations\n\n";
        
        std::cout << "□ Error Handling:\n";
        std::cout << "  □ Validated input before operations\n";
        std::cout << "  □ Handled capacity limits appropriately\n";
        std::cout << "  □ Implemented proper exception safety\n";
        std::cout << "  □ Considered iterator invalidation rules\n\n";
        
        std::cout << "□ Thread Safety:\n";
        std::cout << "  □ Added synchronization for concurrent access\n";
        std::cout << "  □ Used appropriate locking strategy\n";
        std::cout << "  □ Considered lock-free alternatives\n";
        std::cout << "  □ Documented thread safety guarantees\n\n";
    }
};
```

## Assessment and Learning Objectives

### Self-Assessment Checklist

By the end of this section, you should be able to:

#### Knowledge Assessment
□ **Container Selection**: Choose appropriate containers based on access patterns, size constraints, and performance requirements  
□ **Performance Analysis**: Understand time and space complexity trade-offs between different container types  
□ **Memory Management**: Apply memory optimization techniques using small_vector, static_vector, and circular_buffer  
□ **Multi-Index Design**: Design efficient multi-index schemas for complex data access patterns  
□ **Bimap Usage**: Implement bidirectional mappings with appropriate relation types  
□ **Circular Buffer Applications**: Use circular buffers for streaming data and sliding window calculations  

#### Practical Skills Assessment
□ **Benchmarking**: Can measure and compare performance of different container implementations  
□ **Index Design**: Can design multi-index containers with optimal index selection  
□ **Memory Profiling**: Can analyze and optimize memory usage patterns  
□ **Error Handling**: Can implement robust error handling and validation  
□ **Thread Safety**: Can add appropriate synchronization for concurrent access  

### Practical Assessment Exercises

#### Exercise 1: Container Performance Analysis
```cpp
// TODO: Implement and analyze performance characteristics
class PerformanceAnalyzer {
public:
    // Compare containers for specific workloads
    void benchmark_insert_performance(size_t data_size);
    void benchmark_lookup_performance(size_t data_size, double hit_ratio);
    void benchmark_iteration_performance(size_t data_size);
    void benchmark_memory_usage(size_t data_size);
    
    // Generate performance report
    void generate_performance_report();
};

// Requirements:
// - Test with datasets: 100, 1K, 10K, 100K elements
// - Compare std::map vs flat_map vs multi_index
// - Measure insertion, lookup, iteration, and memory usage
// - Document when to use each container type
```

#### Exercise 2: Advanced Multi-Index System
```cpp
// TODO: Design and implement a comprehensive database system
class LibraryManagementSystem {
private:
    // Design multi-index container for books with indices:
    // - ISBN (primary key)
    // - Title (for text searches)
    // - Author (non-unique)
    // - Publication year (for range queries)
    // - Category (hashed for fast lookups)
    // - Available copies (for availability queries)
    // - Composite index (Category + Publication year)
    
public:
    // Implement comprehensive functionality
    void add_book(const Book& book);
    void remove_book(const std::string& isbn);
    std::vector<Book> search_by_title(const std::string& title);
    std::vector<Book> search_by_author(const std::string& author);
    std::vector<Book> find_books_by_year_range(int start_year, int end_year);
    std::vector<Book> find_available_books_in_category(const std::string& category);
    void generate_inventory_report();
    void export_catalog_to_csv();
    
    // Advanced features
    void implement_book_recommendations();
    void track_borrowing_history();
    void implement_search_autocomplete();
};

// Success criteria:
// - Handle 50,000+ books efficiently
// - Support complex multi-criteria searches
// - Maintain data consistency across all indices
// - Provide sub-second response times for all queries
```

#### Exercise 3: Real-Time Data Processing Pipeline
```cpp
// TODO: Build a real-time monitoring system
class RealTimeMonitoringSystem {
private:
    // Use multiple circular buffers for different data streams
    boost::circular_buffer<MetricData> cpu_metrics_;
    boost::circular_buffer<MetricData> memory_metrics_;
    boost::circular_buffer<MetricData> network_metrics_;
    boost::circular_buffer<AlertEvent> recent_alerts_;
    
public:
    // Core functionality
    void process_metric_data(const MetricData& data);
    MovingStatistics calculate_moving_average(MetricType type, std::chrono::minutes window);
    std::vector<Anomaly> detect_anomalies();
    void trigger_alerts(const std::vector<Anomaly>& anomalies);
    
    // Advanced analytics
    void implement_trend_analysis();
    void calculate_cross_correlation();
    void predict_future_values();
    void implement_adaptive_thresholds();
    
    // Reporting
    void generate_real_time_dashboard();
    void export_historical_data();
};

// Performance requirements:
// - Process 10,000+ metrics per second
// - Maintain 24-hour rolling windows
// - Detect anomalies in real-time
// - Generate alerts within 1 second of detection
```

### Comprehensive Quiz Questions

#### Conceptual Questions
1. **Container Selection**: When would you choose `flat_map` over `std::map`? Provide three specific scenarios with justification.

2. **Memory Optimization**: Explain the benefits of `small_vector<T, 8>` over `std::vector<T>`. What factors determine the optimal inline capacity?

3. **Multi-Index Design**: Design a multi-index container for a social media platform that needs to efficiently query users by: user ID, username, email, registration date, and location. Explain your index choices.

4. **Circular Buffer Applications**: Describe three different use cases where `circular_buffer` would be superior to `std::deque`. Include performance justifications.

5. **Bimap Relations**: Compare the different bimap relation types and provide use cases for each combination.

#### Technical Questions
6. **Performance Analysis**: Given a read-heavy workload with 100,000 key-value pairs and 1 million lookups, compare the expected performance of `flat_map` vs `std::map`. Include cache considerations.

7. **Thread Safety**: How would you make a multi-index container thread-safe while maintaining good performance for read-heavy workloads?

8. **Memory Management**: Calculate the approximate memory overhead of storing 10,000 integers in:
   - `std::vector<int>`
   - `std::map<int, int>`
   - `boost::multi_index_container` with 3 indices

9. **Iterator Stability**: Explain the iterator invalidation rules for `stable_vector` and provide a use case where this is critical.

10. **Composite Keys**: Design an efficient composite key for querying employees by department and salary range. Explain the ordering strategy.

### Project-Based Assessment

#### Final Project: Distributed Cache System
Design and implement a high-performance distributed cache system using Boost containers:

**Requirements:**
- Support for multiple data types
- LRU eviction policy using circular buffers
- Multi-index access (by key, access time, frequency)
- Thread-safe operations
- Performance monitoring and statistics
- Configurable memory limits
- Persistence layer integration

**Evaluation Criteria:**
- Container selection justification
- Performance benchmarks
- Memory efficiency analysis
- Code quality and documentation
- Error handling and robustness
- Scalability considerations

### Additional Learning Resources

#### Recommended Reading
- **Primary**: "Boost C++ Libraries" by Boris Schäling - Containers chapter
- **Advanced**: "Effective STL" by Scott Meyers - Container selection principles
- **Performance**: "Optimized C++" by Kurt Guntheroth - Container performance analysis

#### Online Resources
- [Boost Container Documentation](https://www.boost.org/doc/libs/release/doc/html/container.html)
- [Boost Multi-Index Tutorial](https://www.boost.org/doc/libs/release/libs/multi_index/doc/tutorial/index.html)
- [Performance Analysis Tools](https://github.com/google/benchmark) for container benchmarking

#### Hands-On Labs
1. **Container Benchmarking Lab**: Implement comprehensive performance testing framework
2. **Multi-Index Design Lab**: Build complex database-like systems
3. **Memory Optimization Lab**: Analyze and optimize real-world applications
4. **Thread Safety Lab**: Implement concurrent data structures

### Certification Criteria

To demonstrate mastery of Boost Containers and Data Structures:

□ **Understanding**: Can explain when and why to use each container type  
□ **Implementation**: Can implement complex systems using appropriate containers  
□ **Optimization**: Can analyze and optimize performance and memory usage  
□ **Design**: Can design efficient multi-index schemas and data access patterns  
□ **Debugging**: Can identify and fix container-related performance issues  
□ **Best Practices**: Follows established patterns for robustness and maintainability  

### Next Steps

Upon completion of this section, you should be ready to:
- Move on to [String Processing and Text Handling](04_String_Processing_Text_Handling.md)
- Apply container knowledge to real-world projects
- Contribute to open-source projects using Boost containers
- Mentor others in container selection and optimization
- Design high-performance data structures for specific domains
