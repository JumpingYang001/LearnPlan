# Project 1: Multi-Indexed Database

*Estimated Duration: 2-3 weeks*
*Difficulty: Intermediate*

## Project Overview

Create a comprehensive database system using Boost.MultiIndex that supports multiple access patterns and complex queries. This project demonstrates the power of multi-indexed containers for organizing and accessing data efficiently.

## Learning Objectives

- Master Boost.MultiIndex container design and implementation
- Understand different index types and their performance characteristics
- Implement complex query operations and data manipulation
- Design efficient data structures for real-world applications

## Project Requirements

### Core Features

1. **Student Management System**
   - Store student records with multiple attributes
   - Support queries by ID, name, GPA, major, and enrollment date
   - Implement range queries and complex filters

2. **Index Types to Implement**
   - Unique ordered index (Student ID)
   - Non-unique ordered index (GPA, enrollment date)
   - Hashed index (student name for fast lookup)
   - Composite index (major + GPA)
   - Random access index (for positional operations)

3. **Query Operations**
   - Find by exact match
   - Range queries (GPA between values, dates in range)
   - Top N queries (highest/lowest GPA)
   - Composite queries (students in major with GPA > threshold)

### Advanced Features

4. **Data Persistence**
   - Save/load database to/from file
   - Support for different serialization formats

5. **Statistics and Reporting**
   - Generate reports by major, GPA distribution
   - Calculate statistics (average GPA, enrollment trends)

6. **Data Validation and Integrity**
   - Input validation for all fields
   - Referential integrity checks
   - Duplicate detection and handling

## Implementation Guide

### Step 1: Define the Student Record Structure

```cpp
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <string>
#include <boost/date_time/gregorian/gregorian.hpp>

struct Student {
    int id;
    std::string first_name;
    std::string last_name;
    std::string major;
    double gpa;
    boost::gregorian::date enrollment_date;
    
    Student(int id, const std::string& fname, const std::string& lname,
            const std::string& major, double gpa, const boost::gregorian::date& date)
        : id(id), first_name(fname), last_name(lname), 
          major(major), gpa(gpa), enrollment_date(date) {}
    
    std::string full_name() const {
        return first_name + " " + last_name;
    }
};
```

### Step 2: Define the Multi-Index Container

```cpp
namespace bmi = boost::multi_index;

// Tag structures for index identification
struct by_id {};
struct by_name {};
struct by_gpa {};
struct by_major {};
struct by_enrollment_date {};
struct by_major_gpa {};
struct by_position {};

typedef bmi::multi_index_container<
    Student,
    bmi::indexed_by<
        // Unique ordered index by ID
        bmi::ordered_unique<
            bmi::tag<by_id>,
            bmi::member<Student, int, &Student::id>
        >,
        
        // Hashed index by full name
        bmi::hashed_non_unique<
            bmi::tag<by_name>,
            bmi::const_mem_fun<Student, std::string, &Student::full_name>
        >,
        
        // Ordered index by GPA (descending)
        bmi::ordered_non_unique<
            bmi::tag<by_gpa>,
            bmi::member<Student, double, &Student::gpa>,
            std::greater<double>
        >,
        
        // Ordered index by major
        bmi::ordered_non_unique<
            bmi::tag<by_major>,
            bmi::member<Student, std::string, &Student::major>
        >,
        
        // Ordered index by enrollment date
        bmi::ordered_non_unique<
            bmi::tag<by_enrollment_date>,
            bmi::member<Student, boost::gregorian::date, &Student::enrollment_date>
        >,
        
        // Composite index by major and GPA
        bmi::ordered_non_unique<
            bmi::tag<by_major_gpa>,
            bmi::composite_key<
                Student,
                bmi::member<Student, std::string, &Student::major>,
                bmi::member<Student, double, &Student::gpa>
            >
        >,
        
        // Random access index for positional operations
        bmi::random_access<bmi::tag<by_position>>
    >
> StudentDatabase;
```

### Step 3: Database Management Class

```cpp
class StudentDB {
private:
    StudentDatabase students_;
    
public:
    // Basic operations
    bool add_student(const Student& student);
    bool remove_student(int id);
    bool update_student(int id, const Student& updated_student);
    
    // Query operations
    const Student* find_by_id(int id) const;
    std::vector<Student> find_by_name(const std::string& name) const;
    std::vector<Student> find_by_major(const std::string& major) const;
    std::vector<Student> find_by_gpa_range(double min_gpa, double max_gpa) const;
    std::vector<Student> find_top_students(size_t count) const;
    
    // Advanced queries
    std::vector<Student> find_students_in_major_with_min_gpa(
        const std::string& major, double min_gpa) const;
    std::vector<Student> find_students_enrolled_between_dates(
        const boost::gregorian::date& start, 
        const boost::gregorian::date& end) const;
    
    // Statistics
    double calculate_average_gpa() const;
    std::map<std::string, size_t> get_major_distribution() const;
    std::map<std::string, double> get_average_gpa_by_major() const;
    
    // Utility
    size_t size() const { return students_.size(); }
    bool empty() const { return students_.empty(); }
    void clear() { students_.clear(); }
    
    // Iteration support
    auto begin() const { return students_.begin(); }
    auto end() const { return students_.end(); }
};
```

### Step 4: Implementation Examples

```cpp
bool StudentDB::add_student(const Student& student) {
    auto result = students_.insert(student);
    return result.second; // true if inserted, false if already exists
}

const Student* StudentDB::find_by_id(int id) const {
    auto& id_index = students_.get<by_id>();
    auto it = id_index.find(id);
    return (it != id_index.end()) ? &(*it) : nullptr;
}

std::vector<Student> StudentDB::find_by_gpa_range(double min_gpa, double max_gpa) const {
    auto& gpa_index = students_.get<by_gpa>();
    std::vector<Student> result;
    
    // Note: GPA index is in descending order
    auto lower = gpa_index.lower_bound(max_gpa);
    auto upper = gpa_index.upper_bound(min_gpa);
    
    for (auto it = lower; it != upper; ++it) {
        result.push_back(*it);
    }
    
    return result;
}

std::vector<Student> StudentDB::find_students_in_major_with_min_gpa(
    const std::string& major, double min_gpa) const {
    
    auto& composite_index = students_.get<by_major_gpa>();
    std::vector<Student> result;
    
    auto lower = composite_index.lower_bound(boost::make_tuple(major, min_gpa));
    auto upper = composite_index.upper_bound(boost::make_tuple(major));
    
    for (auto it = lower; it != upper; ++it) {
        result.push_back(*it);
    }
    
    return result;
}

double StudentDB::calculate_average_gpa() const {
    if (students_.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& student : students_) {
        total += student.gpa;
    }
    
    return total / students_.size();
}
```

## Sample Usage and Testing

### Step 5: Create Test Data and Demonstrate Usage

```cpp
#include <iostream>
#include <iomanip>

void populate_sample_data(StudentDB& db) {
    using namespace boost::gregorian;
    
    // Add sample students
    db.add_student(Student(1001, "Alice", "Johnson", "Computer Science", 3.8, date(2020, 9, 1)));
    db.add_student(Student(1002, "Bob", "Smith", "Mathematics", 3.6, date(2020, 9, 1)));
    db.add_student(Student(1003, "Charlie", "Brown", "Computer Science", 3.9, date(2021, 1, 15)));
    db.add_student(Student(1004, "Diana", "Wilson", "Physics", 3.7, date(2021, 9, 1)));
    db.add_student(Student(1005, "Eve", "Davis", "Computer Science", 3.5, date(2022, 1, 10)));
    db.add_student(Student(1006, "Frank", "Miller", "Mathematics", 3.4, date(2022, 9, 1)));
    db.add_student(Student(1007, "Grace", "Taylor", "Physics", 3.9, date(2023, 1, 15)));
}

void demonstrate_queries(const StudentDB& db) {
    std::cout << "=== Student Database Queries ===\n\n";
    
    // Query by ID
    std::cout << "1. Find student by ID (1003):\n";
    if (const auto* student = db.find_by_id(1003)) {
        std::cout << "   " << student->full_name() << " - " << student->major 
                  << " - GPA: " << student->gpa << "\n\n";
    }
    
    // Query by major
    std::cout << "2. Students in Computer Science:\n";
    auto cs_students = db.find_by_major("Computer Science");
    for (const auto& student : cs_students) {
        std::cout << "   " << student.full_name() << " - GPA: " << student.gpa << "\n";
    }
    std::cout << "\n";
    
    // Top students
    std::cout << "3. Top 3 students by GPA:\n";
    auto top_students = db.find_top_students(3);
    for (const auto& student : top_students) {
        std::cout << "   " << student.full_name() << " - " << student.major 
                  << " - GPA: " << student.gpa << "\n";
    }
    std::cout << "\n";
    
    // Complex query
    std::cout << "4. Computer Science students with GPA >= 3.7:\n";
    auto filtered = db.find_students_in_major_with_min_gpa("Computer Science", 3.7);
    for (const auto& student : filtered) {
        std::cout << "   " << student.full_name() << " - GPA: " << student.gpa << "\n";
    }
    std::cout << "\n";
    
    // Statistics
    std::cout << "5. Statistics:\n";
    std::cout << "   Total students: " << db.size() << "\n";
    std::cout << "   Average GPA: " << std::fixed << std::setprecision(2) 
              << db.calculate_average_gpa() << "\n";
    
    auto major_dist = db.get_major_distribution();
    std::cout << "   Major distribution:\n";
    for (const auto& pair : major_dist) {
        std::cout << "     " << pair.first << ": " << pair.second << " students\n";
    }
}

int main() {
    StudentDB db;
    populate_sample_data(db);
    demonstrate_queries(db);
    return 0;
}
```

## Advanced Features Implementation

### Data Persistence

```cpp
class StudentDB {
public:
    // Serialization support
    bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);
    
private:
    void serialize_student(std::ostream& os, const Student& student) const;
    Student deserialize_student(std::istream& is) const;
};
```

### Report Generation

```cpp
class ReportGenerator {
public:
    static void generate_gpa_report(const StudentDB& db, std::ostream& os);
    static void generate_major_summary(const StudentDB& db, std::ostream& os);
    static void generate_enrollment_trends(const StudentDB& db, std::ostream& os);
};
```

## Testing Strategy

### Unit Tests

1. **Index Functionality Tests**
   - Test each index type individually
   - Verify query correctness and performance
   - Test edge cases (empty database, single record)

2. **Data Integrity Tests**
   - Test duplicate ID prevention
   - Validate data consistency across indices
   - Test concurrent access scenarios

3. **Performance Tests**
   - Benchmark query performance with large datasets
   - Compare with single-index alternatives
   - Memory usage analysis

### Integration Tests

1. **End-to-End Scenarios**
   - Complete student lifecycle (add, update, graduate)
   - Batch operations and bulk data loading
   - Error handling and recovery

## Performance Considerations

1. **Index Selection**
   - Choose appropriate index types for query patterns
   - Consider memory overhead of multiple indices
   - Balance query speed vs update performance

2. **Memory Management**
   - Use memory pools for frequent allocations
   - Consider string interning for repeated values
   - Monitor memory fragmentation

3. **Query Optimization**
   - Use most selective index for filtering
   - Minimize object copying in result sets
   - Implement result caching for expensive queries

## Extension Ideas

1. **Web Interface**
   - REST API using Boost.Beast
   - JSON serialization with Boost.PropertyTree
   - Authentication and authorization

2. **Advanced Analytics**
   - Trend analysis using time-series data
   - Predictive modeling for academic performance
   - Data visualization integration

3. **Distributed Database**
   - Sharding across multiple nodes
   - Replication and consistency
   - Load balancing for queries

## Assessment Criteria

- [ ] Correctly implements all required index types
- [ ] Demonstrates understanding of multi-index design principles
- [ ] Implements efficient query algorithms
- [ ] Includes comprehensive error handling
- [ ] Provides performance analysis and optimization
- [ ] Creates thorough test suite
- [ ] Documents design decisions and trade-offs

## Deliverables

1. Complete source code with documentation
2. Test suite with performance benchmarks
3. User manual with API documentation
4. Performance analysis report
5. Future enhancement roadmap
