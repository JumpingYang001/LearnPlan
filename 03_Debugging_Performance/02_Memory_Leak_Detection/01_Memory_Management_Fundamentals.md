# Memory Management Fundamentals

*Duration: 1.5 weeks*

## Overview
This section covers the fundamentals of memory management in C/C++, including different types of memory allocation, common memory-related issues, debugging techniques, and modern C++ memory management strategies. Understanding memory management is crucial for writing efficient, safe, and leak-free programs.

## Memory Layout in C/C++ Programs

Before diving into memory allocation, it's essential to understand how memory is organized in a typical C/C++ program:

```
High Address
┌─────────────────┐
│     Stack       │  ← Local variables, function parameters, return addresses
│       ↓         │     (grows downward)
├─────────────────┤
│                 │
│   Free Space    │
│                 │
├─────────────────┤
│       ↑         │
│      Heap       │  ← Dynamic memory allocation (malloc/new)
│                 │     (grows upward)
├─────────────────┤
│  Uninitialized  │  ← BSS segment (global/static variables set to zero)
│   Data (BSS)    │
├─────────────────┤
│   Initialized   │  ← Global/static variables with initial values
│      Data       │
├─────────────────┤
│      Text       │  ← Program code (read-only)
│    (Code)       │
└─────────────────┘
Low Address
```

### Memory Segments Explained

**1. Text Segment (Code)**
- Contains the executable instructions
- Read-only and shared among multiple instances
- Size fixed at compile time

**2. Data Segment**
- **Initialized Data**: Global and static variables with initial values
- **BSS (Block Started by Symbol)**: Global and static variables initialized to zero

**3. Heap**
- Used for dynamic memory allocation (`malloc`, `new`)
- Managed by programmer
- Grows upward (toward higher addresses)
- Can lead to fragmentation

**4. Stack**
- Used for function calls, local variables, parameters
- Automatically managed
- Grows downward (toward lower addresses)
- Limited size (typically 1-8 MB)

### Demonstrating Memory Segments

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Global variables (Data segment)
int global_initialized = 42;
int global_uninitialized;  // BSS segment
static int static_var = 100;

void demonstrate_memory_layout() {
    // Stack variables
    int local_var = 10;
    char local_array[100];
    
    // Heap allocation
    int* heap_ptr = malloc(sizeof(int));
    *heap_ptr = 99;
    
    printf("=== Memory Layout Demonstration ===\n");
    printf("Text segment (function): %p\n", (void*)demonstrate_memory_layout);
    printf("Global initialized:      %p (value: %d)\n", (void*)&global_initialized, global_initialized);
    printf("Global uninitialized:    %p (value: %d)\n", (void*)&global_uninitialized, global_uninitialized);
    printf("Static variable:         %p (value: %d)\n", (void*)&static_var, static_var);
    printf("Local variable:          %p (value: %d)\n", (void*)&local_var, local_var);
    printf("Local array:             %p\n", (void*)local_array);
    printf("Heap allocation:         %p (value: %d)\n", (void*)heap_ptr, *heap_ptr);
    printf("Stack pointer (approx):  %p\n", (void*)&local_var);
    
    free(heap_ptr);
}
```

## Memory Allocation in C/C++

### Stack Allocation
Stack allocation is automatic and fast, but limited in size and scope.

```c
#include <stdio.h>

void stack_allocation_demo() {
    // Stack allocation - automatic cleanup
    int local_var = 42;                    // 4 bytes
    char buffer[1024];                     // 1024 bytes
    int array[100];                        // 400 bytes
    
    printf("Stack variable address: %p\n", (void*)&local_var);
    printf("Stack usage: ~%zu bytes\n", sizeof(local_var) + sizeof(buffer) + sizeof(array));
    
    // Variables automatically destroyed when function exits
} // All stack variables are automatically cleaned up here

void demonstrate_stack_growth() {
    static int call_depth = 0;
    int local_var;
    
    printf("Call depth: %d, Local var address: %p\n", call_depth, (void*)&local_var);
    
    if (call_depth < 5) {
        call_depth++;
        demonstrate_stack_growth(); // Recursive call
        call_depth--;
    }
}
```

### Heap Allocation in C

Dynamic memory allocation gives you control over memory lifetime and size.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void basic_heap_allocation() {
    printf("=== Basic Heap Allocation ===\n");
    
    // Single integer allocation
    int* single_int = malloc(sizeof(int));
    if (single_int == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return;
    }
    *single_int = 42;
    printf("Single int: %d at address %p\n", *single_int, (void*)single_int);
    free(single_int);
    
    // Array allocation
    int size = 10;
    int* array = malloc(size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Array allocation failed!\n");
        return;
    }
    
    // Initialize array
    for (int i = 0; i < size; i++) {
        array[i] = i * i;
    }
    
    printf("Array contents: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    
    free(array);
}

void advanced_heap_allocation() {
    printf("=== Advanced Heap Allocation ===\n");
    
    // calloc - initializes to zero
    int* zero_array = calloc(5, sizeof(int));
    printf("Calloc array (should be zeros): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", zero_array[i]);
    }
    printf("\n");
    
    // realloc - resize existing memory
    zero_array = realloc(zero_array, 10 * sizeof(int));
    if (zero_array == NULL) {
        fprintf(stderr, "Realloc failed!\n");
        return;
    }
    
    // Initialize new elements
    for (int i = 5; i < 10; i++) {
        zero_array[i] = i;
    }
    
    printf("Realloc array: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", zero_array[i]);
    }
    printf("\n");
    
    free(zero_array);
}

// Dynamic string allocation example
char* create_dynamic_string(const char* source) {
    if (source == NULL) return NULL;
    
    size_t len = strlen(source);
    char* new_string = malloc(len + 1); // +1 for null terminator
    
    if (new_string == NULL) {
        return NULL; // Allocation failed
    }
    
    strcpy(new_string, source);
    return new_string; // Caller must free this memory
}

void string_allocation_demo() {
    printf("=== String Allocation Demo ===\n");
    
    char* dynamic_str = create_dynamic_string("Hello, Dynamic World!");
    if (dynamic_str != NULL) {
        printf("Dynamic string: %s\n", dynamic_str);
        printf("String address: %p\n", (void*)dynamic_str);
        free(dynamic_str); // Important: free the allocated memory
    }
}
```

### Heap Allocation in C++

C++ provides both C-style allocation and more type-safe alternatives.

```cpp
#include <iostream>
#include <vector>
#include <string>

class Person {
private:
    std::string name;
    int age;
    
public:
    Person(const std::string& n, int a) : name(n), age(a) {
        std::cout << "Person constructor: " << name << std::endl;
    }
    
    ~Person() {
        std::cout << "Person destructor: " << name << std::endl;
    }
    
    void introduce() const {
        std::cout << "Hi, I'm " << name << ", age " << age << std::endl;
    }
};

void cpp_heap_allocation() {
    std::cout << "=== C++ Heap Allocation ===\n";
    
    // Single object allocation
    Person* person1 = new Person("Alice", 25);
    person1->introduce();
    delete person1; // Must manually delete
    
    // Array allocation
    Person* people = new Person[2]{
        Person("Bob", 30),
        Person("Charlie", 35)
    };
    
    for (int i = 0; i < 2; i++) {
        people[i].introduce();
    }
    
    delete[] people; // Must use delete[] for arrays
    
    // Better approach: use containers
    std::vector<Person> person_vector;
    person_vector.emplace_back("David", 28);
    person_vector.emplace_back("Eve", 32);
    
    for (const auto& person : person_vector) {
        person.introduce();
    }
    // Automatic cleanup when vector goes out of scope
}

// Memory allocation with placement new
void placement_new_demo() {
    std::cout << "=== Placement New Demo ===\n";
    
    // Allocate raw memory
    char buffer[sizeof(Person)];
    
    // Construct object in specific memory location
    Person* person = new(buffer) Person("Placement", 40);
    person->introduce();
    
    // Must manually call destructor for placement new
    person->~Person();
    
    // No delete needed since we didn't use regular new
}
```

## Common Memory Issues

Understanding and preventing memory issues is crucial for robust C/C++ programming. Let's explore each type with detailed examples and solutions.

### 1. Memory Leaks

A memory leak occurs when allocated memory is not freed, causing the program to consume more and more memory over time.

#### Simple Memory Leak Example
```c
#include <stdio.h>
#include <stdlib.h>

// BAD: Memory leak example
void memory_leak_example() {
    int* ptr = malloc(100 * sizeof(int));
    
    if (ptr == NULL) {
        printf("Allocation failed\n");
        return;
    }
    
    // Do some work with ptr
    for (int i = 0; i < 100; i++) {
        ptr[i] = i;
    }
    
    // PROBLEM: forgot to call free(ptr)!
    // This creates a memory leak
    
    return; // Memory is lost forever
}

// GOOD: Proper memory management
void proper_memory_management() {
    int* ptr = malloc(100 * sizeof(int));
    
    if (ptr == NULL) {
        printf("Allocation failed\n");
        return;
    }
    
    // Do some work with ptr
    for (int i = 0; i < 100; i++) {
        ptr[i] = i;
    }
    
    // SOLUTION: Always free allocated memory
    free(ptr);
    ptr = NULL; // Good practice: set to NULL after freeing
}
```

#### Complex Memory Leak Scenarios
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int data;
    struct Node* next;
} Node;

// BAD: Linked list with memory leak
Node* create_leaky_list(int size) {
    Node* head = NULL;
    Node* current = NULL;
    
    for (int i = 0; i < size; i++) {
        Node* new_node = malloc(sizeof(Node));
        if (new_node == NULL) {
            printf("Allocation failed\n");
            // PROBLEM: Previous nodes are not freed on failure!
            return NULL;
        }
        
        new_node->data = i;
        new_node->next = NULL;
        
        if (head == NULL) {
            head = new_node;
            current = new_node;
        } else {
            current->next = new_node;
            current = new_node;
        }
    }
    
    return head;
}

// GOOD: Proper cleanup on failure
Node* create_safe_list(int size) {
    Node* head = NULL;
    Node* current = NULL;
    
    for (int i = 0; i < size; i++) {
        Node* new_node = malloc(sizeof(Node));
        if (new_node == NULL) {
            printf("Allocation failed, cleaning up...\n");
            // Clean up previously allocated nodes
            while (head != NULL) {
                Node* temp = head;
                head = head->next;
                free(temp);
            }
            return NULL;
        }
        
        new_node->data = i;
        new_node->next = NULL;
        
        if (head == NULL) {
            head = new_node;
            current = new_node;
        } else {
            current->next = new_node;
            current = new_node;
        }
    }
    
    return head;
}

// Proper cleanup function
void free_list(Node* head) {
    while (head != NULL) {
        Node* temp = head;
        head = head->next;
        free(temp);
    }
}
```

### 2. Double Free

Double free occurs when you call `free()` on the same memory address more than once.

```c
#include <stdio.h>
#include <stdlib.h>

// BAD: Double free example
void double_free_example() {
    int* ptr = malloc(sizeof(int));
    *ptr = 42;
    
    printf("Value: %d\n", *ptr);
    
    free(ptr);  // First free - OK
    
    // Some other code...
    
    free(ptr);  // PROBLEM: Double free - undefined behavior!
    // This can cause:
    // - Program crash
    // - Heap corruption
    // - Security vulnerabilities
}

// GOOD: Preventing double free
void prevent_double_free() {
    int* ptr = malloc(sizeof(int));
    *ptr = 42;
    
    printf("Value: %d\n", *ptr);
    
    free(ptr);  // Free the memory
    ptr = NULL; // SOLUTION: Set pointer to NULL
    
    // Later in code...
    if (ptr != NULL) {  // Check before freeing
        free(ptr);
    }
    // OR
    free(ptr);  // free(NULL) is safe and does nothing
}

// Double free with multiple pointers
void multiple_pointer_issue() {
    int* ptr1 = malloc(sizeof(int));
    int* ptr2 = ptr1;  // Both point to same memory
    
    *ptr1 = 100;
    
    free(ptr1);  // Free the memory
    ptr1 = NULL;
    
    // PROBLEM: ptr2 still points to freed memory
    free(ptr2);  // Double free!
    
    // SOLUTION: Set all pointers to NULL
    // ptr2 = NULL;
}
```

### 3. Use-After-Free

Use-after-free occurs when you access memory that has already been freed.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BAD: Use-after-free example
void use_after_free_example() {
    char* buffer = malloc(100);
    strcpy(buffer, "Hello, World!");
    
    printf("Before free: %s\n", buffer);
    
    free(buffer);  // Free the memory
    
    // PROBLEM: Using freed memory
    printf("After free: %s\n", buffer);  // Undefined behavior!
    
    // Even worse:
    strcpy(buffer, "New data");  // Writing to freed memory!
}

// GOOD: Avoiding use-after-free
void prevent_use_after_free() {
    char* buffer = malloc(100);
    strcpy(buffer, "Hello, World!");
    
    printf("Before free: %s\n", buffer);
    
    free(buffer);   // Free the memory
    buffer = NULL;  // SOLUTION: Set to NULL immediately
    
    // Safe check before use
    if (buffer != NULL) {
        printf("After free: %s\n", buffer);
    } else {
        printf("Buffer is NULL, cannot access\n");
    }
}

// Dangling pointer example
typedef struct {
    char* name;
    int age;
} Person;

Person* create_person(const char* name, int age) {
    Person* p = malloc(sizeof(Person));
    if (p == NULL) return NULL;
    
    p->name = malloc(strlen(name) + 1);
    if (p->name == NULL) {
        free(p);
        return NULL;
    }
    
    strcpy(p->name, name);
    p->age = age;
    return p;
}

void free_person(Person* p) {
    if (p != NULL) {
        free(p->name);  // Free nested allocation first
        p->name = NULL;
        free(p);
    }
}

// BAD: Dangling pointer usage
void dangling_pointer_example() {
    Person* person = create_person("Alice", 30);
    Person* alias = person;  // Another pointer to same memory
    
    printf("Person: %s, age %d\n", person->name, person->age);
    
    free_person(person);  // Free the person
    person = NULL;
    
    // PROBLEM: alias still points to freed memory
    printf("Alias: %s, age %d\n", alias->name, alias->age);  // Use-after-free!
}
```

### 4. Buffer Overflows/Underflows

Buffer overflow occurs when you write past the end of an allocated buffer.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BAD: Buffer overflow examples
void buffer_overflow_examples() {
    printf("=== Buffer Overflow Examples ===\n");
    
    // Stack buffer overflow
    char stack_buffer[10];
    // PROBLEM: Writing past buffer end
    strcpy(stack_buffer, "This string is way too long for the buffer!");
    printf("Stack buffer: %s\n", stack_buffer);  // Undefined behavior
    
    // Heap buffer overflow
    char* heap_buffer = malloc(10);
    // PROBLEM: Writing past allocated size
    strcpy(heap_buffer, "Another very long string that exceeds buffer size!");
    printf("Heap buffer: %s\n", heap_buffer);   // Heap corruption
    
    free(heap_buffer);
}

// GOOD: Safe buffer operations
void safe_buffer_operations() {
    printf("=== Safe Buffer Operations ===\n");
    
    const char* source = "This is a test string";
    size_t source_len = strlen(source);
    
    // Allocate sufficient space
    char* safe_buffer = malloc(source_len + 1);  // +1 for null terminator
    if (safe_buffer == NULL) {
        printf("Allocation failed\n");
        return;
    }
    
    // Safe copy operations
    strncpy(safe_buffer, source, source_len);
    safe_buffer[source_len] = '\0';  // Ensure null termination
    
    printf("Safe buffer: %s\n", safe_buffer);
    
    free(safe_buffer);
}

// Buffer underflow example
void buffer_underflow_example() {
    char* buffer = malloc(100);
    if (buffer == NULL) return;
    
    strcpy(buffer, "Hello");
    
    // PROBLEM: Accessing before buffer start
    char* before_buffer = buffer - 1;
    *before_buffer = 'X';  // Buffer underflow!
    
    printf("Buffer: %s\n", buffer);
    
    free(buffer);
}

// Safe string handling
char* safe_string_concat(const char* str1, const char* str2) {
    if (str1 == NULL || str2 == NULL) return NULL;
    
    size_t len1 = strlen(str1);
    size_t len2 = strlen(str2);
    size_t total_len = len1 + len2 + 1;  // +1 for null terminator
    
    char* result = malloc(total_len);
    if (result == NULL) return NULL;
    
    // Safe concatenation
    strcpy(result, str1);
    strcat(result, str2);
    
    return result;  // Caller must free
}
```

### Memory Issue Detection Tools

```c
// Compile-time detection with static analysis
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wformat-security"
#pragma GCC diagnostic error "-Warray-bounds"

void compile_time_detection_demo() {
    int array[5];
    
    // This will generate a warning/error with proper compiler flags
    // array[10] = 42;  // Array bounds violation
    
    for (int i = 0; i < 5; i++) {  // Safe loop
        array[i] = i;
    }
}

#pragma GCC diagnostic pop

// Runtime detection helpers
void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed for %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    printf("Allocated %zu bytes at %p\n", size, ptr);
    return ptr;
}

void safe_free(void** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        printf("Freeing memory at %p\n", *ptr);
        free(*ptr);
        *ptr = NULL;  // Set pointer to NULL
    }
}

// Usage example
void safe_memory_demo() {
    int* array = (int*)safe_malloc(10 * sizeof(int));
    
    for (int i = 0; i < 10; i++) {
        array[i] = i * i;
    }
    
    safe_free((void**)&array);  // Safe free with NULL setting
    
    // array is now NULL, safe to check
    if (array == NULL) {
        printf("Array safely freed and set to NULL\n");
    }
}
```

## RAII and Smart Pointers (C++)

**RAII (Resource Acquisition Is Initialization)** is a C++ programming idiom where resources are automatically managed through object lifetimes. Smart pointers are RAII-compliant classes that automatically manage memory.

### Understanding RAII Principles

RAII follows these principles:
1. **Acquire resources in constructor**
2. **Release resources in destructor**
3. **Resources are automatically managed** when objects go out of scope

```cpp
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

// RAII example with file handling
class FileHandler {
private:
    std::ofstream file;
    std::string filename;
    
public:
    FileHandler(const std::string& fname) : filename(fname) {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        std::cout << "File opened: " << filename << std::endl;
    }
    
    ~FileHandler() {
        if (file.is_open()) {
            file.close();
            std::cout << "File closed: " << filename << std::endl;
        }
    }
    
    void write(const std::string& data) {
        if (file.is_open()) {
            file << data << std::endl;
        }
    }
    
    // Prevent copying (or implement proper copy semantics)
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
};

void raii_file_demo() {
    try {
        FileHandler handler("test.txt");
        handler.write("Hello, RAII!");
        handler.write("Automatic cleanup!");
        
        // File automatically closed when handler goes out of scope
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    // Destructor called here - file is automatically closed
}
```

### Smart Pointers Overview

C++11 introduced smart pointers to automate memory management:

| Smart Pointer | Use Case | Ownership |
|---------------|----------|-----------|
| `std::unique_ptr` | Exclusive ownership | Single owner |
| `std::shared_ptr` | Shared ownership | Multiple owners (reference counted) |
| `std::weak_ptr` | Observer pattern | Non-owning observer |

### std::unique_ptr - Exclusive Ownership

`unique_ptr` provides exclusive ownership of a resource and automatically deletes it when the pointer goes out of scope.

```cpp
#include <memory>
#include <iostream>
#include <vector>

class Resource {
public:
    Resource(int id) : id_(id) {
        std::cout << "Resource " << id_ << " created\n";
    }
    
    ~Resource() {
        std::cout << "Resource " << id_ << " destroyed\n";
    }
    
    void use() {
        std::cout << "Using resource " << id_ << "\n";
    }
    
private:
    int id_;
};

void unique_ptr_basic_usage() {
    std::cout << "=== unique_ptr Basic Usage ===\n";
    
    // Create unique_ptr
    std::unique_ptr<Resource> ptr1 = std::make_unique<Resource>(1);
    ptr1->use();
    
    // Alternative creation (less preferred)
    std::unique_ptr<Resource> ptr2(new Resource(2));
    ptr2->use();
    
    // Check if pointer is valid
    if (ptr1) {
        std::cout << "ptr1 is valid\n";
    }
    
    // Get raw pointer (use carefully)
    Resource* raw_ptr = ptr1.get();
    raw_ptr->use();
    
    // Release ownership (becomes nullptr)
    Resource* released = ptr1.release();
    if (!ptr1) {
        std::cout << "ptr1 is now nullptr\n";
    }
    
    // Must manually delete released pointer
    delete released;
    
    // Reset with new resource
    ptr1.reset(new Resource(3));
    ptr1->use();
    
    // Automatic cleanup when ptr1 and ptr2 go out of scope
}

void unique_ptr_arrays() {
    std::cout << "=== unique_ptr with Arrays ===\n";
    
    // Array version of unique_ptr
    std::unique_ptr<int[]> array_ptr = std::make_unique<int[]>(5);
    
    // Initialize array
    for (int i = 0; i < 5; i++) {
        array_ptr[i] = i * i;
    }
    
    // Print array
    for (int i = 0; i < 5; i++) {
        std::cout << array_ptr[i] << " ";
    }
    std::cout << "\n";
    
    // Automatic cleanup of entire array
}

// Move semantics with unique_ptr
std::unique_ptr<Resource> factory(int id) {
    return std::make_unique<Resource>(id);
}

void unique_ptr_move_semantics() {
    std::cout << "=== unique_ptr Move Semantics ===\n";
    
    std::unique_ptr<Resource> ptr1 = factory(10);
    
    // Move ownership (ptr1 becomes nullptr)
    std::unique_ptr<Resource> ptr2 = std::move(ptr1);
    
    if (!ptr1) {
        std::cout << "ptr1 is nullptr after move\n";
    }
    
    if (ptr2) {
        std::cout << "ptr2 now owns the resource\n";
        ptr2->use();
    }
    
    // Store in container
    std::vector<std::unique_ptr<Resource>> resources;
    resources.push_back(factory(20));
    resources.push_back(factory(21));
    resources.push_back(std::move(ptr2));
    
    std::cout << "Resources in vector: " << resources.size() << "\n";
    for (auto& res : resources) {
        res->use();
    }
    // All resources automatically cleaned up when vector is destroyed
}
```

### std::shared_ptr - Shared Ownership

`shared_ptr` allows multiple pointers to share ownership of the same resource using reference counting.

```cpp
#include <memory>
#include <iostream>
#include <vector>

class SharedResource {
public:
    SharedResource(const std::string& name) : name_(name) {
        std::cout << "SharedResource '" << name_ << "' created\n";
    }
    
    ~SharedResource() {
        std::cout << "SharedResource '" << name_ << "' destroyed\n";
    }
    
    void process() {
        std::cout << "Processing " << name_ << "\n";
    }
    
    const std::string& getName() const { return name_; }
    
private:
    std::string name_;
};

void shared_ptr_basic_usage() {
    std::cout << "=== shared_ptr Basic Usage ===\n";
    
    // Create shared_ptr
    std::shared_ptr<SharedResource> ptr1 = std::make_shared<SharedResource>("Resource1");
    std::cout << "Reference count: " << ptr1.use_count() << "\n";
    
    {
        // Create another shared_ptr pointing to same resource
        std::shared_ptr<SharedResource> ptr2 = ptr1;
        std::cout << "Reference count after copy: " << ptr1.use_count() << "\n";
        
        ptr2->process();
        
        // Create third pointer
        std::shared_ptr<SharedResource> ptr3(ptr1);
        std::cout << "Reference count with 3 pointers: " << ptr1.use_count() << "\n";
        
    } // ptr2 and ptr3 go out of scope, reference count decreases
    
    std::cout << "Reference count after scope exit: " << ptr1.use_count() << "\n";
    
    // Reset pointer
    ptr1.reset();
    std::cout << "ptr1 reset, resource should be destroyed\n";
}

// Sharing resources across objects
class Owner {
private:
    std::shared_ptr<SharedResource> resource_;
    std::string name_;
    
public:
    Owner(const std::string& name, std::shared_ptr<SharedResource> res) 
        : name_(name), resource_(res) {
        std::cout << "Owner '" << name_ << "' created\n";
    }
    
    ~Owner() {
        std::cout << "Owner '" << name_ << "' destroyed\n";
    }
    
    void useResource() {
        if (resource_) {
            std::cout << "Owner '" << name_ << "' using ";
            resource_->process();
        }
    }
    
    int getResourceRefCount() const {
        return resource_ ? resource_.use_count() : 0;
    }
};

void shared_ptr_multiple_owners() {
    std::cout << "=== shared_ptr Multiple Owners ===\n";
    
    auto shared_resource = std::make_shared<SharedResource>("SharedData");
    std::cout << "Initial ref count: " << shared_resource.use_count() << "\n";
    
    {
        Owner owner1("Owner1", shared_resource);
        std::cout << "After Owner1: " << owner1.getResourceRefCount() << "\n";
        
        {
            Owner owner2("Owner2", shared_resource);
            std::cout << "After Owner2: " << owner2.getResourceRefCount() << "\n";
            
            owner1.useResource();
            owner2.useResource();
            
        } // owner2 destroyed
        
        std::cout << "After Owner2 destroyed: " << owner1.getResourceRefCount() << "\n";
        
    } // owner1 destroyed
    
    std::cout << "After Owner1 destroyed: " << shared_resource.use_count() << "\n";
    
    // Resource still alive because shared_resource still holds it
    shared_resource->process();
    
} // shared_resource goes out of scope, resource finally destroyed
```

### std::weak_ptr - Breaking Circular References

`weak_ptr` provides non-owning "weak" reference to an object managed by `shared_ptr`, helping to break circular references.

```cpp
#include <memory>
#include <iostream>

class Node {
public:
    int data;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> parent;  // Use weak_ptr to break circular reference
    
    Node(int value) : data(value) {
        std::cout << "Node " << data << " created\n";
    }
    
    ~Node() {
        std::cout << "Node " << data << " destroyed\n";
    }
    
    void setParent(std::shared_ptr<Node> p) {
        parent = p;  // weak_ptr assignment
    }
    
    void printParent() {
        if (auto p = parent.lock()) {  // Convert weak_ptr to shared_ptr
            std::cout << "Node " << data << " parent is " << p->data << "\n";
        } else {
            std::cout << "Node " << data << " has no parent or parent is destroyed\n";
        }
    }
};

// PROBLEM: Circular reference with shared_ptr
void circular_reference_problem() {
    std::cout << "=== Circular Reference Problem ===\n";
    
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    
    // Create circular reference
    node1->next = node2;
    // If we used shared_ptr for parent:
    // node2->parent = node1;  // This would create circular reference!
    
    // Instead, use weak_ptr
    node2->setParent(node1);
    
    std::cout << "Reference counts - node1: " << node1.use_count() 
              << ", node2: " << node2.use_count() << "\n";
    
    node2->printParent();
    
    // No memory leak - both nodes will be properly destroyed
}

void weak_ptr_usage() {
    std::cout << "=== weak_ptr Usage ===\n";
    
    std::weak_ptr<Node> weak_node;
    
    {
        auto shared_node = std::make_shared<Node>(42);
        weak_node = shared_node;  // weak_ptr points to the resource
        
        std::cout << "shared_ptr use_count: " << shared_node.use_count() << "\n";
        std::cout << "weak_ptr expired: " << weak_node.expired() << "\n";
        
        // Use weak_ptr safely
        if (auto locked = weak_node.lock()) {
            std::cout << "Accessed via weak_ptr: " << locked->data << "\n";
        }
        
    } // shared_node goes out of scope, resource is destroyed
    
    std::cout << "After shared_ptr destroyed:\n";
    std::cout << "weak_ptr expired: " << weak_node.expired() << "\n";
    
    // Trying to access destroyed resource
    if (auto locked = weak_node.lock()) {
        std::cout << "This won't print - resource is destroyed\n";
    } else {
        std::cout << "Cannot access - resource has been destroyed\n";
    }
}
```

### Custom Deleters and Smart Pointers

```cpp
#include <memory>
#include <iostream>
#include <cstdio>

// Custom deleter for FILE*
struct FileDeleter {
    void operator()(FILE* file) {
        if (file) {
            std::cout << "Closing file with custom deleter\n";
            fclose(file);
        }
    }
};

// Custom deleter for arrays allocated with malloc
struct MallocDeleter {
    void operator()(void* ptr) {
        if (ptr) {
            std::cout << "Freeing malloc'd memory\n";
            free(ptr);
        }
    }
};

void custom_deleters_demo() {
    std::cout << "=== Custom Deleters Demo ===\n";
    
    // unique_ptr with custom deleter for FILE*
    std::unique_ptr<FILE, FileDeleter> file_ptr(fopen("test.txt", "w"));
    if (file_ptr) {
        fprintf(file_ptr.get(), "Hello from smart pointer!\n");
        // File automatically closed by custom deleter
    }
    
    // unique_ptr with lambda deleter
    auto lambda_deleter = [](int* ptr) {
        std::cout << "Lambda deleter called\n";
        delete[] ptr;
    };
    
    std::unique_ptr<int[], decltype(lambda_deleter)> array_ptr(
        new int[10], lambda_deleter
    );
    
    for (int i = 0; i < 10; i++) {
        array_ptr[i] = i;
    }
    
    // shared_ptr with custom deleter
    std::shared_ptr<int> malloc_ptr(
        static_cast<int*>(malloc(sizeof(int) * 5)),
        MallocDeleter{}
    );
    
    // Automatic cleanup with custom deleters
}

// RAII wrapper for C resources
template<typename T, T invalid_value, typename Deleter>
class RAIIWrapper {
private:
    T resource_;
    Deleter deleter_;
    
public:
    RAIIWrapper(T resource, Deleter deleter = Deleter{}) 
        : resource_(resource), deleter_(deleter) {}
    
    ~RAIIWrapper() {
        if (resource_ != invalid_value) {
            deleter_(resource_);
        }
    }
    
    T get() const { return resource_; }
    T release() {
        T temp = resource_;
        resource_ = invalid_value;
        return temp;
    }
    
    // Non-copyable
    RAIIWrapper(const RAIIWrapper&) = delete;
    RAIIWrapper& operator=(const RAIIWrapper&) = delete;
    
    // Movable
    RAIIWrapper(RAIIWrapper&& other) noexcept 
        : resource_(other.release()), deleter_(std::move(other.deleter_)) {}
};

void raii_wrapper_demo() {
    std::cout << "=== RAII Wrapper Demo ===\n";
    
    // Wrap FILE* with RAII
    RAIIWrapper<FILE*, nullptr, FileDeleter> file_wrapper(
        fopen("test2.txt", "w")
    );
    
    if (file_wrapper.get()) {
        fprintf(file_wrapper.get(), "RAII wrapper example\n");
    }
    
    // File automatically closed when wrapper is destroyed
}
```

### Best Practices Summary

**✅ DO:**
- Use `std::make_unique` and `std::make_shared` for creating smart pointers
- Prefer `unique_ptr` over `shared_ptr` when possible (better performance)
- Use `weak_ptr` to break circular references
- Use custom deleters for non-standard cleanup
- Follow RAII principles for all resource management

**❌ DON'T:**
- Mix raw pointers with smart pointers for the same resource
- Use `shared_ptr` unnecessarily (overhead of reference counting)
- Forget about circular references with `shared_ptr`
- Use `get()` to extract raw pointer unless absolutely necessary
- Delete smart pointer explicitly (they manage themselves)

## Learning Objectives

By the end of this section, you should be able to:

### Core Concepts
- **Understand memory layout** in C/C++ programs (stack, heap, data segments)
- **Differentiate between stack and heap allocation** and know when to use each
- **Identify and prevent common memory issues** (leaks, double-free, use-after-free, buffer overflows)
- **Apply RAII principles** for automatic resource management in C++

### Practical Skills
- **Write memory-safe C code** using proper allocation/deallocation patterns
- **Use smart pointers effectively** in C++ (`unique_ptr`, `shared_ptr`, `weak_ptr`)
- **Debug memory issues** using tools like Valgrind, AddressSanitizer
- **Implement custom RAII wrappers** for C resources
- **Design memory-efficient data structures** and algorithms

### Self-Assessment Checklist

Before proceeding to advanced memory debugging, ensure you can:

□ Explain the difference between stack and heap memory  
□ Write a program that properly handles dynamic memory allocation  
□ Identify memory leaks in code review  
□ Fix double-free and use-after-free bugs  
□ Use `unique_ptr` and `shared_ptr` appropriately  
□ Understand when to use `weak_ptr` to break circular references  
□ Implement basic RAII patterns  
□ Use memory debugging tools to find issues  

### Practical Exercises

**Exercise 1: Memory Layout Investigation**
```c
// TODO: Complete this program to demonstrate memory layout
#include <stdio.h>
#include <stdlib.h>

int global_var = 100;
static int static_var = 200;
int uninitialized_global;

void investigate_memory_layout() {
    int local_var = 300;
    static int local_static = 400;
    int* heap_var = malloc(sizeof(int));
    *heap_var = 500;
    
    // Print addresses and analyze the layout
    // Your code here
    
    free(heap_var);
}
```

**Exercise 2: Memory Leak Detection and Fix**
```c
// TODO: Find and fix all memory leaks in this code
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Person {
    char* name;
    int age;
    struct Person* next;
} Person;

Person* create_person_list() {
    Person* head = malloc(sizeof(Person));
    head->name = malloc(20);
    strcpy(head->name, "Alice");
    head->age = 25;
    
    Person* second = malloc(sizeof(Person));
    second->name = malloc(20);
    strcpy(second->name, "Bob");
    second->age = 30;
    
    head->next = second;
    second->next = NULL;
    
    return head;  // Memory leaks here - fix them!
}
```

**Exercise 3: Smart Pointer Implementation**
```cpp
// TODO: Complete this basic smart pointer implementation
template<typename T>
class SimpleUniquePtr {
private:
    T* ptr_;
    
public:
    explicit SimpleUniquePtr(T* ptr) : ptr_(ptr) {}
    
    ~SimpleUniquePtr() {
        // Your code here
    }
    
    T& operator*() const {
        // Your code here
    }
    
    T* operator->() const {
        // Your code here
    }
    
    // Disable copy construction and assignment
    // Your code here
};
```

**Exercise 4: RAII File Handler**
```cpp
// TODO: Implement a RAII file handler class
class FileManager {
public:
    FileManager(const std::string& filename, const std::string& mode);
    ~FileManager();
    
    bool isOpen() const;
    void write(const std::string& data);
    std::string read();
    
private:
    // Your implementation here
};
```

## Memory Debugging Tools and Techniques

### Compile-Time Detection

**Compiler Flags for Memory Safety:**
```bash
# GCC/Clang flags for better memory safety
gcc -Wall -Wextra -Werror \
    -fsanitize=address \
    -fsanitize=undefined \
    -fstack-protector-strong \
    -D_FORTIFY_SOURCE=2 \
    -g -O1 program.c

# Static analysis
cppcheck --enable=all program.c
clang-static-analyzer program.c
```

### Runtime Detection Tools

**AddressSanitizer (ASan):**
```bash
# Compile with AddressSanitizer
gcc -fsanitize=address -g -o program program.c

# Run and catch memory errors
./program
# Output will show detailed error information for:
# - Buffer overflows
# - Use-after-free
# - Double-free
# - Memory leaks
```

**Valgrind (Linux/macOS):**
```bash
# Memory leak detection
valgrind --leak-check=full --show-leak-kinds=all ./program

# Detailed memory debugging
valgrind --tool=memcheck --track-origins=yes ./program

# Cache profiling
valgrind --tool=cachegrind ./program
```

**Dr. Memory (Windows):**
```bash
# Windows memory debugging
drmemory.exe -- program.exe
```

### Memory Profiling Example

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Memory usage tracking (simple example)
static size_t total_allocated = 0;
static size_t allocation_count = 0;

void* tracked_malloc(size_t size) {
    void* ptr = malloc(size + sizeof(size_t));
    if (ptr) {
        *(size_t*)ptr = size;
        total_allocated += size;
        allocation_count++;
        printf("Allocated %zu bytes (total: %zu, count: %zu)\n", 
               size, total_allocated, allocation_count);
        return (char*)ptr + sizeof(size_t);
    }
    return NULL;
}

void tracked_free(void* ptr) {
    if (ptr) {
        char* real_ptr = (char*)ptr - sizeof(size_t);
        size_t size = *(size_t*)real_ptr;
        total_allocated -= size;
        allocation_count--;
        printf("Freed %zu bytes (total: %zu, count: %zu)\n", 
               size, total_allocated, allocation_count);
        free(real_ptr);
    }
}

void memory_tracking_demo() {
    printf("=== Memory Tracking Demo ===\n");
    
    int* array1 = tracked_malloc(10 * sizeof(int));
    char* string1 = tracked_malloc(100);
    
    strcpy(string1, "Hello, tracked memory!");
    
    tracked_free(array1);
    tracked_free(string1);
    
    printf("Final memory stats: %zu bytes, %zu allocations\n", 
           total_allocated, allocation_count);
}
```

## Study Materials

### Essential Reading
- **Primary:** "Effective Modern C++" by Scott Meyers - Chapters on smart pointers
- **C Focus:** "Expert C Programming" by Peter van der Linden - Memory management chapters
- **Reference:** "The C++ Programming Language" by Bjarne Stroustrup - RAII and resource management
- **Online:** [CPP Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/) - Resource management section

### Video Resources
- "Memory Management in C" - CS50 Harvard lectures
- "Smart Pointers in C++" - CppCon talks
- "Understanding Valgrind" - Linux debugging tutorials
- "RAII and Modern C++" - Bjarne Stroustrup presentations

### Hands-on Labs
- **Lab 1:** Implement a memory pool allocator
- **Lab 2:** Build a reference-counted smart pointer
- **Lab 3:** Create a garbage collector for C
- **Lab 4:** Performance comparison: manual vs automatic memory management

### Practice Questions

**Conceptual Questions:**
1. What are the advantages and disadvantages of stack vs heap allocation?
2. How does RAII help prevent resource leaks?
3. When should you use `shared_ptr` vs `unique_ptr`?
4. What causes memory fragmentation and how can it be minimized?
5. Explain the difference between shallow and deep copying in the context of memory management.

**Debugging Scenarios:**
6. A program crashes with "double free or corruption" - what could be the causes?
7. Memory usage keeps growing during program execution - how would you investigate?
8. A multi-threaded program has intermittent crashes - what memory issues might be involved?
9. How would you detect and fix a buffer overflow in a string manipulation function?
10. What tools would you use to find memory leaks in a large C++ application?

**Code Analysis:**
```c
// Question: What's wrong with this code?
char* create_buffer(int size) {
    char buffer[size];
    for (int i = 0; i < size; i++) {
        buffer[i] = 'A' + (i % 26);
    }
    return buffer;  // Problem here!
}

// Question: Fix the memory management issues
void process_data() {
    int* data = malloc(100 * sizeof(int));
    
    if (some_condition) {
        return;  // Memory leak!
    }
    
    data = realloc(data, 200 * sizeof(int));  // Potential issue!
    
    for (int i = 0; i < 200; i++) {
        data[i] = i;
    }
    
    free(data);
    data[0] = 42;  // Use-after-free!
}
```

### Development Environment Setup

**Required Tools:**
```bash
# Linux/macOS
sudo apt-get install valgrind
sudo apt-get install cppcheck
pip install cpplint

# Install modern GCC/Clang with sanitizers
sudo apt-get install gcc-10 clang-12

# Windows
# Install Visual Studio with C++ tools
# Download Dr. Memory for Windows debugging
```

**Recommended IDE Extensions:**
- **VS Code:** C/C++ Extension Pack, AddressSanitizer support
- **CLion:** Built-in Valgrind integration, memory profiling
- **Visual Studio:** Diagnostic tools, PVS-Studio integration

**Build Configuration:**
```cmake
# CMake configuration for memory debugging
cmake_minimum_required(VERSION 3.10)
project(MemoryManagement)

set(CMAKE_CXX_STANDARD 17)

# Debug build with sanitizers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=undefined -g")
endif()

# Release build with optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -DNDEBUG")
endif()
```

## Advanced Topics Preview

### Memory Pools and Custom Allocators
```cpp
// Preview: Custom memory pool
class MemoryPool {
public:
    MemoryPool(size_t block_size, size_t block_count);
    ~MemoryPool();
    
    void* allocate();
    void deallocate(void* ptr);
    
private:
    // Implementation details...
};
```

### Lock-Free Memory Management
```cpp
// Preview: Atomic smart pointer for lock-free programming
template<typename T>
class AtomicSharedPtr {
    std::atomic<std::shared_ptr<T>> ptr_;
public:
    std::shared_ptr<T> load() const;
    void store(std::shared_ptr<T> new_ptr);
    bool compare_exchange_weak(std::shared_ptr<T>& expected, 
                              std::shared_ptr<T> desired);
};
```

