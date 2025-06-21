# C++17 Library Additions

## Overview

C++17 introduced significant library enhancements that complement the language features. These additions provide better tools for file system operations, string handling, memory management, and functional programming.

## Key Library Features

### 1. std::filesystem

C++17 introduces a comprehensive filesystem library for portable file and directory operations.

```cpp
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

void demonstrate_basic_filesystem() {
    std::cout << "\n=== Basic Filesystem Operations ===" << std::endl;
    
    // Current path operations
    fs::path current = fs::current_path();
    std::cout << "Current directory: " << current << std::endl;
    
    // Path construction and manipulation
    fs::path file_path = current / "test_files" / "example.txt";
    std::cout << "Constructed path: " << file_path << std::endl;
    std::cout << "Filename: " << file_path.filename() << std::endl;
    std::cout << "Parent path: " << file_path.parent_path() << std::endl;
    std::cout << "Extension: " << file_path.extension() << std::endl;
    std::cout << "Stem: " << file_path.stem() << std::endl;
    
    // Path operations
    fs::path absolute_path = fs::absolute(file_path);
    fs::path canonical_path;
    
    try {
        canonical_path = fs::canonical(file_path);
        std::cout << "Canonical path: " << canonical_path << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cout << "Cannot get canonical path (file may not exist): " << e.what() << std::endl;
    }
    
    // Check path properties
    std::cout << "Path properties:" << std::endl;
    std::cout << "  Absolute: " << file_path.is_absolute() << std::endl;
    std::cout << "  Relative: " << file_path.is_relative() << std::endl;
    std::cout << "  Has filename: " << file_path.has_filename() << std::endl;
    std::cout << "  Has extension: " << file_path.has_extension() << std::endl;
}

void demonstrate_directory_operations() {
    std::cout << "\n=== Directory Operations ===" << std::endl;
    
    fs::path test_dir = "test_directory";
    
    try {
        // Create directory
        if (fs::create_directory(test_dir)) {
            std::cout << "Created directory: " << test_dir << std::endl;
        } else {
            std::cout << "Directory already exists: " << test_dir << std::endl;
        }
        
        // Create nested directories
        fs::path nested_dir = test_dir / "nested" / "deep";
        fs::create_directories(nested_dir);
        std::cout << "Created nested directories: " << nested_dir << std::endl;
        
        // Create some test files
        std::vector<std::string> test_files = {
            "file1.txt", "file2.cpp", "file3.h", "document.pdf"
        };
        
        for (const auto& filename : test_files) {
            fs::path file_path = test_dir / filename;
            std::ofstream file(file_path);
            file << "Test content for " << filename << std::endl;
            file.close();
        }
        
        // Create files in nested directory
        std::ofstream nested_file(nested_dir / "deep_file.txt");
        nested_file << "Content in deep directory" << std::endl;
        nested_file.close();
        
        std::cout << "Created test files" << std::endl;
        
    } catch (const fs::filesystem_error& e) {
        std::cout << "Directory operation error: " << e.what() << std::endl;
    }
}

void demonstrate_directory_iteration() {
    std::cout << "\n=== Directory Iteration ===" << std::endl;
    
    fs::path test_dir = "test_directory";
    
    if (!fs::exists(test_dir)) {
        std::cout << "Test directory doesn't exist. Run directory operations first." << std::endl;
        return;
    }
    
    // Simple directory iteration
    std::cout << "Contents of " << test_dir << ":" << std::endl;
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        std::cout << "  " << entry.path().filename();
        if (entry.is_directory()) {
            std::cout << " [DIR]";
        } else if (entry.is_regular_file()) {
            std::cout << " [FILE, " << entry.file_size() << " bytes]";
        }
        std::cout << std::endl;
    }
    
    // Recursive directory iteration
    std::cout << "\nRecursive contents:" << std::endl;
    for (const auto& entry : fs::recursive_directory_iterator(test_dir)) {
        std::string indent(entry.depth() * 2, ' ');
        std::cout << indent << entry.path().filename();
        
        if (entry.is_directory()) {
            std::cout << " [DIR]";
        } else if (entry.is_regular_file()) {
            std::cout << " [FILE, " << entry.file_size() << " bytes]";
        }
        std::cout << std::endl;
    }
    
    // Filtered iteration
    std::cout << "\nC++ files only:" << std::endl;
    for (const auto& entry : fs::recursive_directory_iterator(test_dir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension();
            if (ext == ".cpp" || ext == ".h" || ext == ".hpp") {
                std::cout << "  " << entry.path() << std::endl;
            }
        }
    }
}

void demonstrate_file_status_operations() {
    std::cout << "\n=== File Status Operations ===" << std::endl;
    
    fs::path test_file = "test_directory/file1.txt";
    
    if (!fs::exists(test_file)) {
        std::cout << "Test file doesn't exist." << std::endl;
        return;
    }
    
    // File existence and type checking
    std::cout << "File: " << test_file << std::endl;
    std::cout << "  Exists: " << fs::exists(test_file) << std::endl;
    std::cout << "  Is regular file: " << fs::is_regular_file(test_file) << std::endl;
    std::cout << "  Is directory: " << fs::is_directory(test_file) << std::endl;
    std::cout << "  Is symlink: " << fs::is_symlink(test_file) << std::endl;
    
    // File size
    try {
        auto size = fs::file_size(test_file);
        std::cout << "  Size: " << size << " bytes" << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cout << "  Size: Error - " << e.what() << std::endl;
    }
    
    // File times
    try {
        auto last_write = fs::last_write_time(test_file);
        auto time_t = std::chrono::system_clock::to_time_t(
            std::chrono::file_clock::to_sys(last_write));
        std::cout << "  Last modified: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cout << "  Last modified: Error - " << e.what() << std::endl;
    }
    
    // File permissions (platform dependent)
    try {
        auto perms = fs::status(test_file).permissions();
        std::cout << "  Permissions: ";
        
        // Check common permissions
        if ((perms & fs::perms::owner_read) != fs::perms::none) std::cout << "r";
        else std::cout << "-";
        if ((perms & fs::perms::owner_write) != fs::perms::none) std::cout << "w";
        else std::cout << "-";
        if ((perms & fs::perms::owner_exec) != fs::perms::none) std::cout << "x";
        else std::cout << "-";
        
        std::cout << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cout << "  Permissions: Error - " << e.what() << std::endl;
    }
}

void demonstrate_file_operations() {
    std::cout << "\n=== File Operations ===" << std::endl;
    
    fs::path source = "test_directory/file1.txt";
    fs::path copy_dest = "test_directory/file1_copy.txt";
    fs::path move_dest = "test_directory/file1_moved.txt";
    
    if (!fs::exists(source)) {
        std::cout << "Source file doesn't exist." << std::endl;
        return;
    }
    
    try {
        // Copy file
        fs::copy_file(source, copy_dest, fs::copy_options::overwrite_existing);
        std::cout << "Copied " << source << " to " << copy_dest << std::endl;
        
        // Rename/move file
        fs::rename(copy_dest, move_dest);
        std::cout << "Moved " << copy_dest << " to " << move_dest << std::endl;
        
        // Copy directory
        fs::path backup_dir = "test_directory_backup";
        fs::copy("test_directory", backup_dir, 
                fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        std::cout << "Created backup directory: " << backup_dir << std::endl;
        
        // Space information
        auto space = fs::space(".");
        std::cout << "Disk space information:" << std::endl;
        std::cout << "  Total: " << space.capacity / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Free: " << space.free / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Available: " << space.available / (1024 * 1024) << " MB" << std::endl;
        
    } catch (const fs::filesystem_error& e) {
        std::cout << "File operation error: " << e.what() << std::endl;
    }
}

// Utility function to find files with specific criteria
std::vector<fs::path> find_files(const fs::path& root, 
                                const std::string& pattern,
                                bool recursive = true) {
    std::vector<fs::path> results;
    
    try {
        auto iterator = recursive ? 
            fs::directory_iterator(fs::recursive_directory_iterator(root)) :
            fs::directory_iterator(fs::directory_iterator(root));
        
        // Note: This is a simplified approach. For real pattern matching,
        // you'd want to use regex or glob-like matching
        for (const auto& entry : (recursive ? 
                                 fs::recursive_directory_iterator(root) :
                                 fs::directory_iterator(root))) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find(pattern) != std::string::npos) {
                    results.push_back(entry.path());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cout << "Search error: " << e.what() << std::endl;
    }
    
    return results;
}

void demonstrate_filesystem_utilities() {
    std::cout << "\n=== Filesystem Utilities ===" << std::endl;
    
    // Find all .txt files
    auto txt_files = find_files(".", ".txt", true);
    std::cout << "Found " << txt_files.size() << " .txt files:" << std::endl;
    for (const auto& file : txt_files) {
        std::cout << "  " << file << std::endl;
    }
    
    // Calculate directory size
    auto calculate_directory_size = [](const fs::path& dir) -> uintmax_t {
        uintmax_t size = 0;
        try {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    size += fs::file_size(entry);
                }
            }
        } catch (const fs::filesystem_error&) {
            // Handle errors gracefully
        }
        return size;
    };
    
    if (fs::exists("test_directory")) {
        auto dir_size = calculate_directory_size("test_directory");
        std::cout << "Test directory size: " << dir_size << " bytes" << std::endl;
    }
}
```

### 2. std::string_view

A non-owning reference to a string, providing efficient string operations without copying.

```cpp
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <algorithm>

void demonstrate_basic_string_view() {
    std::cout << "\n=== Basic string_view Operations ===" << std::endl;
    
    // Creating string_view from different sources
    std::string str = "Hello, World!";
    const char* cstr = "C-style string";
    char buffer[] = {'B', 'u', 'f', 'f', 'e', 'r', '\0'};
    
    std::string_view sv1 = str;           // From std::string
    std::string_view sv2 = cstr;          // From C-string
    std::string_view sv3 = buffer;        // From char array
    std::string_view sv4 = "String literal"; // From literal
    
    std::cout << "string_view from string: " << sv1 << std::endl;
    std::cout << "string_view from C-string: " << sv2 << std::endl;
    std::cout <- "string_view from buffer: " << sv3 << std::endl;
    std::cout << "string_view from literal: " << sv4 << std::endl;
    
    // Substring operations (no copying!)
    std::string_view substr = sv1.substr(7, 5);  // "World"
    std::cout << "Substring: " << substr << std::endl;
    
    // Basic properties
    std::cout << "Length: " << sv1.length() << std::endl;
    std::cout << "Size: " << sv1.size() << std::endl;
    std::cout << "Empty: " << sv1.empty() << std::endl;
    std::cout << "First char: " << sv1.front() << std::endl;
    std::cout << "Last char: " << sv1.back() << std::endl;
}

// Function accepting string_view (can work with any string type)
void process_string(std::string_view sv) {
    std::cout << "Processing: '" << sv << "' (length: " << sv.length() << ")" << std::endl;
    
    // Find operations
    auto pos = sv.find("o");
    if (pos != std::string_view::npos) {
        std::cout << "  Found 'o' at position: " << pos << std::endl;
    }
    
    // Check prefix/suffix
    if (sv.starts_with("Hello")) {
        std::cout << "  Starts with 'Hello'" << std::endl;
    }
    
    if (sv.ends_with("!")) {
        std::cout << "  Ends with '!'" << std::endl;
    }
}

void demonstrate_string_view_functions() {
    std::cout << "\n=== string_view Function Parameters ===" << std::endl;
    
    // All these calls work without string conversion/copying
    process_string("String literal");
    process_string(std::string("std::string object"));
    
    const char* cstring = "C-style string";
    process_string(cstring);
    
    char buffer[] = "Character array";
    process_string(buffer);
    
    // Partial strings
    std::string long_string = "This is a very long string for demonstration";
    process_string(std::string_view(long_string).substr(10, 8)); // "very lon"
}

// Performance comparison function
void performance_comparison() {
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    
    const std::string large_string(100000, 'A');
    const int iterations = 100000;
    
    // Function that copies string
    auto process_by_copy = [](const std::string& s) {
        return s.substr(1000, 100).length();
    };
    
    // Function that uses string_view
    auto process_by_view = [](std::string_view sv) {
        return sv.substr(1000, 100).length();
    };
    
    // Measure copy performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        process_by_copy(large_string);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Measure string_view performance
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        process_by_view(large_string);
    }
    end = std::chrono::high_resolution_clock::now();
    auto view_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Copy approach: " << copy_time.count() << " microseconds" << std::endl;
    std::cout << "string_view approach: " << view_time.count() << " microseconds" << std::endl;
    std::cout << "Speedup: " << (double)copy_time.count() / view_time.count() << "x" << std::endl;
}

// Advanced string_view operations
void demonstrate_advanced_string_view() {
    std::cout << "\n=== Advanced string_view Operations ===" << std::endl;
    
    std::string_view text = "  Hello, World! Welcome to C++17!  ";
    
    // Remove prefix/suffix whitespace (manual implementation)
    auto trim_left = [](std::string_view sv) {
        sv.remove_prefix(std::min(sv.find_first_not_of(" \t\n\r"), sv.size()));
        return sv;
    };
    
    auto trim_right = [](std::string_view sv) {
        sv.remove_suffix(std::min(sv.size() - sv.find_last_not_of(" \t\n\r") - 1, sv.size()));
        return sv;
    };
    
    auto trim = [&](std::string_view sv) {
        return trim_right(trim_left(sv));
    };
    
    std::cout << "Original: '" << text << "'" << std::endl;
    std::cout << "Trimmed: '" << trim(text) << "'" << std::endl;
    
    // Tokenization with string_view
    auto tokenize = [](std::string_view sv, char delimiter) {
        std::vector<std::string_view> tokens;
        size_t start = 0;
        size_t end = 0;
        
        while ((end = sv.find(delimiter, start)) != std::string_view::npos) {
            if (end != start) {  // Skip empty tokens
                tokens.push_back(sv.substr(start, end - start));
            }
            start = end + 1;
        }
        
        if (start < sv.length()) {
            tokens.push_back(sv.substr(start));
        }
        
        return tokens;
    };
    
    std::string csv_data = "apple,banana,cherry,date,elderberry";
    auto tokens = tokenize(csv_data, ',');
    
    std::cout << "Tokenized CSV: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "'" << tokens[i] << "'";
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

// Safe string_view usage patterns
class StringProcessor {
private:
    std::string_view current_text;  // Dangerous: storing string_view as member
    
public:
    // Dangerous: string_view might outlive the temporary string
    void set_text_dangerous(const std::string& text) {
        current_text = text;  // text might be temporary
    }
    
    // Safe: process immediately
    void process_text_safe(std::string_view text) {
        // Process immediately, don't store
        std::cout << "Processing safely: " << text << std::endl;
    }
    
    // Safe: copy if you need to store
    void set_text_safe(std::string_view text) {
        stored_text = std::string(text);  // Make a copy
        current_text = stored_text;       // Now safe to store view
    }
    
private:
    std::string stored_text;  // Owned storage
};

void demonstrate_string_view_safety() {
    std::cout << "\n=== string_view Safety ===" << std::endl;
    
    StringProcessor processor;
    
    // Safe usage
    processor.process_text_safe("Safe immediate processing");
    processor.set_text_safe("Safely stored text");
    
    // Demonstrate potential danger (commented out to avoid undefined behavior)
    /*
    {
        std::string temp = "Temporary string";
        processor.set_text_dangerous(temp);
    }  // temp goes out of scope here
    // Using processor.current_text now would be undefined behavior
    */
    
    // Safe patterns
    std::string permanent_string = "Permanent string";
    processor.set_text_dangerous(permanent_string);  // Safe: permanent_string stays alive
    
    std::cout << "String safety demonstrated" << std::endl;
}
```

### 3. std::optional

Represents a value that may or may not be present, eliminating the need for sentinel values.

```cpp
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <map>

// Function that might not return a value
std::optional<int> safe_divide(int numerator, int denominator) {
    if (denominator == 0) {
        return std::nullopt;  // No value
    }
    return numerator / denominator;
}

// Function that might not find a value
std::optional<std::string> find_name_by_id(int id) {
    static std::map<int, std::string> database = {
        {1, "Alice"}, {2, "Bob"}, {3, "Charlie"}
    };
    
    auto it = database.find(id);
    if (it != database.end()) {
        return it->second;
    }
    return std::nullopt;
}

void demonstrate_basic_optional() {
    std::cout << "\n=== Basic std::optional Usage ===" << std::endl;
    
    // Creating optional values
    std::optional<int> opt1 = 42;           // Has value
    std::optional<int> opt2;                // No value (nullopt)
    std::optional<int> opt3 = std::nullopt; // Explicitly no value
    
    // Checking if optional has value
    std::cout << "opt1 has value: " << opt1.has_value() << std::endl;
    std::cout << "opt2 has value: " << opt2.has_value() << std::endl;
    std::cout << "opt3 has value: " << opt3.has_value() << std::endl;
    
    // Accessing values
    if (opt1) {  // Implicit bool conversion
        std::cout << "opt1 value: " << *opt1 << std::endl;        // Dereference
        std::cout << "opt1 value: " << opt1.value() << std::endl; // .value() method
    }
    
    // Safe access with default value
    std::cout << "opt2 value or default: " << opt2.value_or(-1) << std::endl;
    
    // Using functions that return optional
    auto result1 = safe_divide(10, 2);
    auto result2 = safe_divide(10, 0);
    
    std::cout << "10 / 2 = " << (result1 ? std::to_string(*result1) : "undefined") << std::endl;
    std::cout << "10 / 0 = " << (result2 ? std::to_string(*result2) : "undefined") << std::endl;
    
    // Name lookup examples
    for (int id : {1, 2, 5}) {
        auto name = find_name_by_id(id);
        std::cout << "ID " << id << ": " << name.value_or("Not found") << std::endl;
    }
}

// Using optional in more complex scenarios
class Person {
private:
    std::string name;
    std::optional<int> age;
    std::optional<std::string> email;
    
public:
    Person(const std::string& n) : name(n) {}
    
    void set_age(int a) { age = a; }
    void set_email(const std::string& e) { email = e; }
    
    const std::string& get_name() const { return name; }
    std::optional<int> get_age() const { return age; }
    std::optional<std::string> get_email() const { return email; }
    
    void print() const {
        std::cout << "Person: " << name;
        if (age) {
            std::cout << ", Age: " << *age;
        }
        if (email) {
            std::cout << ", Email: " << *email;
        }
        std::cout << std::endl;
    }
};

void demonstrate_optional_in_classes() {
    std::cout << "\n=== std::optional in Classes ===" << std::endl;
    
    Person person1("Alice");
    person1.set_age(30);
    person1.set_email("alice@example.com");
    
    Person person2("Bob");
    person2.set_age(25);
    // No email set
    
    Person person3("Charlie");
    // No age or email set
    
    person1.print();
    person2.print();
    person3.print();
}

// Optional chaining and transformations
template<typename T, typename F>
auto transform_optional(const std::optional<T>& opt, F func) -> std::optional<decltype(func(*opt))> {
    if (opt) {
        return func(*opt);
    }
    return std::nullopt;
}

std::optional<std::string> get_user_name(int user_id) {
    static std::map<int, std::string> users = {
        {1, "admin"}, {2, "user"}, {3, "guest"}
    };
    
    auto it = users.find(user_id);
    return (it != users.end()) ? std::optional<std::string>(it->second) : std::nullopt;
}

std::optional<std::string> get_user_email(const std::string& username) {
    static std::map<std::string, std::string> emails = {
        {"admin", "admin@company.com"},
        {"user", "user@company.com"}
        // "guest" has no email
    };
    
    auto it = emails.find(username);
    return (it != emails.end()) ? std::optional<std::string>(it->second) : std::nullopt;
}

void demonstrate_optional_chaining() {
    std::cout << "\n=== Optional Chaining ===" << std::endl;
    
    // Manual chaining
    for (int user_id : {1, 2, 3, 4}) {
        std::cout << "User ID " << user_id << ": ";
        
        auto username = get_user_name(user_id);
        if (username) {
            auto email = get_user_email(*username);
            if (email) {
                std::cout << *email;
            } else {
                std::cout << "No email for " << *username;
            }
        } else {
            std::cout << "User not found";
        }
        std::cout << std::endl;
    }
    
    // Using transform helper
    std::cout << "\nUsing transform helper:" << std::endl;
    for (int user_id : {1, 2, 3, 4}) {
        auto result = transform_optional(get_user_name(user_id), 
                                       [](const std::string& name) {
                                           return get_user_email(name);
                                       });
        
        // Flatten the optional<optional<string>> to optional<string>
        std::optional<std::string> email;
        if (result && *result) {
            email = **result;
        }
        
        std::cout << "User ID " << user_id << ": " << email.value_or("No email") << std::endl;
    }
}

// Error handling with optional
class Calculator {
public:
    static std::optional<double> sqrt(double x) {
        if (x < 0) {
            return std::nullopt;  // Can't take square root of negative number
        }
        return std::sqrt(x);
    }
    
    static std::optional<double> log(double x) {
        if (x <= 0) {
            return std::nullopt;  // Logarithm undefined for non-positive numbers
        }
        return std::log(x);
    }
    
    static std::optional<double> divide(double a, double b) {
        if (b == 0.0) {
            return std::nullopt;  // Division by zero
        }
        return a / b;
    }
};

void demonstrate_optional_error_handling() {
    std::cout << "\n=== Error Handling with Optional ===" << std::endl;
    
    std::vector<double> test_values = {4.0, -1.0, 0.0, 16.0};
    
    for (double value : test_values) {
        std::cout << "Value: " << value << std::endl;
        
        auto sqrt_result = Calculator::sqrt(value);
        std::cout << "  sqrt: " << (sqrt_result ? std::to_string(*sqrt_result) : "undefined") << std::endl;
        
        auto log_result = Calculator::log(value);
        std::cout << "  log: " << (log_result ? std::to_string(*log_result) : "undefined") << std::endl;
        
        auto div_result = Calculator::divide(10.0, value);
        std::cout << "  10/x: " << (div_result ? std::to_string(*div_result) : "undefined") << std::endl;
        
        std::cout << std::endl;
    }
}

// Optional comparison and operations
void demonstrate_optional_operations() {
    std::cout << "\n=== Optional Operations ===" << std::endl;
    
    std::optional<int> a = 5;
    std::optional<int> b = 10;
    std::optional<int> c;  // nullopt
    
    // Comparison with values
    std::cout << "a == 5: " << (a == 5) << std::endl;
    std::cout << "c == nullopt: " << (c == std::nullopt) << std::endl;
    
    // Comparison between optionals
    std::cout << "a < b: " << (a < b) << std::endl;
    std::cout << "a < c: " << (a < c) << std::endl;  // false (value vs nullopt)
    std::cout << "c < a: " << (c < a) << std::endl;  // true (nullopt vs value)
    
    // Reset and assignment
    a.reset();  // Make it nullopt
    std::cout << "After reset, a has value: " << a.has_value() << std::endl;
    
    a = 42;     // Assign new value
    std::cout << "After assignment, a value: " << *a << std::endl;
    
    // Emplace
    b.emplace(100);  // Construct in place
    std::cout << "After emplace, b value: " << *b << std::endl;
}

// Practical example: Configuration system
class Configuration {
private:
    std::optional<std::string> database_url;
    std::optional<int> port;
    std::optional<bool> debug_mode;
    std::optional<int> max_connections;
    
public:
    void load_from_file(const std::string& /* filename */) {
        // Simulate loading configuration
        database_url = "postgresql://localhost:5432/mydb";
        port = 8080;
        // debug_mode and max_connections not set in config file
    }
    
    std::string get_database_url() const {
        return database_url.value_or("sqlite://default.db");
    }
    
    int get_port() const {
        return port.value_or(3000);  // Default port
    }
    
    bool is_debug_mode() const {
        return debug_mode.value_or(false);  // Default to false
    }
    
    int get_max_connections() const {
        return max_connections.value_or(100);  // Default max connections
    }
    
    void print_config() const {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Database URL: " << get_database_url() << std::endl;
        std::cout << "  Port: " << get_port() << std::endl;
        std::cout << "  Debug Mode: " << std::boolalpha << is_debug_mode() << std::endl;
        std::cout << "  Max Connections: " << get_max_connections() << std::endl;
    }
};

void demonstrate_practical_optional() {
    std::cout << "\n=== Practical Optional Example ===" << std::endl;
    
    Configuration config;
    config.load_from_file("config.ini");
    config.print_config();
}
```

### 4. std::variant

A type-safe union that can hold one of several alternative types.

```cpp
#include <iostream>
#include <variant>
#include <string>
#include <vector>
#include <type_traits>

// Basic variant usage
void demonstrate_basic_variant() {
    std::cout << "\n=== Basic std::variant Usage ===" << std::endl;
    
    // Variant that can hold int, double, or string
    std::variant<int, double, std::string> var;
    
    // Default construction (first type)
    std::cout << "Default variant index: " << var.index() << std::endl;  // 0 (int)
    std::cout << "Default variant value: " << std::get<int>(var) << std::endl;  // 0
    
    // Assignment
    var = 42;
    std::cout << "After int assignment - index: " << var.index() << ", value: " << std::get<int>(var) << std::endl;
    
    var = 3.14;
    std::cout << "After double assignment - index: " << var.index() << ", value: " << std::get<double>(var) << std::endl;
    
    var = std::string("Hello");
    std::cout << "After string assignment - index: " << var.index() << ", value: " << std::get<std::string>(var) << std::endl;
    
    // Check which type is currently held
    if (std::holds_alternative<std::string>(var)) {
        std::cout << "Variant currently holds a string" << std::endl;
    }
}

// Safe access to variant values
void demonstrate_variant_access() {
    std::cout << "\n=== Variant Access Methods ===" << std::endl;
    
    std::variant<int, double, std::string> var = 42;
    
    // std::get with type (throws if wrong type)
    try {
        int value = std::get<int>(var);
        std::cout << "Got int value: " << value << std::endl;
    } catch (const std::bad_variant_access& e) {
        std::cout << "Bad variant access: " << e.what() << std::endl;
    }
    
    // std::get with index
    try {
        int value = std::get<0>(var);  // First type (int)
        std::cout << "Got value by index: " << value << std::endl;
    } catch (const std::bad_variant_access& e) {
        std::cout << "Bad variant access: " << e.what() << std::endl;
    }
    
    // std::get_if (returns pointer, nullptr if wrong type)
    if (auto ptr = std::get_if<int>(&var)) {
        std::cout << "Got int via get_if: " << *ptr << std::endl;
    }
    
    if (auto ptr = std::get_if<double>(&var)) {
        std::cout << "Got double via get_if: " << *ptr << std::endl;
    } else {
        std::cout << "Variant doesn't hold double" << std::endl;
    }
    
    // Demonstrate exception when accessing wrong type
    var = std::string("Hello");
    try {
        int value = std::get<int>(var);  // This will throw
        std::cout << "This shouldn't print" << std::endl;
    } catch (const std::bad_variant_access& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
}

// Using std::visit to handle all possible types
void demonstrate_variant_visit() {
    std::cout << "\n=== Variant Visitation ===" << std::endl;
    
    std::variant<int, double, std::string> var;
    
    // Generic visitor using operator()
    struct Visitor {
        void operator()(int value) const {
            std::cout << "Visiting int: " << value << std::endl;
        }
        
        void operator()(double value) const {
            std::cout << "Visiting double: " << value << std::endl;
        }
        
        void operator()(const std::string& value) const {
            std::cout << "Visiting string: " << value << std::endl;
        }
    };
    
    // Test with different values
    std::vector<std::variant<int, double, std::string>> variants = {
        42, 3.14, std::string("Hello World")
    };
    
    for (const auto& v : variants) {
        std::visit(Visitor{}, v);
    }
    
    // Lambda visitor
    auto lambda_visitor = [](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int>) {
            std::cout << "Lambda visiting int: " << value << std::endl;
        } else if constexpr (std::is_same_v<T, double>) {
            std::cout << "Lambda visiting double: " << value << std::endl;
        } else if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "Lambda visiting string: " << value << std::endl;
        }
    };
    
    std::cout << "\nUsing lambda visitor:" << std::endl;
    for (const auto& v : variants) {
        std::visit(lambda_visitor, v);
    }
}

// Practical example: Expression evaluator
struct Number {
    double value;
    Number(double v) : value(v) {}
};

struct BinaryOp {
    char op;
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;
    
    BinaryOp(char o, std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : op(o), left(std::move(l)), right(std::move(r)) {}
};

struct UnaryOp {
    char op;
    std::unique_ptr<Expression> operand;
    
    UnaryOp(char o, std::unique_ptr<Expression> expr)
        : op(o), operand(std::move(expr)) {}
};

// Forward declaration for recursive variant
struct Expression;

// Define the variant after forward declarations
using ExpressionVariant = std::variant<Number, BinaryOp, UnaryOp>;

struct Expression {
    ExpressionVariant expr;
    
    template<typename T>
    Expression(T&& t) : expr(std::forward<T>(t)) {}
};

// Evaluator visitor
struct Evaluator {
    double operator()(const Number& num) const {
        return num.value;
    }
    
    double operator()(const BinaryOp& binop) const {
        double left_val = std::visit(*this, binop.left->expr);
        double right_val = std::visit(*this, binop.right->expr);
        
        switch (binop.op) {
            case '+': return left_val + right_val;
            case '-': return left_val - right_val;
            case '*': return left_val * right_val;
            case '/': 
                if (right_val == 0) throw std::runtime_error("Division by zero");
                return left_val / right_val;
            default: throw std::runtime_error("Unknown binary operator");
        }
    }
    
    double operator()(const UnaryOp& unop) const {
        double val = std::visit(*this, unop.operand->expr);
        
        switch (unop.op) {
            case '-': return -val;
            case '+': return val;
            default: throw std::runtime_error("Unknown unary operator");
        }
    }
};

// Helper functions to create expressions
std::unique_ptr<Expression> make_number(double value) {
    return std::make_unique<Expression>(Number(value));
}

std::unique_ptr<Expression> make_binary_op(char op, 
                                          std::unique_ptr<Expression> left,
                                          std::unique_ptr<Expression> right) {
    return std::make_unique<Expression>(BinaryOp(op, std::move(left), std::move(right)));
}

std::unique_ptr<Expression> make_unary_op(char op, std::unique_ptr<Expression> operand) {
    return std::make_unique<Expression>(UnaryOp(op, std::move(operand)));
}

void demonstrate_expression_evaluator() {
    std::cout << "\n=== Expression Evaluator with Variant ===" << std::endl;
    
    // Build expression: (2 + 3) * (-4)
    auto expr = make_binary_op('*',
        make_binary_op('+', make_number(2), make_number(3)),
        make_unary_op('-', make_number(4))
    );
    
    try {
        double result = std::visit(Evaluator{}, expr->expr);
        std::cout << "Expression result: " << result << std::endl;  // Should be -20
    } catch (const std::exception& e) {
        std::cout << "Evaluation error: " << e.what() << std::endl;
    }
    
    // Build expression: 10 / (3 - 3) to test division by zero
    auto expr2 = make_binary_op('/',
        make_number(10),
        make_binary_op('-', make_number(3), make_number(3))
    );
    
    try {
        double result = std::visit(Evaluator{}, expr2->expr);
        std::cout << "Expression 2 result: " << result << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Evaluation error: " << e.what() << std::endl;
    }
}

// Event system using variant
struct MouseClick {
    int x, y;
    int button;
};

struct KeyPress {
    char key;
    bool shift_pressed;
};

struct WindowResize {
    int width, height;
};

using Event = std::variant<MouseClick, KeyPress, WindowResize>;

class EventHandler {
public:
    void handle_event(const Event& event) {
        std::visit([this](const auto& e) { this->process(e); }, event);
    }
    
private:
    void process(const MouseClick& click) {
        std::cout << "Mouse clicked at (" << click.x << ", " << click.y 
                  << ") with button " << click.button << std::endl;
    }
    
    void process(const KeyPress& key) {
        std::cout << "Key '" << key.key << "' pressed";
        if (key.shift_pressed) std::cout << " (with Shift)";
        std::cout << std::endl;
    }
    
    void process(const WindowResize& resize) {
        std::cout << "Window resized to " << resize.width << "x" << resize.height << std::endl;
    }
};

void demonstrate_event_system() {
    std::cout << "\n=== Event System with Variant ===" << std::endl;
    
    EventHandler handler;
    
    std::vector<Event> events = {
        MouseClick{100, 200, 1},
        KeyPress{'A', true},
        WindowResize{800, 600},
        MouseClick{150, 250, 2},
        KeyPress{'x', false}
    };
    
    for (const auto& event : events) {
        handler.handle_event(event);
    }
}

int main() {
    demonstrate_basic_filesystem();
    demonstrate_directory_operations();
    demonstrate_directory_iteration();
    demonstrate_file_status_operations();
    demonstrate_file_operations();
    demonstrate_filesystem_utilities();
    
    demonstrate_basic_string_view();
    demonstrate_string_view_functions();
    performance_comparison();
    demonstrate_advanced_string_view();
    demonstrate_string_view_safety();
    
    demonstrate_basic_optional();
    demonstrate_optional_in_classes();
    demonstrate_optional_chaining();
    demonstrate_optional_error_handling();
    demonstrate_optional_operations();
    demonstrate_practical_optional();
    
    demonstrate_basic_variant();
    demonstrate_variant_access();
    demonstrate_variant_visit();
    demonstrate_expression_evaluator();
    demonstrate_event_system();
    
    return 0;
}
```

## Summary

C++17 library additions provide powerful tools for modern C++ development:

- **std::filesystem**: Comprehensive file system operations with cross-platform support
- **std::string_view**: Non-owning string references for efficient string operations
- **std::optional**: Type-safe nullable values eliminating sentinel value patterns
- **std::variant**: Type-safe unions for sum types and visitor patterns

Key benefits:
- Better resource management and performance
- Type safety and expressiveness
- Cross-platform compatibility
- Modern alternatives to C-style patterns
- Support for functional programming concepts

These library features complement the language enhancements in C++17, providing a more complete and modern C++ programming experience.
