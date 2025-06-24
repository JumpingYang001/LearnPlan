# File System Operations

*Duration: 1 week*

## Overview

The Boost.Filesystem library provides a comprehensive, portable solution for file system operations in C++. Before the standardization of `std::filesystem` in C++17, Boost.Filesystem was the de facto standard for cross-platform file system manipulation. Even today, it remains relevant for projects requiring C++11/14 compatibility or additional features not available in the standard library.

### Why Boost.Filesystem?

**Historical Context:**
- Developed as a response to the lack of standardized file system operations in C++
- Influenced the design of C++17's `std::filesystem`
- Provides consistent behavior across Windows, Linux, macOS, and other platforms

**Key Advantages:**
- **Cross-platform compatibility**: Write once, run anywhere
- **Type safety**: Strong typing prevents common path manipulation errors
- **Exception safety**: RAII and proper error handling
- **Unicode support**: Handles international filenames correctly
- **Performance**: Optimized for common operations

**Real-world Applications:**
- File managers and organizers
- Backup and synchronization tools
- Build systems and deployment scripts
- Log analyzers and monitoring tools
- Configuration management systems

### Architecture Overview

```
Boost.Filesystem Architecture
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Path Operations  │  Directory Ops  │  File Status/Perms   │
├─────────────────────────────────────────────────────────────┤
│           Boost.Filesystem Core Library                     │
├─────────────────────────────────────────────────────────────┤
│  Platform Abstraction Layer (Windows/POSIX)                │
├─────────────────────────────────────────────────────────────┤
│  Operating System APIs                                      │
│  Windows: Win32 API    │    Unix: POSIX syscalls           │
└─────────────────────────────────────────────────────────────┘
```

### Prerequisites

Before diving into Boost.Filesystem, ensure you have:
- Basic C++ knowledge (classes, exceptions, iterators)
- Understanding of file system concepts (directories, files, permissions)
- Familiarity with RAII and exception handling
- Basic knowledge of Unicode and character encodings

### Installation and Setup

**Installing Boost:**
```bash
# Ubuntu/Debian
sudo apt-get install libboost-filesystem-dev

# CentOS/RHEL
sudo yum install boost-filesystem-devel

# macOS with Homebrew
brew install boost

# Windows with vcpkg
vcpkg install boost-filesystem
```

**CMake Configuration:**
```cmake
find_package(Boost REQUIRED COMPONENTS filesystem system)
target_link_libraries(your_target 
    Boost::filesystem 
    Boost::system
)
```

**Compilation:**
```bash
# Direct compilation
g++ -std=c++11 -lboost_filesystem -lboost_system program.cpp -o program

# With CMake
mkdir build && cd build
cmake ..
make
```

## Learning Topics

### 1. Path Manipulation - The Foundation

Understanding path manipulation is crucial as it forms the foundation of all file system operations.

#### Core Concepts:
- **Path construction**: Building paths from strings and components
- **Path parsing**: Extracting meaningful parts from existing paths
- **Path concatenation**: Safely joining path components
- **Path resolution**: Converting relative paths to absolute paths
- **Cross-platform compatibility**: Handling different path separators and conventions

#### What You'll Learn:
```cpp
// From this basic approach with potential issues:
std::string path = "C:\\Users\\John\\Documents\\file.txt";  // Windows-specific
std::string unix_path = "/home/john/documents/file.txt";    // Unix-specific

// To this robust, cross-platform approach:
namespace fs = boost::filesystem;
fs::path user_docs = fs::path(get_user_home()) / "Documents" / "file.txt";
fs::path canonical_path = fs::canonical(user_docs);
```

#### Common Pitfalls and Solutions:
- **Problem**: Hardcoded path separators (`/` vs `\`)
- **Solution**: Use `fs::path` operator `/` for automatic separator handling
- **Problem**: Relative path confusion
- **Solution**: Always resolve to canonical paths when needed

### 2. Directory Operations - Navigating the File System

Master the art of working with directories, from simple listing to complex traversal patterns.

#### Core Concepts:
- **Directory creation**: Single and recursive directory creation
- **Directory traversal**: Iterating through directory contents
- **Filtering and searching**: Finding specific files and directories
- **Temporary directories**: Working with system temp locations
- **Directory monitoring**: Detecting changes in directories

#### Real-world Scenarios:
- Building a file backup system that traverses source directories
- Creating a log file cleaner that finds old files across multiple directories
- Implementing a project template generator that creates directory structures

### 3. File Status and Permissions - Understanding File Properties

Learn to query and manipulate file attributes, permissions, and metadata.

#### Core Concepts:
- **File type detection**: Regular files, directories, symlinks, etc.
- **Permission systems**: Unix permissions, Windows ACLs
- **File metadata**: Size, timestamps, ownership
- **Symbolic link handling**: Following vs. not following links
- **File system capabilities**: What's supported on different platforms

#### Security Implications:
- Understanding permission models across platforms
- Safe handling of symbolic links to prevent security vulnerabilities
- Proper validation of file types before operations

### 4. Advanced File Operations - Beyond Basic Read/Write

Explore sophisticated file manipulation techniques that go beyond simple I/O.

#### Core Concepts:
- **Atomic operations**: Ensuring file operations complete successfully or not at all
- **File copying strategies**: Different copy options and their implications
- **File system monitoring**: Watching for changes
- **Batch operations**: Efficiently processing multiple files
- **Error recovery**: Handling partial failures gracefully

### 5. Cross-Platform Development - Write Once, Run Everywhere

Master the nuances of developing file system code that works consistently across different operating systems.

#### Platform-Specific Challenges:
- **Path separators**: `\` on Windows vs `/` on Unix
- **Drive letters**: Windows `C:` vs Unix mount points
- **Case sensitivity**: Windows case-insensitive vs Unix case-sensitive
- **Path length limits**: Different maximum path lengths
- **Character encoding**: Unicode handling across platforms
- **Permission models**: Windows ACLs vs Unix permissions

### 6. Performance Optimization - Efficient File System Operations

Learn techniques to make your file system code fast and efficient.

#### Performance Topics:
- **Bulk operations**: Processing multiple files efficiently
- **Memory usage**: Managing memory when traversing large directory trees
- **I/O optimization**: Minimizing disk access patterns
- **Caching strategies**: When and how to cache file system information
- **Asynchronous operations**: Non-blocking file system operations

### 7. Migration and Modernization - Boost to std::filesystem

Understand the evolution from Boost.Filesystem to C++17's std::filesystem and migration strategies.

#### Migration Considerations:
- **API compatibility**: What changed and what stayed the same
- **Performance differences**: Benchmarking old vs new
- **Feature gaps**: What's available in Boost but not in std
- **Backwards compatibility**: Supporting older C++ standards
- **Testing strategies**: Ensuring migration doesn't break functionality

## Code Examples and Deep Dives

### Basic Path Operations - Building Blocks

Understanding path operations is fundamental to all file system work. Let's explore the building blocks step by step.

#### Path Construction - The Right Way

```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

namespace fs = boost::filesystem;

void demonstrate_path_basics() {
    std::cout << "=== Path Construction Fundamentals ===\n\n";
    
    // Method 1: Direct construction from string
    fs::path current_dir = fs::current_path();
    std::cout << "Current directory: " << current_dir << "\n";
    
    // Method 2: Using the / operator (RECOMMENDED)
    fs::path file_path = current_dir / "example.txt";
    std::cout << "File path (portable): " << file_path << "\n";
    
    // Method 3: Platform-specific paths (AVOID when possible)
    fs::path absolute_path("/usr/local/bin/program");           // Unix
    fs::path windows_path("C:\\Program Files\\App\\app.exe");   // Windows
    std::cout << "Unix path: " << absolute_path << "\n";
    std::cout << "Windows path: " << windows_path << "\n";
    
    // Method 4: Building complex paths step by step
    fs::path complex_path = fs::path("home") / "user" / "projects" / "myapp" / "src" / "main.cpp";
    std::cout << "Complex path: " << complex_path << "\n";
    
    // Method 5: Converting from different string types
    std::string std_string = "/path/to/file.txt";
    std::wstring wide_string = L"C:\\path\\to\\file.txt";
    fs::path from_string(std_string);
    fs::path from_wstring(wide_string);
    
    std::cout << "\n=== Path Component Analysis ===\n";
    std::cout << "Analyzing path: " << file_path << "\n";
    std::cout << "  Root name: '" << file_path.root_name() << "'\n";
    std::cout << "  Root directory: '" << file_path.root_directory() << "'\n";
    std::cout << "  Root path: '" << file_path.root_path() << "'\n";
    std::cout << "  Relative path: '" << file_path.relative_path() << "'\n";
    std::cout << "  Parent path: '" << file_path.parent_path() << "'\n";
    std::cout << "  Filename: '" << file_path.filename() << "'\n";
    std::cout << "  Stem: '" << file_path.stem() << "'\n";
    std::cout << "  Extension: '" << file_path.extension() << "'\n";
    
    // Boolean checks
    std::cout << "\n=== Path Properties ===\n";
    std::cout << "  Is absolute: " << std::boolalpha << file_path.is_absolute() << "\n";
    std::cout << "  Is relative: " << file_path.is_relative() << "\n";
    std::cout << "  Has root name: " << file_path.has_root_name() << "\n";
    std::cout << "  Has root directory: " << file_path.has_root_directory() << "\n";
    std::cout << "  Has parent path: " << file_path.has_parent_path() << "\n";
    std::cout << "  Has filename: " << file_path.has_filename() << "\n";
    std::cout << "  Has stem: " << file_path.has_stem() << "\n";
    std::cout << "  Has extension: " << file_path.has_extension() << "\n";
}

void demonstrate_path_manipulation() {
    std::cout << "\n=== Advanced Path Manipulation ===\n\n";
    
    fs::path base_path("/home/user/documents");
    fs::path relative_path("projects/myapp");
    
    // Path concatenation (the safe way)
    fs::path combined = base_path / relative_path;
    std::cout << "Combined path: " << combined << "\n";
    
    // String concatenation vs path concatenation
    fs::path wrong_way = base_path.string() + relative_path.string();  // WRONG!
    fs::path right_way = base_path / relative_path;                     // CORRECT!
    std::cout << "Wrong concatenation: " << wrong_way << "\n";
    std::cout << "Right concatenation: " << right_way << "\n";
    
    // Component replacement
    fs::path modified = combined;
    fs::path old_filename = modified.filename();
    modified.replace_filename("config.ini");
    std::cout << "Changed '" << old_filename << "' to '" << modified.filename() << "'\n";
    std::cout << "Full path: " << modified << "\n";
    
    // Extension manipulation
    fs::path original_ext = modified.extension();
    modified.replace_extension(".json");
    std::cout << "Changed extension from '" << original_ext 
              << "' to '" << modified.extension() << "'\n";
    std::cout << "Final path: " << modified << "\n";
    
    // Path normalization - handling messy paths
    std::cout << "\n=== Path Normalization ===\n";
    fs::path messy_path("/home/user/../user/./documents//file.txt");
    std::cout << "Messy path: " << messy_path << "\n";
    
    try {
        // canonical() requires the path to exist
        // For demonstration, we'll use a path that exists
        fs::path clean_path = fs::current_path() / ".." / "." / fs::current_path().filename();
        fs::path canonical_path = fs::canonical(clean_path);
        std::cout << "Before canonical: " << clean_path << "\n";
        std::cout << "After canonical: " << canonical_path << "\n";
    } catch (const fs::filesystem_error& e) {
        std::cout << "Note: canonical() requires existing paths\n";
        // Use weakly_canonical for non-existing paths (Boost 1.60+)
        // fs::path weakly_canonical_path = fs::weakly_canonical(messy_path);
    }
    
    // Relative path calculation
    std::cout << "\n=== Relative Path Calculation ===\n";
    fs::path from("/home/user/projects");
    fs::path to("/home/user/documents/file.txt");
    
    try {
        // This requires both paths to exist for fs::relative
        std::cout << "From: " << from << "\n";
        std::cout << "To: " << to << "\n";
        // fs::path relative = fs::relative(to, from);  // May require existing paths
        // std::cout << "Relative path: " << relative << "\n";
        
        // Manual relative path calculation for demonstration
        std::cout << "Relative path would be: ../documents/file.txt\n";
    } catch (const fs::filesystem_error& e) {
        std::cout << "Relative calculation error: " << e.what() << "\n";
    }
}

// Utility function to safely demonstrate path operations
void safe_path_demo() {
    std::cout << "\n=== Safe Path Operations ===\n";
    
    // Always validate paths before operations
    auto validate_path = [](const fs::path& p) -> bool {
        if (p.empty()) {
            std::cerr << "Error: Empty path\n";
            return false;
        }
        
        if (p.string().length() > 255) {  // Simplified check
            std::cerr << "Error: Path too long\n";
            return false;
        }
        
        // Check for invalid characters (basic check)
        std::string path_str = p.string();
        if (path_str.find('\0') != std::string::npos) {
            std::cerr << "Error: Null character in path\n";
            return false;
        }
        
        return true;
    };
    
    fs::path test_path = fs::current_path() / "test_file.txt";
    
    if (validate_path(test_path)) {
        std::cout << "Path validation passed: " << test_path << "\n";
        
        // Safe path operations
        std::cout << "Path exists: " << std::boolalpha << fs::exists(test_path) << "\n";
        std::cout << "Is absolute: " << test_path.is_absolute() << "\n";
        std::cout << "Generic string: " << test_path.generic_string() << "\n";
        std::cout << "Native string: " << test_path.string() << "\n";
    }
}
```

#### Understanding Path Components with Visual Examples

```
Example Path: "/home/user/projects/myapp/src/main.cpp"

┌─────────────────────────────────────────────────────────────┐
│                    Complete Path                            │
└─────────────────────────────────────────────────────────────┘
├─┤                                                     ├─────┤
root_name                                              filename
(empty on Unix)                                        "main.cpp"
    ├─┤
    root_directory                                      ├─────┤
    "/"                                                  stem
                                                        "main"
├───────┤                                                    ├┤
root_path                                              extension
"/"                                                     ".cpp"

├───────────────────────────────────────────────────┤
                relative_path
    "home/user/projects/myapp/src"

├─────────────────────────────────────────────┤
              parent_path
    "/home/user/projects/myapp/src"
```

**Windows Path Example: "C:\\Program Files\\MyApp\\config.ini"**
```
├─┤├─┤                                            ├────────────┤
│C:││\\│                                          │config.ini  │
│  ││  │                                          │            │
root root_directory                               filename
name "\"
├───┤                                             ├──────┤├───┤
root_path                                         stem    ext
"C:\"                                             "config"".ini"
```

### Directory Operations
```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

namespace fs = boost::filesystem;

void demonstrate_directory_operations() {
    fs::path test_dir = fs::current_path() / "test_directory";
    
    try {
        // Create directory
        if (fs::create_directory(test_dir)) {
            std::cout << "Created directory: " << test_dir << "\n";
        } else {
            std::cout << "Directory already exists: " << test_dir << "\n";
        }
        
        // Create subdirectories
        fs::path sub_dir = test_dir / "subdir1" / "subdir2";
        fs::create_directories(sub_dir);
        std::cout << "Created subdirectories: " << sub_dir << "\n";
        
        // Create some test files
        std::vector<fs::path> test_files = {
            test_dir / "file1.txt",
            test_dir / "file2.dat",
            sub_dir / "nested_file.log"
        };
        
        for (const auto& file : test_files) {
            std::ofstream ofs(file.string());
            ofs << "Test content for " << file.filename() << "\n";
            std::cout << "Created file: " << file << "\n";
        }
        
        // List directory contents
        std::cout << "\nDirectory contents of " << test_dir << ":\n";
        for (const auto& entry : fs::directory_iterator(test_dir)) {
            std::cout << "  " << entry.path().filename();
            if (fs::is_directory(entry)) {
                std::cout << " [DIR]";
            } else {
                std::cout << " (" << fs::file_size(entry) << " bytes)";
            }
            std::cout << "\n";
        }
        
        // Cleanup
        fs::remove_all(test_dir);
        std::cout << "\nCleaned up test directory\n";
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
}

void demonstrate_recursive_traversal() {
    fs::path search_path = fs::current_path();
    
    std::cout << "Recursive directory traversal of " << search_path << ":\n";
    
    int file_count = 0;
    int dir_count = 0;
    uintmax_t total_size = 0;
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(search_path)) {
            if (fs::is_regular_file(entry)) {
                file_count++;
                total_size += fs::file_size(entry);
                std::cout << "FILE: " << entry.path() << " (" 
                          << fs::file_size(entry) << " bytes)\n";
            } else if (fs::is_directory(entry)) {
                dir_count++;
                std::cout << "DIR:  " << entry.path() << "\n";
            }
            
            // Limit output for demonstration
            if (file_count + dir_count > 20) {
                std::cout << "... (truncated)\n";
                break;
            }
        }
        
        std::cout << "\nSummary:\n";
        std::cout << "  Files: " << file_count << "\n";
        std::cout << "  Directories: " << dir_count << "\n";
        std::cout << "  Total size: " << total_size << " bytes\n";
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error during traversal: " << e.what() << "\n";
    }
}
```

### File Operations and Status
```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <ctime>

namespace fs = boost::filesystem;

void demonstrate_file_status() {
    fs::path test_file = fs::current_path() / "status_test.txt";
    
    try {
        // Create a test file
        {
            std::ofstream ofs(test_file.string());
            ofs << "This is a test file for status checking.\n";
            ofs << "It contains multiple lines of text.\n";
            ofs << "Line 3 of the test file.\n";
        }
        
        if (fs::exists(test_file)) {
            std::cout << "File exists: " << test_file << "\n";
            
            // File type checks
            std::cout << "Is regular file: " << std::boolalpha 
                      << fs::is_regular_file(test_file) << "\n";
            std::cout << "Is directory: " << fs::is_directory(test_file) << "\n";
            std::cout << "Is symlink: " << fs::is_symlink(test_file) << "\n";
            
            // File size
            std::cout << "File size: " << fs::file_size(test_file) << " bytes\n";
            
            // File times
            std::time_t write_time = fs::last_write_time(test_file);
            std::cout << "Last write time: " << std::ctime(&write_time);
            
            // File permissions (Unix-style)
            fs::file_status status = fs::status(test_file);
            fs::perms permissions = status.permissions();
            
            std::cout << "Permissions: ";
            std::cout << ((permissions & fs::owner_read) ? "r" : "-");
            std::cout << ((permissions & fs::owner_write) ? "w" : "-");
            std::cout << ((permissions & fs::owner_exe) ? "x" : "-");
            std::cout << ((permissions & fs::group_read) ? "r" : "-");
            std::cout << ((permissions & fs::group_write) ? "w" : "-");
            std::cout << ((permissions & fs::group_exe) ? "x" : "-");
            std::cout << ((permissions & fs::others_read) ? "r" : "-");
            std::cout << ((permissions & fs::others_write) ? "w" : "-");
            std::cout << ((permissions & fs::others_exe) ? "x" : "-");
            std::cout << "\n";
        }
        
        // Copy file
        fs::path copy_file = test_file;
        copy_file.replace_extension(".backup");
        
        fs::copy_file(test_file, copy_file);
        std::cout << "Copied to: " << copy_file << "\n";
        
        // Compare files
        if (fs::file_size(test_file) == fs::file_size(copy_file)) {
            std::cout << "Files have same size\n";
        }
        
        // Cleanup
        fs::remove(test_file);
        fs::remove(copy_file);
        std::cout << "Cleaned up test files\n";
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
}

void demonstrate_file_operations() {
    fs::path source_dir = fs::current_path() / "source";
    fs::path dest_dir = fs::current_path() / "destination";
    
    try {
        // Create source directory with files
        fs::create_directory(source_dir);
        
        for (int i = 1; i <= 3; ++i) {
            fs::path file = source_dir / ("file" + std::to_string(i) + ".txt");
            std::ofstream ofs(file.string());
            ofs << "Content of file " << i << "\n";
        }
        
        // Copy entire directory
        fs::copy_directory(source_dir, dest_dir);
        
        // Copy files to destination
        for (const auto& entry : fs::directory_iterator(source_dir)) {
            if (fs::is_regular_file(entry)) {
                fs::path dest_file = dest_dir / entry.path().filename();
                fs::copy_file(entry.path(), dest_file);
            }
        }
        
        std::cout << "Copied directory structure\n";
        
        // Rename a file
        fs::path old_name = dest_dir / "file1.txt";
        fs::path new_name = dest_dir / "renamed_file.txt";
        
        if (fs::exists(old_name)) {
            fs::rename(old_name, new_name);
            std::cout << "Renamed " << old_name.filename() 
                      << " to " << new_name.filename() << "\n";
        }
        
        // List final structure
        std::cout << "\nFinal directory structure:\n";
        std::cout << "Source directory:\n";
        for (const auto& entry : fs::directory_iterator(source_dir)) {
            std::cout << "  " << entry.path().filename() << "\n";
        }
        
        std::cout << "Destination directory:\n";
        for (const auto& entry : fs::directory_iterator(dest_dir)) {
            std::cout << "  " << entry.path().filename() << "\n";
        }
        
        // Cleanup
        fs::remove_all(source_dir);
        fs::remove_all(dest_dir);
        std::cout << "\nCleaned up test directories\n";
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
}
```

### File System Utilities
```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <set>
#include <regex>

namespace fs = boost::filesystem;

class FileSystemUtils {
public:
    // Find files by extension
    static std::vector<fs::path> findFilesByExtension(
        const fs::path& dir, 
        const std::string& extension) {
        
        std::vector<fs::path> matches;
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            return matches;
        }
        
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (fs::is_regular_file(entry) && 
                entry.path().extension() == extension) {
                matches.push_back(entry.path());
            }
        }
        
        return matches;
    }
    
    // Find files by pattern
    static std::vector<fs::path> findFilesByPattern(
        const fs::path& dir, 
        const std::regex& pattern) {
        
        std::vector<fs::path> matches;
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            return matches;
        }
        
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (fs::is_regular_file(entry)) {
                std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, pattern)) {
                    matches.push_back(entry.path());
                }
            }
        }
        
        return matches;
    }
    
    // Calculate directory size
    static uintmax_t calculateDirectorySize(const fs::path& dir) {
        uintmax_t total_size = 0;
        
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            return 0;
        }
        
        try {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (fs::is_regular_file(entry)) {
                    total_size += fs::file_size(entry);
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error calculating size: " << e.what() << "\n";
        }
        
        return total_size;
    }
    
    // Get unique filename
    static fs::path getUniqueFilename(const fs::path& base_path) {
        if (!fs::exists(base_path)) {
            return base_path;
        }
        
        fs::path parent = base_path.parent_path();
        std::string stem = base_path.stem().string();
        std::string extension = base_path.extension().string();
        
        int counter = 1;
        fs::path unique_path;
        
        do {
            std::string new_name = stem + "_" + std::to_string(counter) + extension;
            unique_path = parent / new_name;
            counter++;
        } while (fs::exists(unique_path));
        
        return unique_path;
    }
    
    // Safe file operations with backup
    static bool safeReplace(const fs::path& source, const fs::path& target) {
        try {
            fs::path backup = target;
            backup += ".backup";
            
            // Create backup if target exists
            if (fs::exists(target)) {
                fs::copy_file(target, backup);
            }
            
            // Replace with source
            fs::copy_file(source, target, fs::copy_option::overwrite_if_exists);
            
            // Remove backup on success
            if (fs::exists(backup)) {
                fs::remove(backup);
            }
            
            return true;
            
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Safe replace failed: " << e.what() << "\n";
            return false;
        }
    }
};

void demonstrate_file_utilities() {
    fs::path test_dir = fs::current_path() / "utility_test";
    
    try {
        // Create test structure
        fs::create_directory(test_dir);
        
        // Create various test files
        std::vector<std::string> files = {
            "document.txt", "image.jpg", "backup.txt.bak",
            "data.csv", "config.ini", "log_2023.txt"
        };
        
        for (const auto& filename : files) {
            fs::path file_path = test_dir / filename;
            std::ofstream ofs(file_path.string());
            ofs << "Test content for " << filename << "\n";
        }
        
        // Find files by extension
        auto txt_files = FileSystemUtils::findFilesByExtension(test_dir, ".txt");
        std::cout << "Text files found:\n";
        for (const auto& file : txt_files) {
            std::cout << "  " << file.filename() << "\n";
        }
        
        // Find files by pattern
        std::regex log_pattern(R"(log_\d+\.txt)");
        auto log_files = FileSystemUtils::findFilesByPattern(test_dir, log_pattern);
        std::cout << "\nLog files found:\n";
        for (const auto& file : log_files) {
            std::cout << "  " << file.filename() << "\n";
        }
        
        // Calculate directory size
        uintmax_t dir_size = FileSystemUtils::calculateDirectorySize(test_dir);
        std::cout << "\nDirectory size: " << dir_size << " bytes\n";
        
        // Test unique filename generation
        fs::path base_file = test_dir / "document.txt";
        fs::path unique_file = FileSystemUtils::getUniqueFilename(base_file);
        std::cout << "Unique filename: " << unique_file.filename() << "\n";
        
        // Cleanup
        fs::remove_all(test_dir);
        std::cout << "\nCleaned up test directory\n";
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
}
```

### Cross-Platform Considerations
```cpp
#include <boost/filesystem.hpp>
#include <iostream>

namespace fs = boost::filesystem;

void demonstrate_cross_platform() {
    std::cout << "Cross-platform filesystem operations:\n\n";
    
    // Path separators
    std::cout << "Native path separator: '" 
              << fs::path::preferred_separator << "'\n";
    
    // Current working directory
    fs::path cwd = fs::current_path();
    std::cout << "Current directory: " << cwd << "\n";
    
    // Construct platform-appropriate paths
    fs::path config_dir;
    
#ifdef _WIN32
    // Windows: Use APPDATA
    const char* appdata = std::getenv("APPDATA");
    if (appdata) {
        config_dir = fs::path(appdata) / "MyApplication";
    } else {
        config_dir = cwd / "config";
    }
#else
    // Unix-like: Use home directory
    const char* home = std::getenv("HOME");
    if (home) {
        config_dir = fs::path(home) / ".myapplication";
    } else {
        config_dir = cwd / "config";
    }
#endif
    
    std::cout << "Config directory: " << config_dir << "\n";
    
    // Temporary directory
    fs::path temp_dir = fs::temp_directory_path();
    std::cout << "Temporary directory: " << temp_dir << "\n";
    
    // Create a temporary file
    fs::path temp_file = fs::unique_path(temp_dir / "myapp_%%%%-%%%%-%%%%-%%%%");
    std::cout << "Unique temp file: " << temp_file << "\n";
    
    // Demonstrate path conversion
    std::string generic_path = "/path/to/file.txt";
    fs::path converted(generic_path);
    
    std::cout << "Generic path: " << generic_path << "\n";
    std::cout << "Native path: " << converted.make_preferred() << "\n";
    std::cout << "Generic string: " << converted.generic_string() << "\n";
    std::cout << "Native string: " << converted.string() << "\n";
    
    // Drive/root information
    std::cout << "\nPath analysis for current directory:\n";
    std::cout << "  Root name: " << cwd.root_name() << "\n";
    std::cout << "  Root directory: " << cwd.root_directory() << "\n";
    std::cout << "  Root path: " << cwd.root_path() << "\n";
    std::cout << "  Has root name: " << std::boolalpha 
              << cwd.has_root_name() << "\n";
    std::cout << "  Has root directory: " << cwd.has_root_directory() << "\n";
    std::cout << "  Is absolute: " << cwd.is_absolute() << "\n";
    std::cout << "  Is relative: " << cwd.is_relative() << "\n";
}
```

## Error Handling and Troubleshooting

### Understanding Boost.Filesystem Exceptions

Boost.Filesystem uses exceptions to signal errors, making error handling explicit and mandatory. Understanding the exception hierarchy is crucial for robust applications.

#### Exception Hierarchy

```
std::exception
    └── boost::system::system_error
            └── boost::filesystem::filesystem_error
```

#### Common Exception Scenarios

```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <exception>

namespace fs = boost::filesystem;

void demonstrate_exception_handling() {
    std::cout << "=== Exception Handling Patterns ===\n\n";
    
    // 1. Basic exception handling
    try {
        fs::path non_existent = "/this/path/does/not/exist/file.txt";
        uintmax_t size = fs::file_size(non_existent);  // Will throw
        std::cout << "File size: " << size << "\n";
    } catch (const fs::filesystem_error& e) {
        std::cout << "Filesystem error: " << e.what() << "\n";
        std::cout << "Error code: " << e.code() << "\n";
        std::cout << "Path1: " << e.path1() << "\n";
        std::cout << "Path2: " << e.path2() << "\n";
    }
    
    // 2. Using error codes instead of exceptions
    boost::system::error_code ec;
    fs::path test_path = "/might/not/exist";
    
    // Non-throwing version
    bool exists = fs::exists(test_path, ec);
    if (ec) {
        std::cout << "Error checking existence: " << ec.message() << "\n";
    } else {
        std::cout << "Path exists: " << std::boolalpha << exists << "\n";
    }
    
    // 3. Comprehensive error handling function
    auto safe_file_operation = [](const fs::path& path) -> bool {
        try {
            if (!fs::exists(path)) {
                std::cerr << "Error: Path does not exist: " << path << "\n";
                return false;
            }
            
            if (!fs::is_regular_file(path)) {
                std::cerr << "Error: Not a regular file: " << path << "\n";
                return false;
            }
            
            // Attempt to get file size
            uintmax_t size = fs::file_size(path);
            std::cout << "File size: " << size << " bytes\n";
            return true;
            
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << "\n";
            std::cerr << "  Error code: " << e.code().value() << "\n";
            std::cerr << "  Category: " << e.code().category().name() << "\n";
            return false;
        } catch (const std::exception& e) {
            std::cerr << "General error: " << e.what() << "\n";
            return false;
        }
    };
    
    // Test the safe function
    fs::path current_file = __FILE__;  // This file should exist
    safe_file_operation(current_file);
}

// Error code patterns for performance-critical code
void demonstrate_error_codes() {
    std::cout << "\n=== Error Code Patterns ===\n";
    
    boost::system::error_code ec;
    fs::path test_dir = fs::current_path() / "test_operations";
    
    // Pattern 1: Check before operation
    if (!fs::exists(test_dir, ec) && !ec) {
        fs::create_directory(test_dir, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << ec.message() << "\n";
            return;
        }
    }
    
    // Pattern 2: Bulk operations with error accumulation
    std::vector<fs::path> files_to_create = {
        test_dir / "file1.txt",
        test_dir / "file2.txt",
        test_dir / "file3.txt"
    };
    
    int success_count = 0;
    for (const auto& file_path : files_to_create) {
        std::ofstream ofs(file_path.string());
        if (ofs.good()) {
            ofs << "Test content\n";
            success_count++;
        } else {
            std::cerr << "Failed to create: " << file_path << "\n";
        }
    }
    
    std::cout << "Successfully created " << success_count 
              << " out of " << files_to_create.size() << " files\n";
    
    // Cleanup
    fs::remove_all(test_dir, ec);
    if (ec) {
        std::cerr << "Warning: Failed to cleanup test directory: " << ec.message() << "\n";
    }
}
```

### Common Error Scenarios and Solutions

#### 1. Permission Denied Errors

```cpp
void handle_permission_errors() {
    std::cout << "=== Permission Error Handling ===\n";
    
    auto check_and_fix_permissions = [](const fs::path& path) {
        boost::system::error_code ec;
        
        // Check if we can read the file
        if (!fs::exists(path, ec)) {
            std::cout << "File doesn't exist: " << path << "\n";
            return false;
        }
        
        // Try to get file status
        fs::file_status status = fs::status(path, ec);
        if (ec) {
            std::cout << "Cannot get file status: " << ec.message() << "\n";
            return false;
        }
        
        // Check permissions
        fs::perms permissions = status.permissions();
        std::cout << "Current permissions for " << path.filename() << ": ";
        
        // Display permission bits
        std::cout << ((permissions & fs::owner_read) ? "r" : "-");
        std::cout << ((permissions & fs::owner_write) ? "w" : "-");
        std::cout << ((permissions & fs::owner_exe) ? "x" : "-");
        std::cout << "\n";
        
        // Try to make file readable if it isn't
        if (!(permissions & fs::owner_read)) {
            std::cout << "File is not readable, attempting to fix...\n";
            fs::permissions(path, fs::owner_read | permissions, ec);
            if (ec) {
                std::cout << "Failed to change permissions: " << ec.message() << "\n";
                return false;
            }
            std::cout << "Permissions updated successfully\n";
        }
        
        return true;
    };
    
    // Test with a file (create one for testing)
    fs::path test_file = fs::current_path() / "permission_test.txt";
    std::ofstream ofs(test_file.string());
    ofs << "Test content\n";
    ofs.close();
    
    if (check_and_fix_permissions(test_file)) {
        std::cout << "Permission handling successful\n";
    }
    
    // Cleanup
    boost::system::error_code ec;
    fs::remove(test_file, ec);
}
```

#### 2. Long Path Names

```cpp
void handle_long_paths() {
    std::cout << "\n=== Long Path Handling ===\n";
    
    // Check system path limits
    auto check_path_limits = [](const fs::path& path) {
        std::string path_str = path.string();
        
        // Basic length checks
        if (path_str.length() > 260) {  // Windows MAX_PATH
            std::cout << "Warning: Path exceeds Windows MAX_PATH limit (260 chars)\n";
            std::cout << "Path length: " << path_str.length() << "\n";
            std::cout << "Consider using \\\\?\\ prefix on Windows or shorter paths\n";
            return false;
        }
        
        if (path_str.length() > 4096) {  // Typical Unix limit
            std::cout << "Warning: Path exceeds typical Unix PATH_MAX limit\n";
            return false;
        }
        
        return true;
    };
    
    // Generate a long path for testing
    fs::path long_path = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        long_path /= "very_long_directory_name_that_might_cause_issues_on_some_systems";
    }
    long_path /= "final_file.txt";
    
    if (!check_path_limits(long_path)) {
        std::cout << "Path too long, creating shorter alternative...\n";
        
        // Create shorter alternative
        fs::path short_path = fs::current_path() / "short" / "path" / "file.txt";
        std::cout << "Using shorter path: " << short_path << "\n";
    }
}
```

#### 3. Unicode and Special Characters

```cpp
void handle_unicode_paths() {
    std::cout << "\n=== Unicode Path Handling ===\n";
    
    // Test with various Unicode characters
    std::vector<std::string> test_names = {
        "normal_file.txt",
        "файл.txt",           // Cyrillic
        "файł.txt",           // Polish
        "文件.txt",           // Chinese
        "ファイル.txt",        // Japanese
        "special chars !@#$%.txt",
        "spaces in name.txt"
    };
    
    fs::path test_dir = fs::current_path() / "unicode_test";
    boost::system::error_code ec;
    
    // Create test directory
    fs::create_directory(test_dir, ec);
    if (ec) {
        std::cout << "Failed to create test directory: " << ec.message() << "\n";
        return;
    }
    
    for (const auto& name : test_names) {
        fs::path file_path = test_dir / name;
        
        try {
            std::ofstream ofs(file_path.string());
            if (ofs.good()) {
                ofs << "Content for " << name << "\n";
                std::cout << "✓ Created: " << file_path.filename() << "\n";
            } else {
                std::cout << "✗ Failed to create: " << name << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "✗ Exception creating " << name << ": " << e.what() << "\n";
        }
    }
    
    // List created files
    std::cout << "\nCreated files:\n";
    for (const auto& entry : fs::directory_iterator(test_dir, ec)) {
        if (!ec) {
            std::cout << "  " << entry.path().filename() << "\n";
        }
    }
    
    // Cleanup
    fs::remove_all(test_dir, ec);
    if (ec) {
        std::cout << "Warning: Failed to cleanup: " << ec.message() << "\n";
    }
}
```

### Debugging File System Issues

#### Debug Helper Functions

```cpp
class FileSystemDebugger {
public:
    static void diagnose_path(const fs::path& path) {
        std::cout << "\n=== Path Diagnosis: " << path << " ===\n";
        
        boost::system::error_code ec;
        
        // Basic existence check
        bool exists = fs::exists(path, ec);
        std::cout << "Exists: " << std::boolalpha << exists;
        if (ec) std::cout << " (Error: " << ec.message() << ")";
        std::cout << "\n";
        
        if (!exists) return;
        
        // File type
        std::cout << "Type: ";
        if (fs::is_regular_file(path, ec)) std::cout << "Regular file";
        else if (fs::is_directory(path, ec)) std::cout << "Directory";
        else if (fs::is_symlink(path, ec)) std::cout << "Symbolic link";
        else std::cout << "Other/Unknown";
        if (ec) std::cout << " (Error: " << ec.message() << ")";
        std::cout << "\n";
        
        // Size (for files)
        if (fs::is_regular_file(path, ec) && !ec) {
            uintmax_t size = fs::file_size(path, ec);
            if (!ec) {
                std::cout << "Size: " << size << " bytes\n";
            } else {
                std::cout << "Size: Error getting size - " << ec.message() << "\n";
            }
        }
        
        // Permissions
        fs::file_status status = fs::status(path, ec);
        if (!ec) {
            fs::perms perms = status.permissions();
            std::cout << "Permissions: ";
            std::cout << ((perms & fs::owner_read) ? "r" : "-");
            std::cout << ((perms & fs::owner_write) ? "w" : "-");
            std::cout << ((perms & fs::owner_exe) ? "x" : "-");
            std::cout << ((perms & fs::group_read) ? "r" : "-");
            std::cout << ((perms & fs::group_write) ? "w" : "-");
            std::cout << ((perms & fs::group_exe) ? "x" : "-");
            std::cout << ((perms & fs::others_read) ? "r" : "-");
            std::cout << ((perms & fs::others_write) ? "w" : "-");
            std::cout << ((perms & fs::others_exe) ? "x" : "-");
            std::cout << "\n";
        }
        
        // Last write time
        try {
            std::time_t write_time = fs::last_write_time(path);
            std::cout << "Last modified: " << std::ctime(&write_time);
        } catch (const fs::filesystem_error& e) {
            std::cout << "Last modified: Error - " << e.what() << "\n";
        }
    }
    
    static void diagnose_directory_traversal(const fs::path& dir) {
        std::cout << "\n=== Directory Traversal Diagnosis: " << dir << " ===\n";
        
        boost::system::error_code ec;
        
        if (!fs::exists(dir, ec)) {
            std::cout << "Directory doesn't exist\n";
            return;
        }
        
        if (!fs::is_directory(dir, ec)) {
            std::cout << "Path is not a directory\n";
            return;
        }
        
        // Try simple iteration
        try {
            int count = 0;
            for (const auto& entry : fs::directory_iterator(dir)) {
                count++;
                if (count <= 5) {  // Show first 5 entries
                    std::cout << "  " << entry.path().filename();
                    if (fs::is_directory(entry)) std::cout << " [DIR]";
                    std::cout << "\n";
                }
            }
            
            if (count > 5) {
                std::cout << "  ... and " << (count - 5) << " more entries\n";
            }
            
            std::cout << "Total entries: " << count << "\n";
            
        } catch (const fs::filesystem_error& e) {
            std::cout << "Error during iteration: " << e.what() << "\n";
        }
    }
};

void demonstrate_debugging() {
    std::cout << "=== File System Debugging ===\n";
    
    // Debug current directory
    FileSystemDebugger::diagnose_path(fs::current_path());
    
    // Debug a file (this source file)
    FileSystemDebugger::diagnose_path(__FILE__);
    
    // Debug directory traversal
    FileSystemDebugger::diagnose_directory_traversal(fs::current_path());
}
```

### Best Practices for Error Handling

#### 1. Always Use Error Codes for Performance-Critical Code

```cpp
// Good: Error codes don't throw exceptions
boost::system::error_code ec;
bool exists = fs::exists(path, ec);
if (ec) {
    // Handle error without exception overhead
    log_error("File existence check failed", ec);
    return false;
}
```

#### 2. Provide Context in Error Messages

```cpp
void copy_with_context(const fs::path& source, const fs::path& dest) {
    try {
        fs::copy_file(source, dest);
    } catch (const fs::filesystem_error& e) {
        std::string context = "Failed to copy '" + source.string() + 
                             "' to '" + dest.string() + "': " + e.what();
        throw std::runtime_error(context);
    }
}
```

#### 3. Implement Retry Logic for Transient Failures

```cpp
template<typename Operation>
bool retry_operation(Operation op, int max_attempts = 3, int delay_ms = 100) {
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        try {
            op();
            return true;  // Success
        } catch (const fs::filesystem_error& e) {
            if (attempt == max_attempts) {
                std::cerr << "Operation failed after " << max_attempts 
                         << " attempts: " << e.what() << "\n";
                return false;
            }
            
            std::cerr << "Attempt " << attempt << " failed, retrying in " 
                     << delay_ms << "ms...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }
    return false;
}
```

## Practical Exercises and Projects

### Project 1: Smart File Organizer

Build a comprehensive file organizer that sorts files based on various criteria.

#### Requirements:
- Scan directories recursively
- Organize files by type, date, size, or custom rules
- Handle duplicates intelligently
- Provide progress feedback
- Support undo operations

#### Implementation Framework:

```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <vector>
#include <regex>
#include <chrono>

namespace fs = boost::filesystem;

class FileOrganizer {
private:
    struct FileInfo {
        fs::path original_path;
        fs::path suggested_path;
        uintmax_t size;
        std::time_t last_modified;
        std::string category;
        bool is_duplicate;
    };
    
    std::vector<FileInfo> files_to_organize;
    std::map<std::string, std::vector<fs::path>> file_categories;
    
public:
    // Scan directory and categorize files
    void scan_directory(const fs::path& source_dir) {
        std::cout << "Scanning directory: " << source_dir << "\n";
        
        try {
            for (const auto& entry : fs::recursive_directory_iterator(source_dir)) {
                if (fs::is_regular_file(entry)) {
                    FileInfo info;
                    info.original_path = entry.path();
                    info.size = fs::file_size(entry);
                    info.last_modified = fs::last_write_time(entry);
                    info.category = categorize_file(entry.path());
                    info.is_duplicate = false;  // TODO: Implement duplicate detection
                    
                    files_to_organize.push_back(info);
                    file_categories[info.category].push_back(info.original_path);
                }
            }
            
            std::cout << "Found " << files_to_organize.size() << " files in " 
                      << file_categories.size() << " categories\n";
                      
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error scanning directory: " << e.what() << "\n";
        }
    }
    
    // Categorize file based on extension and content
    std::string categorize_file(const fs::path& file_path) {
        std::string ext = file_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        // Define categories
        std::map<std::string, std::vector<std::string>> categories = {
            {"Images", {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}},
            {"Documents", {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"}},
            {"Spreadsheets", {".xls", ".xlsx", ".csv", ".ods"}},
            {"Presentations", {".ppt", ".pptx", ".odp"}},
            {"Audio", {".mp3", ".wav", ".flac", ".aac", ".ogg"}},
            {"Video", {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}},
            {"Archives", {".zip", ".rar", ".7z", ".tar", ".gz"}},
            {"Code", {".cpp", ".h", ".py", ".js", ".html", ".css", ".java"}},
            {"Executables", {".exe", ".msi", ".deb", ".rpm", ".dmg"}}
        };
        
        for (const auto& [category, extensions] : categories) {
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                return category;
            }
        }
        
        return "Other";
    }
    
    // Generate organized directory structure
    void generate_organization_plan(const fs::path& target_dir) {
        for (auto& file_info : files_to_organize) {
            // Create category-based path
            fs::path category_dir = target_dir / file_info.category;
            
            // Add date-based subdirectory for some categories
            if (file_info.category == "Images" || file_info.category == "Documents") {
                std::tm* time_info = std::localtime(&file_info.last_modified);
                char date_str[20];
                std::strftime(date_str, sizeof(date_str), "%Y-%m", time_info);
                category_dir /= date_str;
            }
            
            file_info.suggested_path = category_dir / file_info.original_path.filename();
        }
    }
    
    // Execute the organization plan
    bool execute_organization(bool dry_run = true) {
        std::cout << "\n" << (dry_run ? "DRY RUN - " : "") 
                  << "Organizing files...\n";
        
        int success_count = 0;
        int error_count = 0;
        
        for (const auto& file_info : files_to_organize) {
            try {
                std::cout << file_info.original_path.filename() 
                         << " -> " << file_info.suggested_path << "\n";
                
                if (!dry_run) {
                    // Create directory structure
                    fs::create_directories(file_info.suggested_path.parent_path());
                    
                    // Move or copy file
                    if (fs::exists(file_info.suggested_path)) {
                        // Handle naming conflict
                        fs::path unique_path = generate_unique_filename(file_info.suggested_path);
                        fs::copy_file(file_info.original_path, unique_path);
                    } else {
                        fs::copy_file(file_info.original_path, file_info.suggested_path);
                    }
                }
                
                success_count++;
                
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Error organizing " << file_info.original_path 
                         << ": " << e.what() << "\n";
                error_count++;
            }
        }
        
        std::cout << "\nResults: " << success_count << " successful, " 
                  << error_count << " errors\n";
        
        return error_count == 0;
    }
    
    // Generate unique filename if file already exists
    fs::path generate_unique_filename(const fs::path& base_path) {
        if (!fs::exists(base_path)) {
            return base_path;
        }
        
        fs::path parent = base_path.parent_path();
        std::string stem = base_path.stem().string();
        std::string extension = base_path.extension().string();
        
        int counter = 1;
        fs::path unique_path;
        
        do {
            std::string new_name = stem + "_" + std::to_string(counter) + extension;
            unique_path = parent / new_name;
            counter++;
        } while (fs::exists(unique_path));
        
        return unique_path;
    }
    
    // Print organization summary
    void print_summary() {
        std::cout << "\nOrganization Summary:\n";
        for (const auto& [category, files] : file_categories) {
            std::cout << category << ": " << files.size() << " files\n";
        }
    }
};

// Usage example
void demonstrate_file_organizer() {
    FileOrganizer organizer;
    
    // Scan current directory
    organizer.scan_directory(fs::current_path());
    
    // Print summary
    organizer.print_summary();
    
    // Generate organization plan
    fs::path target_dir = fs::current_path() / "Organized";
    organizer.generate_organization_plan(target_dir);
    
    // Execute dry run
    organizer.execute_organization(true);  // Dry run first
    
    // Uncomment to actually organize files
    // organizer.execute_organization(false);
}
```

### Project 2: Intelligent Backup System

Create a backup system that handles incremental backups, compression, and restoration.

#### Key Features to Implement:

```cpp
class BackupSystem {
private:
    struct BackupMetadata {
        std::time_t timestamp;
        std::string backup_type;  // "full", "incremental", "differential"
        std::vector<fs::path> files_backed_up;
        uintmax_t total_size;
        std::string checksum;
    };
    
    fs::path source_directory;
    fs::path backup_directory;
    std::vector<BackupMetadata> backup_history;
    
public:
    // TODO: Implement these methods
    bool create_full_backup();
    bool create_incremental_backup();
    bool verify_backup_integrity();
    bool restore_from_backup(const std::time_t& timestamp);
    std::vector<fs::path> find_changed_files_since(const std::time_t& since);
    std::string calculate_directory_checksum(const fs::path& dir);
};
```

### Project 3: Log File Analyzer

Build a comprehensive log analysis tool.

#### Requirements:
- Parse different log formats
- Extract timestamps, severity levels, and messages
- Generate statistics and reports
- Support filtering and searching
- Handle compressed log files

```cpp
class LogAnalyzer {
public:
    struct LogEntry {
        std::time_t timestamp;
        std::string level;      // DEBUG, INFO, WARN, ERROR, FATAL
        std::string component;
        std::string message;
        fs::path source_file;
        size_t line_number;
    };
    
    struct AnalysisReport {
        std::map<std::string, int> level_counts;
        std::map<std::string, int> component_counts;
        std::time_t earliest_entry;
        std::time_t latest_entry;
        size_t total_entries;
        std::vector<LogEntry> error_entries;
    };
    
    // TODO: Implement comprehensive log analysis
    void scan_log_directory(const fs::path& log_dir);
    std::vector<LogEntry> parse_log_file(const fs::path& log_file);
    AnalysisReport generate_report();
    void export_report(const fs::path& output_file);
};
```

### Project 4: Directory Synchronization Tool

Implement a robust directory synchronization system.

#### Challenge Requirements:
- Bidirectional synchronization
- Conflict resolution strategies
- Bandwidth-efficient updates
- Progress reporting
- Rollback capabilities

### Coding Challenges

#### Challenge 1: Path Manipulation Master
```cpp
// Implement these functions with proper error handling:

// 1. Find the longest common path prefix
fs::path find_common_prefix(const std::vector<fs::path>& paths);

// 2. Convert absolute path to relative from a base directory
fs::path make_relative_to(const fs::path& path, const fs::path& base);

// 3. Safely resolve symbolic links without infinite loops
fs::path safe_canonical(const fs::path& path, int max_symlinks = 20);

// 4. Generate a path that doesn't conflict with existing files
fs::path get_non_conflicting_path(const fs::path& desired_path);
```

#### Challenge 2: Advanced Directory Traversal
```cpp
// Implement a flexible directory iterator with filtering
class FilteredDirectoryIterator {
private:
    std::function<bool(const fs::path&)> filter;
    bool recursive;
    std::stack<fs::directory_iterator> iterator_stack;
    
public:
    FilteredDirectoryIterator(const fs::path& dir, 
                             std::function<bool(const fs::path&)> filter,
                             bool recursive = false);
    
    // TODO: Implement iterator interface
    FilteredDirectoryIterator& operator++();
    const fs::path& operator*() const;
    bool operator!=(const FilteredDirectoryIterator& other) const;
};

// Usage examples:
// Find all .cpp files
auto cpp_filter = [](const fs::path& p) { 
    return p.extension() == ".cpp"; 
};

// Find files larger than 1MB
auto large_file_filter = [](const fs::path& p) {
    return fs::is_regular_file(p) && fs::file_size(p) > 1024*1024;
};

// Find files modified in last 24 hours
auto recent_filter = [](const fs::path& p) {
    auto now = std::time(nullptr);
    auto file_time = fs::last_write_time(p);
    return (now - file_time) < 24*60*60;  // 24 hours in seconds
};
```

#### Challenge 3: Cross-Platform Compatibility
```cpp
// Write functions that handle platform-specific issues:

class CrossPlatformUtils {
public:
    // Get appropriate config directory for the platform
    static fs::path get_config_directory(const std::string& app_name);
    
    // Get appropriate temp directory with cleanup
    static fs::path get_temp_directory_with_cleanup();
    
    // Convert path to platform-appropriate format
    static std::string to_platform_string(const fs::path& path);
    
    // Check if path is valid on target platform
    static bool is_valid_path_for_platform(const fs::path& path, 
                                          const std::string& platform);
    
    // Get maximum path length for current platform
    static size_t get_max_path_length();
};
```

### Performance Optimization Challenges

#### Challenge 4: Efficient Large Directory Handling
```cpp
// Optimize for directories with millions of files
class LargeDirectoryProcessor {
public:
    // Process directory in chunks to avoid memory issues
    void process_large_directory(const fs::path& dir,
                                std::function<void(const fs::path&)> processor,
                                size_t chunk_size = 10000);
    
    // Parallel directory processing
    void parallel_process_directory(const fs::path& dir,
                                   std::function<void(const fs::path&)> processor,
                                   size_t num_threads = std::thread::hardware_concurrency());
    
    // Memory-efficient file counting
    uintmax_t count_files_memory_efficient(const fs::path& dir);
};
```

### Testing Your Implementation

#### Unit Test Examples
```cpp
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(FilesystemOperationsTest)

BOOST_AUTO_TEST_CASE(test_path_manipulation) {
    fs::path test_path = "/home/user/documents/file.txt";
    
    BOOST_CHECK_EQUAL(test_path.filename(), "file.txt");
    BOOST_CHECK_EQUAL(test_path.stem(), "file");
    BOOST_CHECK_EQUAL(test_path.extension(), ".txt");
    BOOST_CHECK_EQUAL(test_path.parent_path(), "/home/user/documents");
}

BOOST_AUTO_TEST_CASE(test_directory_operations) {
    fs::path test_dir = fs::temp_directory_path() / "boost_fs_test";
    
    // Test directory creation
    BOOST_REQUIRE(fs::create_directory(test_dir));
    BOOST_CHECK(fs::exists(test_dir));
    BOOST_CHECK(fs::is_directory(test_dir));
    
    // Test file creation
    fs::path test_file = test_dir / "test.txt";
    std::ofstream ofs(test_file.string());
    ofs << "test content";
    ofs.close();
    
    BOOST_CHECK(fs::exists(test_file));
    BOOST_CHECK(fs::is_regular_file(test_file));
    BOOST_CHECK_EQUAL(fs::file_size(test_file), 12);  // "test content" length
    
    // Cleanup
    fs::remove_all(test_dir);
    BOOST_CHECK(!fs::exists(test_dir));
}

BOOST_AUTO_TEST_CASE(test_error_handling) {
    fs::path non_existent = "/this/path/should/not/exist";
    
    // Test exception throwing version
    BOOST_CHECK_THROW(fs::file_size(non_existent), fs::filesystem_error);
    
    // Test error code version
    boost::system::error_code ec;
    uintmax_t size = fs::file_size(non_existent, ec);
    BOOST_CHECK(ec);  // Should have error
    BOOST_CHECK_EQUAL(size, static_cast<uintmax_t>(-1));
}

BOOST_AUTO_TEST_SUITE_END()
```

## Performance Considerations

### Directory Traversal
- Use appropriate iteration strategies (recursive vs non-recursive)
- Consider memory usage for large directory trees
- Implement progress reporting for long operations

### File Operations
- Batch operations when possible
- Use native API optimizations
- Handle large files efficiently

### Error Handling
- Always check for filesystem errors
- Implement retry logic for transient failures
- Provide meaningful error messages

## Best Practices

1. **Path Handling**
   - Always use fs::path for cross-platform compatibility
   - Validate paths before operations
   - Use canonical paths for comparisons
   - Handle Unicode filenames properly

2. **Exception Safety**
   - Wrap filesystem operations in try-catch blocks
   - Provide cleanup in case of failures
   - Use RAII for resource management

3. **Performance**
   - Cache frequently accessed path information
   - Use appropriate iteration methods
   - Consider asynchronous operations for large tasks

## Study Guide and Resources

### Week-by-Week Study Plan

#### Week 1: Foundations and Basic Operations
**Days 1-2: Path Fundamentals**
- Study `fs::path` class thoroughly
- Practice path construction and manipulation
- Understand cross-platform path differences
- Complete basic path exercises

**Days 3-4: Directory Operations**
- Learn directory creation and removal
- Master directory iteration patterns
- Practice filtering and searching
- Implement simple directory utilities

**Days 5-7: File Status and Properties**
- Understand file type detection
- Learn permission systems
- Practice file metadata operations
- Build file analysis tools

#### Week 2: Advanced Topics and Projects
**Days 1-2: Error Handling Mastery**
- Study exception vs error code patterns
- Practice robust error handling
- Implement retry mechanisms
- Debug common filesystem errors

**Days 3-4: Performance Optimization**
- Learn efficient directory traversal
- Study memory management for large operations
- Implement caching strategies
- Profile and optimize filesystem code

**Days 5-7: Real-World Projects**
- Build file organizer application
- Implement backup system
- Create log analyzer
- Develop custom filesystem utilities

### Essential Resources

#### Primary Documentation
- **Boost.Filesystem Official Documentation**
  - [Main Documentation](https://www.boost.org/doc/libs/release/libs/filesystem/)
  - [Tutorial](https://www.boost.org/doc/libs/release/libs/filesystem/doc/tutorial.html)
  - [Reference](https://www.boost.org/doc/libs/release/libs/filesystem/doc/reference.html)

#### Books and References
- **"The Boost C++ Libraries" by Boris Schäling** - Chapter on Filesystem
- **"Effective Modern C++" by Scott Meyers** - Exception safety principles
- **"C++ Concurrency in Action" by Anthony Williams** - For thread-safe filesystem operations

#### Online Resources
- **Boost Users Mailing List**: Get help from the community
- **Stack Overflow**: Search for "boost::filesystem" tagged questions
- **GitHub Examples**: Study real-world Boost.Filesystem usage
- **cppreference.com**: Compare with std::filesystem

#### Tools and Setup
```bash
# Development Environment Setup

# Install Boost (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libboost-all-dev

# Install Boost (CentOS/RHEL)
sudo yum install boost-devel

# Install Boost (macOS)
brew install boost

# Windows with vcpkg
vcpkg install boost-filesystem:x64-windows
```

#### CMake Integration
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(FilesystemLearning)

find_package(Boost REQUIRED COMPONENTS filesystem system)

add_executable(filesystem_demo main.cpp)
target_link_libraries(filesystem_demo 
    Boost::filesystem 
    Boost::system
)

# Enable C++11 or later
target_compile_features(filesystem_demo PRIVATE cxx_std_11)

# Add compiler flags for better debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(filesystem_demo PRIVATE -g -Wall -Wextra)
endif()
```

### Practice Exercises by Difficulty

#### Beginner Level (Week 1)

**Exercise 1: Path Explorer**
```cpp
// Create a program that analyzes any given path
void analyze_path(const std::string& path_str) {
    // TODO: 
    // 1. Create fs::path from string
    // 2. Print all path components
    // 3. Check if path is absolute/relative
    // 4. Handle invalid paths gracefully
}
```

**Exercise 2: Simple Directory Lister**
```cpp
// List directory contents with file sizes
void list_directory(const fs::path& dir) {
    // TODO:
    // 1. Check if directory exists
    // 2. Iterate through directory
    // 3. Print filename and size
    // 4. Handle permissions errors
}
```

**Exercise 3: File Type Classifier**
```cpp
// Classify files by extension
std::map<std::string, int> classify_files(const fs::path& dir) {
    // TODO:
    // 1. Traverse directory recursively
    // 2. Count files by extension
    // 3. Return extension -> count mapping
    // 4. Handle various file types
}
```

#### Intermediate Level (Week 2)

**Exercise 4: Disk Usage Calculator**
```cpp
class DiskUsageCalculator {
public:
    struct Usage {
        uintmax_t total_size;
        int file_count;
        int directory_count;
        std::map<std::string, uintmax_t> size_by_extension;
    };
    
    Usage calculate_usage(const fs::path& path);
    void print_usage_report(const Usage& usage);
};
```

**Exercise 5: Smart File Finder**
```cpp
class FileFinder {
public:
    // Find files by various criteria
    std::vector<fs::path> find_by_name(const fs::path& root, 
                                      const std::string& pattern);
    std::vector<fs::path> find_by_size(const fs::path& root, 
                                      uintmax_t min_size, 
                                      uintmax_t max_size);
    std::vector<fs::path> find_by_date(const fs::path& root,
                                      std::time_t after,
                                      std::time_t before);
    std::vector<fs::path> find_empty_directories(const fs::path& root);
};
```

#### Advanced Level

**Exercise 6: Filesystem Watcher**
```cpp
class FilesystemWatcher {
public:
    enum EventType { Created, Modified, Deleted, Moved };
    
    struct Event {
        EventType type;
        fs::path path;
        fs::path old_path;  // for moves
        std::time_t timestamp;
    };
    
    void watch_directory(const fs::path& dir);
    std::vector<Event> get_events();
    void stop_watching();
};
```

### Debugging Guide

#### Common Issues and Solutions

**Issue 1: "Permission denied" errors**
```cpp
// Solution: Always check permissions first
bool can_access_file(const fs::path& path) {
    boost::system::error_code ec;
    fs::file_status status = fs::status(path, ec);
    
    if (ec) {
        std::cerr << "Cannot get file status: " << ec.message() << "\n";
        return false;
    }
    
    fs::perms perms = status.permissions();
    return (perms & fs::owner_read) != fs::no_perms;
}
```

**Issue 2: Unicode filename problems**
```cpp
// Solution: Use proper string conversion
void handle_unicode_filename(const fs::path& path) {
    try {
        // Use wstring on Windows for Unicode support
        std::wcout << L"Processing: " << path.wstring() << L"\n";
        
        // Use UTF-8 string on Unix systems
        std::cout << "Processing: " << path.string() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Unicode handling error: " << e.what() << "\n";
    }
}
```

**Issue 3: Path too long errors**
```cpp
// Solution: Check path lengths and use alternatives
bool is_path_too_long(const fs::path& path) {
#ifdef _WIN32
    return path.string().length() > 260;  // MAX_PATH on Windows
#else
    return path.string().length() > 4096; // Typical Unix limit
#endif
}

fs::path create_short_path_alternative(const fs::path& long_path) {
    if (!is_path_too_long(long_path)) {
        return long_path;
    }
    
    // Create shortened version
    fs::path short_path = long_path.root_path();
    short_path /= "shortened";
    short_path /= long_path.filename();
    
    return short_path;
}
```

#### Debugging Tools and Techniques

**Filesystem Operation Tracer**
```cpp
class FilesystemTracer {
private:
    static bool tracing_enabled;
    
public:
    static void enable_tracing(bool enable = true) {
        tracing_enabled = enable;
    }
    
    template<typename Func, typename... Args>
    static auto trace_operation(const std::string& operation_name, 
                               Func&& func, Args&&... args) {
        if (tracing_enabled) {
            std::cout << "Starting: " << operation_name << "\n";
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                auto result = func(std::forward<Args>(args)...);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                std::cout << "Completed: " << operation_name 
                         << " (" << duration.count() << "ms)\n";
                return result;
            } catch (const std::exception& e) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                std::cout << "Failed: " << operation_name 
                         << " (" << duration.count() << "ms) - " << e.what() << "\n";
                throw;
            }
        } else {
            return func(std::forward<Args>(args)...);
        }
    }
};

// Usage:
// FilesystemTracer::enable_tracing(true);
// auto size = FilesystemTracer::trace_operation("file_size", 
//     [&](){ return fs::file_size(path); });
```

### Migration from Boost to std::filesystem

When you're ready to migrate to C++17's std::filesystem:

#### API Mapping
```cpp
// Boost.Filesystem -> std::filesystem migration guide

// Headers
#include <boost/filesystem.hpp>  // Old
#include <filesystem>            // New (C++17)

// Namespaces
namespace fs = boost::filesystem;  // Old
namespace fs = std::filesystem;    // New

// Most operations are identical:
fs::path p = "example.txt";        // Same
fs::exists(p);                     // Same
fs::file_size(p);                  // Same
fs::create_directory(p);           // Same

// Key differences:
// 1. Error handling - std::filesystem uses std::error_code
// 2. Some function names changed
// 3. Exception types are different
```

#### Migration Strategy
1. **Update includes and namespaces**
2. **Test thoroughly** - behavior may differ slightly
3. **Update error handling** for different exception types
4. **Check platform compatibility** - ensure C++17 support
5. **Performance test** - std::filesystem may have different performance characteristics

### Final Project: Complete Filesystem Utility Suite

Create a comprehensive command-line utility that demonstrates all Boost.Filesystem concepts:

```cpp
class FilesystemUtilitySuite {
public:
    // File operations
    bool copy_with_progress(const fs::path& source, const fs::path& dest);
    bool move_safely(const fs::path& source, const fs::path& dest);
    bool delete_securely(const fs::path& path);
    
    // Directory operations
    void sync_directories(const fs::path& source, const fs::path& dest);
    void backup_directory(const fs::path& source, const fs::path& backup_dir);
    
    // Analysis operations
    void analyze_disk_usage(const fs::path& path);
    void find_duplicate_files(const fs::path& path);
    void clean_temporary_files(const fs::path& path);
    
    // Monitoring operations
    void watch_directory_changes(const fs::path& path);
    void generate_file_report(const fs::path& path, const fs::path& report_file);
};
```

This project should integrate all the concepts you've learned and serve as a portfolio piece demonstrating your mastery of Boost.Filesystem.

## Learning Objectives and Assessment

### Primary Learning Objectives

By the end of this section, you should be able to:

#### 1. Path Manipulation Mastery
- **Construct paths safely** using `fs::path` and the `/` operator
- **Parse path components** (root, directory, filename, extension) accurately
- **Convert between different path formats** (absolute, relative, canonical)
- **Handle cross-platform path differences** transparently
- **Validate paths** for correctness and security

**Self-Assessment:**
- Can you build a path to a user's config directory on both Windows and Unix?
- Can you safely combine user input with base paths without security vulnerabilities?
- Can you extract file extensions reliably for file type detection?

#### 2. Directory Operations Expertise
- **Create and remove directories** with proper error handling
- **Traverse directory trees** efficiently using iterators
- **Filter directory contents** based on various criteria
- **Handle large directories** without memory issues
- **Implement recursive operations** safely

**Self-Assessment:**
- Can you write a function that finds all files with a specific extension in a directory tree?
- Can you implement directory traversal that doesn't follow symbolic links?
- Can you handle permission denied errors gracefully during traversal?

#### 3. File Status and Metadata Management
- **Query file properties** (size, type, permissions, timestamps)
- **Modify file attributes** safely across platforms
- **Handle different file types** (regular files, directories, symlinks)
- **Work with file permissions** in a cross-platform manner
- **Detect and handle special files** appropriately

**Self-Assessment:**
- Can you write a function that checks if a file is writable by the current user?
- Can you detect symbolic links and choose whether to follow them?
- Can you safely get file sizes for very large files (> 4GB)?

#### 4. Robust Error Handling
- **Use both exception and error code patterns** appropriately
- **Provide meaningful error messages** with context
- **Implement retry logic** for transient failures
- **Handle platform-specific errors** correctly
- **Write defensive code** that fails gracefully

**Self-Assessment:**
- Can you write file operations that work reliably across different filesystems?
- Can you handle unicode filenames correctly?
- Can you recover from partial failures in batch operations?

#### 5. Performance Optimization
- **Choose appropriate algorithms** for different directory sizes
- **Minimize filesystem access** through caching and batching
- **Handle memory efficiently** when processing large directory trees
- **Use parallel processing** where appropriate
- **Profile and optimize** filesystem-heavy code

**Self-Assessment:**
- Can you process a directory with millions of files without running out of memory?
- Can you implement efficient duplicate file detection?
- Can you optimize file copying for different storage types?

### Comprehensive Assessment Framework

#### Knowledge Check (30 points)

**Conceptual Questions (10 points):**
1. Explain the difference between `fs::path("/a") / "b"` and `fs::path("/a" + "b")`. (2 points)
2. When would you use `fs::canonical()` vs `fs::absolute()`? (2 points)
3. What are the security implications of following symbolic links? (2 points)
4. How does Boost.Filesystem handle Unicode filenames across platforms? (2 points)
5. Why might `fs::file_size()` throw an exception, and how can you avoid it? (2 points)

**API Knowledge (10 points):**
```cpp
// What's wrong with this code? How would you fix it?
void bad_file_operations() {
    fs::path config_file = "C:\\Users\\John\\config.txt";  // Issue 1
    
    if (fs::exists(config_file)) {
        auto size = fs::file_size(config_file);           // Issue 2
        std::cout << "Config file size: " << size << std::endl;
    }
    
    fs::path backup = config_file + ".backup";            // Issue 3
    fs::copy_file(config_file, backup);                   // Issue 4
    
    for (auto& entry : fs::recursive_directory_iterator("C:\\")) {  // Issue 5
        std::cout << entry.path() << "\n";
    }
}
```

**Cross-Platform Considerations (10 points):**
- Identify 5 differences between Windows and Unix filesystem behavior
- Write code that gets the user's home directory on both platforms
- Explain how to handle case-sensitive vs case-insensitive filesystems

#### Practical Implementation (50 points)

**Project 1: File Duplicate Detector (15 points)**
```cpp
class DuplicateDetector {
public:
    struct DuplicateGroup {
        std::vector<fs::path> files;
        uintmax_t file_size;
        std::string checksum;
    };
    
    // TODO: Implement these methods
    std::vector<DuplicateGroup> find_duplicates(const fs::path& directory);
    std::string calculate_file_hash(const fs::path& file);
    void remove_duplicates(const DuplicateGroup& group, bool keep_newest = true);
};
```

**Grading Criteria:**
- Correctly handles different file sizes (3 points)
- Implements efficient hashing strategy (3 points)
- Provides progress feedback for large operations (3 points)
- Handles errors gracefully (3 points)
- Uses appropriate data structures (3 points)

**Project 2: Safe File Operations Library (20 points)**
```cpp
class SafeFileOperations {
public:
    // Atomic file replacement with backup
    static bool safe_replace_file(const fs::path& source, const fs::path& target);
    
    // Copy with verification
    static bool verified_copy(const fs::path& source, const fs::path& dest);
    
    // Batch operations with rollback
    static bool batch_move_files(const std::vector<std::pair<fs::path, fs::path>>& moves);
    
    // Safe directory removal (checks for important files)
    static bool safe_remove_directory(const fs::path& dir);
};
```

**Grading Criteria:**
- Implements atomic operations correctly (5 points)
- Provides rollback functionality (5 points)
- Validates inputs thoroughly (3 points)
- Handles edge cases (empty files, permissions, etc.) (4 points)
- Thread-safe implementation (3 points)

**Project 3: Directory Monitoring System (15 points)**
```cpp
class DirectoryMonitor {
public:
    enum class ChangeType { Created, Modified, Deleted, Renamed };
    
    struct ChangeEvent {
        fs::path path;
        ChangeType type;
        std::time_t timestamp;
        fs::path old_path;  // For renames
    };
    
    // TODO: Implement directory monitoring
    void start_monitoring(const fs::path& directory);
    void stop_monitoring();
    std::vector<ChangeEvent> get_changes_since(std::time_t since);
};
```

#### Advanced Challenges (20 points)

**Challenge 1: High-Performance File Scanner (10 points)**
- Scan 1 million files in under 10 seconds
- Use minimal memory regardless of directory size
- Support concurrent scanning of multiple directories
- Provide real-time progress updates

**Challenge 2: Cross-Platform Path Converter (10 points)**
```cpp
class PathConverter {
public:
    // Convert Windows paths to Unix format and vice versa
    static fs::path windows_to_unix(const fs::path& windows_path);
    static fs::path unix_to_windows(const fs::path& unix_path);
    
    // Validate path for target platform
    static bool is_valid_for_windows(const fs::path& path);
    static bool is_valid_for_unix(const fs::path& path);
    
    // Suggest corrections for invalid paths
    static fs::path suggest_valid_path(const fs::path& invalid_path, 
                                     const std::string& target_platform);
};
```

### Self-Assessment Tools

#### Debugging Checklist
When your filesystem code isn't working:

- [ ] Did you check if the path exists before operating on it?
- [ ] Are you handling both exceptions and error codes appropriately?
- [ ] Did you test with paths containing spaces and special characters?
- [ ] Are you using the correct path separator for concatenation?
- [ ] Did you consider permissions issues?
- [ ] Are you handling the case where the operation partially succeeds?
- [ ] Did you test on both the target platforms?
- [ ] Are you properly cleaning up resources in error cases?

#### Performance Checklist
For performance-critical filesystem code:

- [ ] Are you minimizing the number of stat() calls?
- [ ] Are you using bulk operations where possible?
- [ ] Did you consider the impact of filesystem type (SSD vs HDD)?
- [ ] Are you caching frequently accessed path information?
- [ ] Did you profile the code with realistic data sizes?
- [ ] Are you using appropriate data structures for your access patterns?
- [ ] Did you consider parallel processing for independent operations?

#### Security Checklist
For production filesystem code:

- [ ] Did you validate all user-provided paths?
- [ ] Are you protected against directory traversal attacks (../ sequences)?
- [ ] Do you handle symbolic links securely?
- [ ] Are you checking permissions before operations?
- [ ] Did you consider race conditions between check and use?
- [ ] Are temporary files created securely?
- [ ] Do you clean up sensitive data from temporary locations?

### Mastery Indicators

You've mastered Boost.Filesystem when you can:

#### Expert Level Indicators:
- Debug filesystem issues across different platforms without access to the target system
- Write filesystem code that performs well on both SSDs and traditional hard drives
- Implement custom filesystem abstractions that hide platform differences
- Optimize filesystem-heavy applications to minimize I/O bottlenecks
- Handle edge cases like full disks, permission changes, and concurrent access gracefully

#### Professional Level Indicators:
- Write filesystem code that passes security audits
- Implement robust backup and synchronization systems
- Create filesystem utilities that handle millions of files efficiently
- Debug and fix filesystem-related race conditions
- Design APIs that make filesystem operations easy and safe for other developers

### Common Mistakes and How to Avoid Them

#### Mistake 1: String Concatenation for Paths
```cpp
// WRONG: This breaks on different platforms
std::string path = base_dir + "/" + filename;

// RIGHT: Use fs::path operator/
fs::path path = fs::path(base_dir) / filename;
```

#### Mistake 2: Not Handling Exceptions
```cpp
// WRONG: Will crash on non-existent files
uintmax_t size = fs::file_size(some_path);

// RIGHT: Handle exceptions or use error codes
try {
    uintmax_t size = fs::file_size(some_path);
} catch (const fs::filesystem_error& e) {
    // Handle error
}
```

#### Mistake 3: Race Conditions
```cpp
// WRONG: File might be deleted between check and use
if (fs::exists(file_path)) {
    auto size = fs::file_size(file_path);  // Might throw!
}

// RIGHT: Handle the exception or use error codes
boost::system::error_code ec;
auto size = fs::file_size(file_path, ec);
if (!ec) {
    // Use size safely
}
```

#### Mistake 4: Inefficient Directory Traversal
```cpp
// WRONG: Recursive function can cause stack overflow
void count_files_recursive(const fs::path& dir, int& count) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (fs::is_directory(entry)) {
            count_files_recursive(entry, count);  // Stack overflow risk!
        } else {
            count++;
        }
    }
}

// RIGHT: Use iterative approach or fs::recursive_directory_iterator
int count_files_iterative(const fs::path& dir) {
    int count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (fs::is_regular_file(entry)) {
            count++;
        }
    }
    return count;
}
```

## Next Steps

Move on to [Concurrency and Multithreading](07_Concurrency_Multithreading.md) to explore Boost's threading and asynchronous capabilities.
