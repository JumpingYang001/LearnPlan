# File System Operations

*Duration: 1 week*

## Overview

This section covers Boost.Filesystem library for portable file system operations, including path manipulation, directory traversal, file operations, and comparisons with std::filesystem.

## Learning Topics

### Path Manipulation
- Path construction and parsing
- Path concatenation and resolution
- Relative and absolute path handling
- Cross-platform path compatibility

### Directory Operations
- Directory creation and removal
- Recursive directory traversal
- Directory listing and filtering
- Temporary directory handling

### File Status and Permissions
- File type detection and properties
- Permission checking and modification
- File size and timestamp operations
- Symbolic link handling

### Comparison with std::filesystem
- API differences and migration strategies
- Performance comparisons
- Compatibility considerations
- When to use Boost vs std

## Code Examples

### Basic Path Operations
```cpp
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

namespace fs = boost::filesystem;

void demonstrate_path_basics() {
    // Create paths
    fs::path current_dir = fs::current_path();
    fs::path file_path = current_dir / "example.txt";
    fs::path absolute_path("/usr/local/bin/program");
    fs::path windows_path("C:\\Program Files\\Application\\app.exe");
    
    std::cout << "Current directory: " << current_dir << "\n";
    std::cout << "File path: " << file_path << "\n";
    std::cout << "Absolute path: " << absolute_path << "\n";
    
    // Path components
    std::cout << "\nPath components for: " << file_path << "\n";
    std::cout << "  Root name: " << file_path.root_name() << "\n";
    std::cout << "  Root directory: " << file_path.root_directory() << "\n";
    std::cout << "  Root path: " << file_path.root_path() << "\n";
    std::cout << "  Relative path: " << file_path.relative_path() << "\n";
    std::cout << "  Parent path: " << file_path.parent_path() << "\n";
    std::cout << "  Filename: " << file_path.filename() << "\n";
    std::cout << "  Stem: " << file_path.stem() << "\n";
    std::cout << "  Extension: " << file_path.extension() << "\n";
}

void demonstrate_path_manipulation() {
    fs::path base_path("/home/user/documents");
    fs::path relative_path("projects/myapp");
    
    // Path concatenation
    fs::path combined = base_path / relative_path;
    std::cout << "Combined path: " << combined << "\n";
    
    // Replace components
    fs::path modified = combined;
    modified.replace_filename("config.ini");
    std::cout << "Modified filename: " << modified << "\n";
    
    modified.replace_extension(".json");
    std::cout << "Changed extension: " << modified << "\n";
    
    // Normalize paths
    fs::path messy_path("/home/user/../user/./documents//file.txt");
    fs::path clean_path = fs::canonical(messy_path, fs::current_path());
    std::cout << "Messy path: " << messy_path << "\n";
    std::cout << "Canonical path: " << clean_path << "\n";
    
    // Relative paths
    fs::path from("/home/user/projects");
    fs::path to("/home/user/documents/file.txt");
    fs::path relative = fs::relative(to, from);
    std::cout << "Relative from " << from << " to " << to << ": " << relative << "\n";
}
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

## Practical Exercises

1. **File Synchronization Tool**
   - Compare directory trees and identify differences
   - Implement bidirectional synchronization
   - Handle conflicts and backup strategies

2. **Log File Analyzer**
   - Scan directories for log files
   - Parse log entries and extract statistics
   - Generate reports on file sizes and dates

3. **Backup System**
   - Create incremental backup functionality
   - Compress and archive files
   - Restore from backup with version control

4. **File Organizer**
   - Organize files by type, date, or size
   - Implement smart folder structures
   - Handle duplicate file detection

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

## Migration from Boost to std::filesystem

### API Differences
```cpp
// Boost.Filesystem
boost::filesystem::path p("/path/to/file");
boost::filesystem::copy_file(src, dst);

// std::filesystem (C++17)
std::filesystem::path p("/path/to/file");
std::filesystem::copy_file(src, dst);
```

### Migration Strategy
1. Update include headers
2. Change namespace references
3. Test thoroughly on target platforms
4. Handle any API differences

## Assessment

- Can perform complex file system operations safely
- Understands cross-platform path handling
- Implements efficient directory traversal algorithms
- Can migrate between Boost and std filesystem APIs

## Next Steps

Move on to [Concurrency and Multithreading](07_Concurrency_Multithreading.md) to explore Boost's threading and asynchronous capabilities.
