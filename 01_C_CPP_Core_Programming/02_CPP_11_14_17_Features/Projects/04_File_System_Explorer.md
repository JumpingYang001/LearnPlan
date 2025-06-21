# File System Explorer Project

## Project Overview

Build a comprehensive file system explorer using C++17's std::filesystem library, incorporating modern C++ features like structured bindings, std::optional, std::variant, and parallel algorithms for high-performance file operations.

## Learning Objectives

- Master std::filesystem API for portable file operations
- Use C++17 structured bindings effectively
- Handle optional and error-prone operations safely
- Implement parallel file processing algorithms
- Create async file operations with std::future
- Build extensible file filtering and search systems

## Project Structure

```
filesystem_explorer_project/
├── src/
│   ├── main.cpp
│   ├── filesystem_explorer.cpp
│   ├── file_processor.cpp
│   ├── search_engine.cpp
│   ├── file_watcher.cpp
│   └── parallel_operations.cpp
├── include/
│   ├── filesystem_explorer.h
│   ├── file_processor.h
│   ├── file_types.h
│   ├── search_engine.h
│   ├── file_watcher.h
│   ├── parallel_operations.h
│   └── utilities.h
├── examples/
│   ├── basic_usage.cpp
│   ├── advanced_search.cpp
│   ├── batch_operations.cpp
│   └── file_monitoring.cpp
├── tests/
│   ├── test_filesystem_ops.cpp
│   ├── test_search_engine.cpp
│   └── test_parallel_ops.cpp
└── CMakeLists.txt
```

## Core Components

### 1. File System Types and Utilities

```cpp
// include/file_types.h
#pragma once
#include <filesystem>
#include <chrono>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <map>
#include <functional>

namespace fs_explorer {

namespace fs = std::filesystem;

// File information structure using structured bindings friendly design
struct FileInfo {
    fs::path path;
    std::optional<uintmax_t> size;
    std::optional<fs::file_time_type> last_write_time;
    std::optional<std::chrono::system_clock::time_point> last_access_time;
    fs::file_type type;
    std::optional<fs::perms> permissions;
    std::optional<std::string> owner;
    std::optional<std::string> group;
    
    // Computed properties
    std::optional<std::string> extension() const {
        if (path.has_extension()) {
            return path.extension().string();
        }
        return std::nullopt;
    }
    
    std::optional<std::string> stem() const {
        if (path.has_stem()) {
            return path.stem().string();
        }
        return std::nullopt;
    }
    
    bool is_hidden() const {
        auto filename = path.filename().string();
        return !filename.empty() && filename[0] == '.';
    }
    
    // Human-readable size
    std::string size_string() const {
        if (!size) return "unknown";
        
        const auto bytes = *size;
        const std::vector<std::string> units = {"B", "KB", "MB", "GB", "TB"};
        
        if (bytes == 0) return "0 B";
        
        double size_val = static_cast<double>(bytes);
        size_t unit_index = 0;
        
        while (size_val >= 1024.0 && unit_index < units.size() - 1) {
            size_val /= 1024.0;
            ++unit_index;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size_val << " " << units[unit_index];
        return oss.str();
    }
    
    // Time formatting
    std::string last_write_time_string() const {
        if (!last_write_time) return "unknown";
        
        auto time_t = std::chrono::system_clock::to_time_t(
            std::chrono::file_clock::to_sys(*last_write_time));
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
};

// Directory statistics
struct DirectoryStats {
    size_t total_files = 0;
    size_t total_directories = 0;
    uintmax_t total_size = 0;
    size_t hidden_items = 0;
    std::map<std::string, size_t> extension_counts;
    std::optional<FileInfo> largest_file;
    std::optional<FileInfo> newest_file;
    std::optional<FileInfo> oldest_file;
    
    void update_with_file(const FileInfo& file_info) {
        if (file_info.type == fs::file_type::regular) {
            ++total_files;
            
            if (file_info.size) {
                total_size += *file_info.size;
                
                if (!largest_file || *file_info.size > *largest_file->size) {
                    largest_file = file_info;
                }
            }
            
            if (file_info.last_write_time) {
                if (!newest_file || *file_info.last_write_time > *newest_file->last_write_time) {
                    newest_file = file_info;
                }
                if (!oldest_file || *file_info.last_write_time < *oldest_file->last_write_time) {
                    oldest_file = file_info;
                }
            }
            
            if (auto ext = file_info.extension()) {
                ++extension_counts[*ext];
            }
            
        } else if (file_info.type == fs::file_type::directory) {
            ++total_directories;
        }
        
        if (file_info.is_hidden()) {
            ++hidden_items;
        }
    }
    
    void print_summary() const {
        std::cout << "Directory Statistics:" << std::endl;
        std::cout << "  Files: " << total_files << std::endl;
        std::cout << "  Directories: " << total_directories << std::endl;
        std::cout << "  Total size: " << format_size(total_size) << std::endl;
        std::cout << "  Hidden items: " << hidden_items << std::endl;
        
        if (largest_file) {
            std::cout << "  Largest file: " << largest_file->path.filename() 
                      << " (" << largest_file->size_string() << ")" << std::endl;
        }
        
        if (!extension_counts.empty()) {
            std::cout << "  Top extensions:" << std::endl;
            
            std::vector<std::pair<std::string, size_t>> sorted_extensions(
                extension_counts.begin(), extension_counts.end());
            
            std::sort(sorted_extensions.begin(), sorted_extensions.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
            
            for (size_t i = 0; i < std::min(size_t(5), sorted_extensions.size()); ++i) {
                const auto& [ext, count] = sorted_extensions[i];
                std::cout << "    " << ext << ": " << count << " files" << std::endl;
            }
        }
    }
    
private:
    std::string format_size(uintmax_t bytes) const {
        const std::vector<std::string> units = {"B", "KB", "MB", "GB", "TB"};
        
        if (bytes == 0) return "0 B";
        
        double size_val = static_cast<double>(bytes);
        size_t unit_index = 0;
        
        while (size_val >= 1024.0 && unit_index < units.size() - 1) {
            size_val /= 1024.0;
            ++unit_index;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size_val << " " << units[unit_index];
        return oss.str();
    }
};

// Search criteria
struct SearchCriteria {
    std::optional<std::string> name_pattern;
    std::optional<std::regex> name_regex;
    std::optional<std::string> extension;
    std::optional<uintmax_t> min_size;
    std::optional<uintmax_t> max_size;
    std::optional<fs::file_time_type> modified_after;
    std::optional<fs::file_time_type> modified_before;
    std::optional<fs::file_type> file_type;
    bool include_hidden = false;
    bool case_sensitive = false;
    size_t max_depth = std::numeric_limits<size_t>::max();
    
    // Content search (for text files)
    std::optional<std::string> content_pattern;
    std::optional<std::regex> content_regex;
    
    bool matches(const FileInfo& file_info) const {
        // Name pattern matching
        if (name_pattern) {
            std::string filename = file_info.path.filename().string();
            if (!case_sensitive) {
                std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
                std::string pattern = *name_pattern;
                std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
                if (filename.find(pattern) == std::string::npos) {
                    return false;
                }
            } else {
                if (filename.find(*name_pattern) == std::string::npos) {
                    return false;
                }
            }
        }
        
        // Regex matching
        if (name_regex) {
            if (!std::regex_search(file_info.path.filename().string(), *name_regex)) {
                return false;
            }
        }
        
        // Extension matching
        if (extension) {
            auto file_ext = file_info.extension();
            if (!file_ext || *file_ext != *extension) {
                return false;
            }
        }
        
        // Size constraints
        if (file_info.size) {
            if (min_size && *file_info.size < *min_size) return false;
            if (max_size && *file_info.size > *max_size) return false;
        }
        
        // Time constraints
        if (file_info.last_write_time) {
            if (modified_after && *file_info.last_write_time < *modified_after) return false;
            if (modified_before && *file_info.last_write_time > *modified_before) return false;
        }
        
        // File type
        if (file_type && file_info.type != *file_type) {
            return false;
        }
        
        // Hidden files
        if (!include_hidden && file_info.is_hidden()) {
            return false;
        }
        
        return true;
    }
};

// Operation result types
template<typename T>
using Result = std::variant<T, std::string>; // Success or error message

template<typename T>
bool is_success(const Result<T>& result) {
    return std::holds_alternative<T>(result);
}

template<typename T>
const T& get_value(const Result<T>& result) {
    return std::get<T>(result);
}

template<typename T>
const std::string& get_error(const Result<T>& result) {
    return std::get<std::string>(result);
}

// File operation types
enum class FileOperation {
    COPY,
    MOVE,
    DELETE,
    CREATE_DIRECTORY,
    SET_PERMISSIONS
};

struct FileOperationRequest {
    FileOperation operation;
    fs::path source_path;
    std::optional<fs::path> destination_path;
    std::optional<fs::perms> permissions;
    bool overwrite_existing = false;
    
    FileOperationRequest(FileOperation op, const fs::path& src)
        : operation(op), source_path(src) {}
    
    FileOperationRequest(FileOperation op, const fs::path& src, const fs::path& dst)
        : operation(op), source_path(src), destination_path(dst) {}
};

using FileOperationResult = Result<fs::path>;

} // namespace fs_explorer
```

### 2. Core File System Explorer

```cpp
// include/filesystem_explorer.h
#pragma once
#include "file_types.h"
#include <future>
#include <atomic>
#include <memory>

namespace fs_explorer {

class FilesystemExplorer {
private:
    fs::path current_directory_;
    std::atomic<bool> cancel_requested_{false};
    
    // Progress reporting
    std::function<void(size_t, size_t)> progress_callback_;
    
    // Error handling
    std::vector<std::string> last_errors_;
    
    // Caching
    mutable std::map<fs::path, DirectoryStats> stats_cache_;
    mutable std::chrono::steady_clock::time_point last_cache_update_;
    
public:
    FilesystemExplorer(const fs::path& initial_directory = fs::current_path());
    
    // Navigation
    bool change_directory(const fs::path& path);
    const fs::path& current_directory() const { return current_directory_; }
    fs::path parent_directory() const;
    std::vector<fs::path> list_drives() const; // Windows specific
    
    // Directory listing with structured bindings support
    std::vector<FileInfo> list_directory(
        const fs::path& directory = {},
        bool recursive = false,
        bool include_hidden = false) const;
    
    // Async directory listing
    std::future<std::vector<FileInfo>> list_directory_async(
        const fs::path& directory = {},
        bool recursive = false,
        bool include_hidden = false) const;
    
    // File information
    std::optional<FileInfo> get_file_info(const fs::path& path) const;
    Result<DirectoryStats> get_directory_stats(const fs::path& directory) const;
    
    // File operations
    FileOperationResult copy_file(const fs::path& source, const fs::path& destination, 
                                 bool overwrite = false);
    FileOperationResult move_file(const fs::path& source, const fs::path& destination);
    FileOperationResult delete_file(const fs::path& path);
    FileOperationResult create_directory(const fs::path& path);
    
    // Batch operations
    std::vector<FileOperationResult> execute_batch_operations(
        const std::vector<FileOperationRequest>& operations);
    
    std::future<std::vector<FileOperationResult>> execute_batch_operations_async(
        const std::vector<FileOperationRequest>& operations);
    
    // Search functionality
    std::vector<FileInfo> search(const SearchCriteria& criteria, 
                                const fs::path& root_directory = {}) const;
    
    std::future<std::vector<FileInfo>> search_async(const SearchCriteria& criteria,
                                                   const fs::path& root_directory = {}) const;
    
    // File content operations
    Result<std::vector<std::string>> read_text_file_lines(const fs::path& path) const;
    Result<std::string> read_text_file(const fs::path& path) const;
    FileOperationResult write_text_file(const fs::path& path, const std::string& content);
    
    // Binary file operations
    Result<std::vector<uint8_t>> read_binary_file(const fs::path& path) const;
    FileOperationResult write_binary_file(const fs::path& path, const std::vector<uint8_t>& data);
    
    // Utility functions
    bool path_exists(const fs::path& path) const;
    bool is_accessible(const fs::path& path) const;
    std::optional<uintmax_t> get_available_space(const fs::path& path) const;
    
    // Progress and cancellation
    void set_progress_callback(std::function<void(size_t, size_t)> callback) {
        progress_callback_ = std::move(callback);
    }
    
    void cancel_operations() { cancel_requested_ = true; }
    void reset_cancellation() { cancel_requested_ = false; }
    bool is_cancelled() const { return cancel_requested_.load(); }
    
    // Error handling
    const std::vector<std::string>& get_last_errors() const { return last_errors_; }
    void clear_errors() { last_errors_.clear(); }
    
    // Cache management
    void clear_cache() const { stats_cache_.clear(); }
    void enable_caching(bool enable) { /* implementation */ }
    
private:
    FileInfo create_file_info(const fs::directory_entry& entry) const;
    void report_progress(size_t current, size_t total) const;
    void add_error(const std::string& error) { last_errors_.push_back(error); }
    
    template<typename Operation>
    auto execute_with_error_handling(Operation&& op) const -> decltype(op());
};

} // namespace fs_explorer
```

```cpp
// src/filesystem_explorer.cpp
#include "filesystem_explorer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <execution>
#include <thread>

namespace fs_explorer {

FilesystemExplorer::FilesystemExplorer(const fs::path& initial_directory)
    : current_directory_(initial_directory) {
    if (!fs::exists(current_directory_) || !fs::is_directory(current_directory_)) {
        current_directory_ = fs::current_path();
    }
}

bool FilesystemExplorer::change_directory(const fs::path& path) {
    try {
        fs::path new_path = path.is_absolute() ? path : current_directory_ / path;
        new_path = fs::canonical(new_path);
        
        if (fs::exists(new_path) && fs::is_directory(new_path)) {
            current_directory_ = new_path;
            return true;
        }
    } catch (const fs::filesystem_error& e) {
        add_error("Failed to change directory: " + std::string(e.what()));
    }
    return false;
}

fs::path FilesystemExplorer::parent_directory() const {
    return current_directory_.parent_path();
}

std::vector<fs::path> FilesystemExplorer::list_drives() const {
    std::vector<fs::path> drives;
    
#ifdef _WIN32
    DWORD drives_mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (drives_mask & (1 << i)) {
            char drive_letter = 'A' + i;
            drives.emplace_back(std::string(1, drive_letter) + ":\\");
        }
    }
#else
    // Unix-like systems typically have everything under root
    drives.emplace_back("/");
#endif
    
    return drives;
}

std::vector<FileInfo> FilesystemExplorer::list_directory(
    const fs::path& directory, bool recursive, bool include_hidden) const {
    
    fs::path target_dir = directory.empty() ? current_directory_ : directory;
    std::vector<FileInfo> files;
    
    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(target_dir)) {
                if (is_cancelled()) break;
                
                auto file_info = create_file_info(entry);
                if (include_hidden || !file_info.is_hidden()) {
                    files.push_back(std::move(file_info));
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(target_dir)) {
                if (is_cancelled()) break;
                
                auto file_info = create_file_info(entry);
                if (include_hidden || !file_info.is_hidden()) {
                    files.push_back(std::move(file_info));
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        add_error("Failed to list directory: " + std::string(e.what()));
    }
    
    return files;
}

std::future<std::vector<FileInfo>> FilesystemExplorer::list_directory_async(
    const fs::path& directory, bool recursive, bool include_hidden) const {
    
    return std::async(std::launch::async, [this, directory, recursive, include_hidden]() {
        return list_directory(directory, recursive, include_hidden);
    });
}

std::optional<FileInfo> FilesystemExplorer::get_file_info(const fs::path& path) const {
    try {
        if (fs::exists(path)) {
            fs::directory_entry entry(path);
            return create_file_info(entry);
        }
    } catch (const fs::filesystem_error& e) {
        add_error("Failed to get file info: " + std::string(e.what()));
    }
    
    return std::nullopt;
}

Result<DirectoryStats> FilesystemExplorer::get_directory_stats(const fs::path& directory) const {
    try {
        DirectoryStats stats;
        
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (is_cancelled()) {
                return "Operation cancelled";
            }
            
            auto file_info = create_file_info(entry);
            stats.update_with_file(file_info);
        }
        
        return stats;
    } catch (const fs::filesystem_error& e) {
        return "Failed to calculate directory stats: " + std::string(e.what());
    }
}

FileOperationResult FilesystemExplorer::copy_file(
    const fs::path& source, const fs::path& destination, bool overwrite) {
    
    try {
        fs::copy_options options = fs::copy_options::none;
        if (overwrite) {
            options |= fs::copy_options::overwrite_existing;
        }
        
        if (fs::is_directory(source)) {
            options |= fs::copy_options::recursive;
        }
        
        fs::copy(source, destination, options);
        return destination;
    } catch (const fs::filesystem_error& e) {
        return "Copy failed: " + std::string(e.what());
    }
}

FileOperationResult FilesystemExplorer::move_file(
    const fs::path& source, const fs::path& destination) {
    
    try {
        fs::rename(source, destination);
        return destination;
    } catch (const fs::filesystem_error& e) {
        return "Move failed: " + std::string(e.what());
    }
}

FileOperationResult FilesystemExplorer::delete_file(const fs::path& path) {
    try {
        uintmax_t removed = fs::remove_all(path);
        if (removed > 0) {
            return path;
        } else {
            return "No files were deleted";
        }
    } catch (const fs::filesystem_error& e) {
        return "Delete failed: " + std::string(e.what());
    }
}

FileOperationResult FilesystemExplorer::create_directory(const fs::path& path) {
    try {
        if (fs::create_directories(path)) {
            return path;
        } else {
            return "Directory already exists";
        }
    } catch (const fs::filesystem_error& e) {
        return "Create directory failed: " + std::string(e.what());
    }
}

std::vector<FileOperationResult> FilesystemExplorer::execute_batch_operations(
    const std::vector<FileOperationRequest>& operations) {
    
    std::vector<FileOperationResult> results;
    results.reserve(operations.size());
    
    for (const auto& [operation, source_path, destination_path, permissions, overwrite] : operations) {
        if (is_cancelled()) {
            results.emplace_back("Operation cancelled");
            continue;
        }
        
        switch (operation) {
        case FileOperation::COPY:
            if (destination_path) {
                results.push_back(copy_file(source_path, *destination_path, overwrite));
            } else {
                results.emplace_back("Copy operation requires destination path");
            }
            break;
            
        case FileOperation::MOVE:
            if (destination_path) {
                results.push_back(move_file(source_path, *destination_path));
            } else {
                results.emplace_back("Move operation requires destination path");
            }
            break;
            
        case FileOperation::DELETE:
            results.push_back(delete_file(source_path));
            break;
            
        case FileOperation::CREATE_DIRECTORY:
            results.push_back(create_directory(source_path));
            break;
            
        case FileOperation::SET_PERMISSIONS:
            try {
                if (permissions) {
                    fs::permissions(source_path, *permissions);
                    results.emplace_back(source_path);
                } else {
                    results.emplace_back("Set permissions requires permissions value");
                }
            } catch (const fs::filesystem_error& e) {
                results.emplace_back("Set permissions failed: " + std::string(e.what()));
            }
            break;
        }
        
        report_progress(results.size(), operations.size());
    }
    
    return results;
}

std::future<std::vector<FileOperationResult>> FilesystemExplorer::execute_batch_operations_async(
    const std::vector<FileOperationRequest>& operations) {
    
    return std::async(std::launch::async, [this, operations]() {
        return execute_batch_operations(operations);
    });
}

std::vector<FileInfo> FilesystemExplorer::search(
    const SearchCriteria& criteria, const fs::path& root_directory) const {
    
    fs::path search_root = root_directory.empty() ? current_directory_ : root_directory;
    std::vector<FileInfo> matches;
    
    try {
        size_t current_depth = 0;
        
        for (const auto& entry : fs::recursive_directory_iterator(search_root)) {
            if (is_cancelled()) break;
            
            // Check depth limit
            size_t depth = std::distance(fs::begin(fs::relative(entry.path(), search_root)),
                                       fs::end(fs::relative(entry.path(), search_root)));
            if (depth > criteria.max_depth) {
                continue;
            }
            
            auto file_info = create_file_info(entry);
            
            if (criteria.matches(file_info)) {
                // Additional content search for text files
                if (criteria.content_pattern || criteria.content_regex) {
                    if (file_info.type == fs::file_type::regular) {
                        auto content_result = read_text_file(file_info.path);
                        if (is_success(content_result)) {
                            const auto& content = get_value(content_result);
                            
                            bool content_matches = true;
                            if (criteria.content_pattern) {
                                std::string search_content = content;
                                std::string pattern = *criteria.content_pattern;
                                
                                if (!criteria.case_sensitive) {
                                    std::transform(search_content.begin(), search_content.end(),
                                                 search_content.begin(), ::tolower);
                                    std::transform(pattern.begin(), pattern.end(),
                                                 pattern.begin(), ::tolower);
                                }
                                
                                content_matches = search_content.find(pattern) != std::string::npos;
                            }
                            
                            if (content_matches && criteria.content_regex) {
                                content_matches = std::regex_search(content, *criteria.content_regex);
                            }
                            
                            if (content_matches) {
                                matches.push_back(std::move(file_info));
                            }
                        }
                    }
                } else {
                    matches.push_back(std::move(file_info));
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        add_error("Search failed: " + std::string(e.what()));
    }
    
    return matches;
}

std::future<std::vector<FileInfo>> FilesystemExplorer::search_async(
    const SearchCriteria& criteria, const fs::path& root_directory) const {
    
    return std::async(std::launch::async, [this, criteria, root_directory]() {
        return search(criteria, root_directory);
    });
}

Result<std::vector<std::string>> FilesystemExplorer::read_text_file_lines(const fs::path& path) const {
    try {
        std::ifstream file(path);
        if (!file) {
            return "Failed to open file for reading";
        }
        
        std::vector<std::string> lines;
        std::string line;
        
        while (std::getline(file, line)) {
            lines.push_back(std::move(line));
        }
        
        return lines;
    } catch (const std::exception& e) {
        return "Failed to read file: " + std::string(e.what());
    }
}

Result<std::string> FilesystemExplorer::read_text_file(const fs::path& path) const {
    try {
        std::ifstream file(path);
        if (!file) {
            return "Failed to open file for reading";
        }
        
        std::ostringstream content;
        content << file.rdbuf();
        
        return content.str();
    } catch (const std::exception& e) {
        return "Failed to read file: " + std::string(e.what());
    }
}

FileOperationResult FilesystemExplorer::write_text_file(
    const fs::path& path, const std::string& content) {
    
    try {
        std::ofstream file(path);
        if (!file) {
            return "Failed to open file for writing";
        }
        
        file << content;
        
        if (file.good()) {
            return path;
        } else {
            return "Failed to write file content";
        }
    } catch (const std::exception& e) {
        return "Failed to write file: " + std::string(e.what());
    }
}

FileInfo FilesystemExplorer::create_file_info(const fs::directory_entry& entry) const {
    FileInfo info;
    info.path = entry.path();
    
    try {
        info.type = entry.status().type();
        
        if (entry.is_regular_file()) {
            std::error_code ec;
            info.size = entry.file_size(ec);
            if (ec) info.size.reset();
        }
        
        std::error_code ec;
        info.last_write_time = entry.last_write_time(ec);
        if (ec) info.last_write_time.reset();
        
        info.permissions = entry.status().permissions();
        
    } catch (const fs::filesystem_error&) {
        // Some fields may not be accessible - leave them as nullopt
    }
    
    return info;
}

void FilesystemExplorer::report_progress(size_t current, size_t total) const {
    if (progress_callback_) {
        progress_callback_(current, total);
    }
}

} // namespace fs_explorer
```

### 3. Advanced Search Engine

```cpp
// include/search_engine.h
#pragma once
#include "file_types.h"
#include <regex>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace fs_explorer {

class SearchEngine {
public:
    struct SearchOptions {
        bool parallel_search = true;
        size_t max_threads = std::thread::hardware_concurrency();
        bool index_content = false;
        size_t max_file_size_for_content = 10 * 1024 * 1024; // 10MB
        std::vector<std::string> text_extensions = {".txt", ".cpp", ".h", ".hpp", ".py", ".js", ".html", ".xml", ".json"};
        bool follow_symlinks = false;
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(); // No timeout
    };
    
private:
    SearchOptions options_;
    std::atomic<bool> search_active_{false};
    std::atomic<bool> cancel_requested_{false};
    
    // Thread pool for parallel search
    std::vector<std::thread> worker_threads_;
    std::mutex work_queue_mutex_;
    std::condition_variable work_available_;
    std::queue<fs::path> work_queue_;
    
    // Results collection
    std::mutex results_mutex_;
    std::vector<FileInfo> search_results_;
    
    // Progress tracking
    std::atomic<size_t> processed_count_{0};
    std::atomic<size_t> total_count_{0};
    std::function<void(size_t, size_t)> progress_callback_;
    
public:
    explicit SearchEngine(const SearchOptions& options = SearchOptions{});
    ~SearchEngine();
    
    // Main search methods
    std::vector<FileInfo> search(const SearchCriteria& criteria, const fs::path& root_path);
    std::future<std::vector<FileInfo>> search_async(const SearchCriteria& criteria, const fs::path& root_path);
    
    // Specialized search methods
    std::vector<FileInfo> find_duplicates(const fs::path& root_path, bool compare_content = false);
    std::vector<FileInfo> find_large_files(const fs::path& root_path, uintmax_t min_size);
    std::vector<FileInfo> find_old_files(const fs::path& root_path, std::chrono::system_clock::time_point before);
    std::vector<FileInfo> find_empty_files(const fs::path& root_path);
    std::vector<FileInfo> find_empty_directories(const fs::path& root_path);
    
    // Content search
    std::vector<std::pair<FileInfo, std::vector<size_t>>> search_file_content(
        const fs::path& root_path, const std::string& pattern, bool regex = false);
    
    // Index-based search (for improved performance)
    bool build_index(const fs::path& root_path);
    std::vector<FileInfo> search_indexed(const SearchCriteria& criteria);
    
    // Configuration
    void set_options(const SearchOptions& options) { options_ = options; }
    const SearchOptions& get_options() const { return options_; }
    
    void set_progress_callback(std::function<void(size_t, size_t)> callback) {
        progress_callback_ = std::move(callback);
    }
    
    // Control
    void cancel_search() { cancel_requested_ = true; }
    bool is_search_active() const { return search_active_.load(); }
    
private:
    void worker_thread_function(const SearchCriteria& criteria);
    void collect_paths_to_search(const fs::path& root_path, std::vector<fs::path>& paths);
    bool matches_criteria(const FileInfo& file_info, const SearchCriteria& criteria);
    bool is_text_file(const fs::path& path) const;
    std::string calculate_file_hash(const fs::path& path) const;
    
    // Content search helpers
    std::vector<size_t> find_pattern_in_file(const fs::path& path, const std::string& pattern, bool regex = false);
    
    // Index data structures
    struct IndexEntry {
        FileInfo file_info;
        std::optional<std::string> content_hash;
        std::vector<std::string> content_words; // For content indexing
    };
    
    std::map<fs::path, IndexEntry> file_index_;
    std::map<std::string, std::vector<fs::path>> word_index_; // Word -> Files containing it
    std::mutex index_mutex_;
    bool index_built_ = false;
};

} // namespace fs_explorer
```

### 4. Parallel Operations

```cpp
// include/parallel_operations.h
#pragma once
#include "file_types.h"
#include <execution>
#include <algorithm>
#include <thread>
#include <future>

namespace fs_explorer {

class ParallelOperations {
public:
    struct OperationStats {
        size_t total_operations = 0;
        size_t successful_operations = 0;
        size_t failed_operations = 0;
        std::chrono::milliseconds total_time{0};
        std::vector<std::string> errors;
    };
    
    // Parallel file operations
    static std::future<OperationStats> copy_files_parallel(
        const std::vector<std::pair<fs::path, fs::path>>& copy_pairs,
        bool overwrite = false,
        size_t max_threads = std::thread::hardware_concurrency());
    
    static std::future<OperationStats> delete_files_parallel(
        const std::vector<fs::path>& files,
        size_t max_threads = std::thread::hardware_concurrency());
    
    // Parallel directory analysis
    static std::future<std::vector<DirectoryStats>> analyze_directories_parallel(
        const std::vector<fs::path>& directories,
        size_t max_threads = std::thread::hardware_concurrency());
    
    // Parallel file processing
    template<typename Processor>
    static auto process_files_parallel(
        const std::vector<fs::path>& files,
        Processor&& processor,
        size_t max_threads = std::thread::hardware_concurrency()) -> std::future<std::vector<decltype(processor(fs::path{}))>> {
        
        using ResultType = decltype(processor(fs::path{}));
        
        return std::async(std::launch::async, [files, processor, max_threads]() {
            std::vector<ResultType> results(files.size());
            
            // Use parallel algorithm if available
            #ifdef __cpp_lib_parallel_algorithm
            std::transform(std::execution::par_unseq,
                          files.begin(), files.end(),
                          results.begin(),
                          processor);
            #else
            // Fallback to manual threading
            const size_t num_threads = std::min(max_threads, files.size());
            const size_t chunk_size = files.size() / num_threads;
            
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);
            
            for (size_t t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = (t == num_threads - 1) ? files.size() : (t + 1) * chunk_size;
                
                futures.push_back(std::async(std::launch::async, [&files, &results, &processor, start, end]() {
                    for (size_t i = start; i < end; ++i) {
                        results[i] = processor(files[i]);
                    }
                }));
            }
            
            // Wait for all threads to complete
            for (auto& future : futures) {
                future.wait();
            }
            #endif
            
            return results;
        });
    }
    
    // Parallel content search
    static std::future<std::vector<std::pair<fs::path, std::vector<size_t>>>> search_content_parallel(
        const std::vector<fs::path>& files,
        const std::string& pattern,
        bool regex = false,
        size_t max_threads = std::thread::hardware_concurrency());
    
    // Parallel hash calculation
    static std::future<std::map<fs::path, std::string>> calculate_hashes_parallel(
        const std::vector<fs::path>& files,
        size_t max_threads = std::thread::hardware_concurrency());
    
private:
    template<typename Container, typename Function>
    static auto parallel_transform_impl(Container&& container, Function&& func, size_t max_threads)
        -> std::vector<decltype(func(*container.begin()))>;
};

// Utility class for managing parallel work
template<typename WorkItem, typename Result>
class ParallelProcessor {
private:
    std::queue<WorkItem> work_queue_;
    std::vector<Result> results_;
    std::mutex work_mutex_;
    std::mutex results_mutex_;
    std::condition_variable work_available_;
    std::atomic<bool> finished_{false};
    std::atomic<size_t> active_workers_{0};
    
    std::function<Result(const WorkItem&)> processor_;
    std::function<void(size_t, size_t)> progress_callback_;
    
public:
    template<typename Processor>
    ParallelProcessor(Processor&& proc) : processor_(std::forward<Processor>(proc)) {}
    
    void set_progress_callback(std::function<void(size_t, size_t)> callback) {
        progress_callback_ = std::move(callback);
    }
    
    std::future<std::vector<Result>> process_all(
        const std::vector<WorkItem>& work_items,
        size_t num_threads = std::thread::hardware_concurrency()) {
        
        return std::async(std::launch::async, [this, work_items, num_threads]() {
            // Initialize work queue
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                for (const auto& item : work_items) {
                    work_queue_.push(item);
                }
            }
            
            results_.reserve(work_items.size());
            finished_ = false;
            
            // Start worker threads
            std::vector<std::thread> workers;
            workers.reserve(num_threads);
            
            for (size_t i = 0; i < num_threads; ++i) {
                workers.emplace_back([this, total_items = work_items.size()]() {
                    worker_function(total_items);
                });
            }
            
            // Wait for all workers to finish
            for (auto& worker : workers) {
                worker.join();
            }
            
            return std::move(results_);
        });
    }
    
private:
    void worker_function(size_t total_items) {
        ++active_workers_;
        
        while (true) {
            WorkItem work_item;
            bool has_work = false;
            
            // Get work item
            {
                std::unique_lock<std::mutex> lock(work_mutex_);
                work_available_.wait(lock, [this]() {
                    return !work_queue_.empty() || finished_;
                });
                
                if (!work_queue_.empty()) {
                    work_item = work_queue_.front();
                    work_queue_.pop();
                    has_work = true;
                } else if (finished_) {
                    break;
                }
            }
            
            if (has_work) {
                // Process work item
                Result result = processor_(work_item);
                
                // Store result
                {
                    std::lock_guard<std::mutex> lock(results_mutex_);
                    results_.push_back(std::move(result));
                    
                    if (progress_callback_) {
                        progress_callback_(results_.size(), total_items);
                    }
                }
            }
            
            // Check if we're done
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                if (work_queue_.empty()) {
                    finished_ = true;
                    work_available_.notify_all();
                    break;
                }
            }
        }
        
        --active_workers_;
    }
};

} // namespace fs_explorer
```

### 5. Usage Examples

```cpp
// examples/basic_usage.cpp
#include <iostream>
#include "filesystem_explorer.h"
#include "search_engine.h"

using namespace fs_explorer;

void demonstrate_basic_operations() {
    std::cout << "\n=== Basic Filesystem Operations ===" << std::endl;
    
    FilesystemExplorer explorer;
    
    std::cout << "Current directory: " << explorer.current_directory() << std::endl;
    
    // List current directory
    auto files = explorer.list_directory();
    std::cout << "\nDirectory contents (" << files.size() << " items):" << std::endl;
    
    for (const auto& file : files) {
        const auto& [path, size, last_write, last_access, type, permissions] = 
            std::tie(file.path, file.size, file.last_write_time, file.last_access_time, file.type, file.permissions);
        
        std::cout << "  " << file.path.filename().string();
        
        if (type == fs::file_type::directory) {
            std::cout << " [DIR]";
        } else if (type == fs::file_type::regular) {
            std::cout << " [FILE";
            if (size) {
                std::cout << ", " << file.size_string();
            }
            std::cout << "]";
        }
        
        if (last_write) {
            std::cout << " - " << file.last_write_time_string();
        }
        
        std::cout << std::endl;
    }
}

void demonstrate_file_search() {
    std::cout << "\n=== File Search Operations ===" << std::endl;
    
    FilesystemExplorer explorer;
    SearchEngine search_engine;
    
    // Search for C++ files
    SearchCriteria cpp_criteria;
    cpp_criteria.name_pattern = ".cpp";
    cpp_criteria.file_type = fs::file_type::regular;
    cpp_criteria.include_hidden = false;
    
    auto cpp_files = explorer.search(cpp_criteria);
    std::cout << "Found " << cpp_files.size() << " C++ files:" << std::endl;
    
    for (const auto& file : cpp_files) {
        std::cout << "  " << file.path << " (" << file.size_string() << ")" << std::endl;
    }
    
    // Search for large files (>1MB)
    SearchCriteria large_file_criteria;
    large_file_criteria.min_size = 1024 * 1024; // 1MB
    large_file_criteria.file_type = fs::file_type::regular;
    
    auto large_files = search_engine.find_large_files(explorer.current_directory(), 1024 * 1024);
    std::cout << "\nFound " << large_files.size() << " large files (>1MB):" << std::endl;
    
    for (const auto& file : large_files) {
        std::cout << "  " << file.path.filename() << " - " << file.size_string() << std::endl;
    }
}

void demonstrate_directory_stats() {
    std::cout << "\n=== Directory Statistics ===" << std::endl;
    
    FilesystemExplorer explorer;
    
    auto stats_result = explorer.get_directory_stats(explorer.current_directory());
    
    if (is_success(stats_result)) {
        const auto& stats = get_value(stats_result);
        stats.print_summary();
    } else {
        std::cout << "Error getting directory stats: " << get_error(stats_result) << std::endl;
    }
}

void demonstrate_async_operations() {
    std::cout << "\n=== Asynchronous Operations ===" << std::endl;
    
    FilesystemExplorer explorer;
    
    // Set up progress callback
    explorer.set_progress_callback([](size_t current, size_t total) {
        if (total > 0) {
            int percentage = static_cast<int>((current * 100) / total);
            std::cout << "\rProgress: " << percentage << "% (" << current << "/" << total << ")" << std::flush;
        }
    });
    
    // Start async directory listing
    auto future = explorer.list_directory_async(explorer.current_directory(), true, false);
    
    std::cout << "Listing directory recursively (async)..." << std::endl;
    
    // Do other work while waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Get results
    auto files = future.get();
    std::cout << "\nAsync operation completed. Found " << files.size() << " items." << std::endl;
    
    // Count by type using structured bindings
    size_t file_count = 0, dir_count = 0, other_count = 0;
    
    for (const auto& [path, size, last_write, last_access, type, permissions] : files) {
        switch (type) {
        case fs::file_type::regular:
            ++file_count;
            break;
        case fs::file_type::directory:
            ++dir_count;
            break;
        default:
            ++other_count;
            break;
        }
    }
    
    std::cout << "Summary: " << file_count << " files, " << dir_count 
              << " directories, " << other_count << " other items" << std::endl;
}

void demonstrate_batch_operations() {
    std::cout << "\n=== Batch Operations ===" << std::endl;
    
    FilesystemExplorer explorer;
    
    // Create test directory structure
    std::vector<FileOperationRequest> setup_operations = {
        {FileOperation::CREATE_DIRECTORY, "test_batch"},
        {FileOperation::CREATE_DIRECTORY, "test_batch/subdir1"},
        {FileOperation::CREATE_DIRECTORY, "test_batch/subdir2"}
    };
    
    auto setup_results = explorer.execute_batch_operations(setup_operations);
    
    std::cout << "Setup operations:" << std::endl;
    for (size_t i = 0; i < setup_results.size(); ++i) {
        const auto& result = setup_results[i];
        std::cout << "  Operation " << i + 1 << ": ";
        
        if (is_success(result)) {
            std::cout << "Success - " << get_value(result) << std::endl;
        } else {
            std::cout << "Failed - " << get_error(result) << std::endl;
        }
    }
    
    // Create some test files
    explorer.write_text_file("test_batch/file1.txt", "Content of file 1");
    explorer.write_text_file("test_batch/file2.txt", "Content of file 2");
    explorer.write_text_file("test_batch/subdir1/nested_file.txt", "Nested file content");
    
    // Cleanup operations
    std::vector<FileOperationRequest> cleanup_operations = {
        {FileOperation::DELETE, "test_batch"}
    };
    
    auto cleanup_future = explorer.execute_batch_operations_async(cleanup_operations);
    
    std::cout << "Cleanup operations running asynchronously..." << std::endl;
    auto cleanup_results = cleanup_future.get();
    
    for (const auto& result : cleanup_results) {
        if (is_success(result)) {
            std::cout << "Cleanup successful" << std::endl;
        } else {
            std::cout << "Cleanup failed: " << get_error(result) << std::endl;
        }
    }
}

int main() {
    try {
        demonstrate_basic_operations();
        demonstrate_file_search();
        demonstrate_directory_stats();
        demonstrate_async_operations();
        demonstrate_batch_operations();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(FilesystemExplorer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/filesystem_explorer.cpp
    src/file_processor.cpp
    src/search_engine.cpp
    src/parallel_operations.cpp
)

# Create library
add_library(filesystem_lib ${SOURCES})
target_link_libraries(filesystem_lib Threads::Threads)

# Platform-specific linking
if(WIN32)
    target_link_libraries(filesystem_lib kernel32)
endif()

# Examples
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage filesystem_lib)

add_executable(advanced_search examples/advanced_search.cpp)
target_link_libraries(advanced_search filesystem_lib)

add_executable(batch_operations examples/batch_operations.cpp)
target_link_libraries(batch_operations filesystem_lib)

# Tests
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(filesystem_tests
        tests/test_filesystem_ops.cpp
        tests/test_search_engine.cpp
        tests/test_parallel_ops.cpp
    )
    target_link_libraries(filesystem_tests GTest::gtest_main filesystem_lib)
    
    enable_testing()
    add_test(NAME Filesystem_Tests COMMAND filesystem_tests)
endif()

# Compiler options
if(MSVC)
    target_compile_options(filesystem_lib PRIVATE /W4)
    target_compile_definitions(filesystem_lib PRIVATE _WIN32_WINNT=0x0601)
else()
    target_compile_options(filesystem_lib PRIVATE -Wall -Wextra -Wpedantic)
    
    # Link filesystem library on older compilers
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        target_link_libraries(filesystem_lib stdc++fs)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        target_link_libraries(filesystem_lib c++fs)
    endif()
endif()
```

## Expected Learning Outcomes

After completing this project, you should master:

1. **std::filesystem API**
   - Path manipulation and navigation
   - File and directory operations
   - File status and metadata access
   - Cross-platform file system operations

2. **C++17 Language Features**
   - Structured bindings for clean data access
   - std::optional for safe value handling
   - std::variant for error handling
   - Parallel algorithms for performance

3. **Asynchronous Programming**
   - std::future and std::async
   - Thread-safe data structures
   - Progress reporting and cancellation

4. **Modern C++ Design Patterns**
   - RAII for resource management
   - Template metaprogramming
   - Error handling strategies

## Extensions and Improvements

1. **Advanced Features**
   - File system watching/monitoring
   - File compression and archiving
   - Network file system support
   - Symbolic link handling

2. **Performance Optimizations**
   - Memory-mapped file I/O
   - Parallel directory traversal
   - Index-based search caching
   - Streaming large file operations

3. **User Interface**
   - Command-line interface
   - GUI integration
   - REST API for remote access
   - Plugin system for extensibility

This project demonstrates comprehensive usage of C++17 filesystem features while building a practical, high-performance file system explorer.
