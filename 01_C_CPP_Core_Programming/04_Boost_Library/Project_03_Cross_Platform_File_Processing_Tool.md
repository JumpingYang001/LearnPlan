# Project 3: Cross-Platform File Processing Tool

*Estimated Duration: 2-3 weeks*
*Difficulty: Intermediate*

## Project Overview

Develop a comprehensive cross-platform file processing utility using Boost.Filesystem, Boost.Program_options, and other Boost libraries. This tool will demonstrate advanced file system operations, command-line argument parsing, and cross-platform compatibility considerations.

## Learning Objectives

- Master Boost.Filesystem for file system operations
- Understand cross-platform file handling differences
- Implement robust command-line interfaces with Boost.Program_options
- Handle various file formats and encoding issues
- Design efficient file processing pipelines
- Implement progress tracking and error recovery

## Project Requirements

### Core Features

1. **File Discovery and Filtering**
   - Recursive directory traversal
   - Pattern-based file filtering (wildcards, regex)
   - File type detection and classification
   - Size and date-based filtering criteria

2. **File Operations**
   - Copy, move, and rename operations
   - Batch processing capabilities
   - Permission and ownership handling
   - Symbolic link support

3. **Content Processing**
   - Text file encoding detection and conversion
   - Search and replace operations
   - Line ending normalization
   - Whitespace cleanup

### Advanced Features

4. **Performance Optimization**
   - Multi-threaded processing
   - Memory-mapped file access
   - Progress tracking and cancellation
   - Bandwidth throttling for network operations

5. **Data Integrity**
   - Checksum verification (MD5, SHA-256)
   - Backup and rollback capabilities
   - Atomic operations
   - Duplicate file detection

6. **Integration Features**
   - Plugin architecture for custom processors
   - Configuration file support
   - Logging and audit trails
   - Scriptable automation interface

## Implementation Guide

### Step 1: Project Structure and Dependencies

```cpp
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_pool.hpp>
#include <boost/format.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/crc.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <queue>

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace ba = boost::algorithm;
using boost::format;

// Forward declarations
class FileProcessor;
class ProgressTracker;
class FileOperation;
class ContentProcessor;
```

### Step 2: Configuration and Command Line Interface

```cpp
struct ProcessingConfig {
    std::vector<std::string> input_paths;
    std::string output_path;
    std::string pattern;
    bool recursive = true;
    bool preserve_structure = true;
    int thread_count = std::thread::hardware_concurrency();
    size_t max_file_size = SIZE_MAX;
    std::string encoding_from = "auto";
    std::string encoding_to = "utf-8";
    bool normalize_line_endings = false;
    bool remove_trailing_whitespace = false;
    std::string backup_suffix = ".bak";
    bool create_backups = false;
    bool verify_checksums = false;
    std::string log_file;
    int verbosity = 1;
    bool dry_run = false;
    
    // Search and replace options
    std::string search_pattern;
    std::string replace_pattern;
    bool use_regex = false;
    bool case_sensitive = true;
    
    // File filtering options
    std::vector<std::string> include_extensions;
    std::vector<std::string> exclude_extensions;
    std::string min_size;
    std::string max_size;
    std::string newer_than;
    std::string older_than;
};

class CommandLineParser {
public:
    static ProcessingConfig parse(int argc, char* argv[]) {
        ProcessingConfig config;
        
        po::options_description desc("File Processing Tool Options");
        desc.add_options()
            ("help,h", "Show help message")
            ("input,i", po::value<std::vector<std::string>>(&config.input_paths)->multitoken()->required(),
             "Input files or directories")
            ("output,o", po::value<std::string>(&config.output_path),
             "Output directory")
            ("pattern,p", po::value<std::string>(&config.pattern)->default_value("*"),
             "File pattern to match")
            ("recursive,r", po::bool_switch(&config.recursive),
             "Process directories recursively")
            ("threads,t", po::value<int>(&config.thread_count),
             "Number of processing threads")
            ("max-size", po::value<std::string>(&config.max_size),
             "Maximum file size to process (e.g., 10MB)")
            ("encoding-from", po::value<std::string>(&config.encoding_from),
             "Source encoding (auto-detect if not specified)")
            ("encoding-to", po::value<std::string>(&config.encoding_to),
             "Target encoding")
            ("normalize-lines", po::bool_switch(&config.normalize_line_endings),
             "Normalize line endings to system default")
            ("trim-whitespace", po::bool_switch(&config.remove_trailing_whitespace),
             "Remove trailing whitespace")
            ("backup", po::bool_switch(&config.create_backups),
             "Create backup files")
            ("backup-suffix", po::value<std::string>(&config.backup_suffix),
             "Backup file suffix")
            ("verify", po::bool_switch(&config.verify_checksums),
             "Verify file checksums")
            ("log", po::value<std::string>(&config.log_file),
             "Log file path")
            ("verbose,v", po::value<int>(&config.verbosity),
             "Verbosity level (0-3)")
            ("dry-run,n", po::bool_switch(&config.dry_run),
             "Show what would be done without making changes")
            ("search", po::value<std::string>(&config.search_pattern),
             "Search pattern")
            ("replace", po::value<std::string>(&config.replace_pattern),
             "Replacement text")
            ("regex", po::bool_switch(&config.use_regex),
             "Use regular expressions for search/replace")
            ("ignore-case", po::bool_switch()->default_value(false),
             "Case-insensitive search")
            ("include-ext", po::value<std::vector<std::string>>(&config.include_extensions)->multitoken(),
             "Include only these file extensions")
            ("exclude-ext", po::value<std::vector<std::string>>(&config.exclude_extensions)->multitoken(),
             "Exclude these file extensions")
            ("newer-than", po::value<std::string>(&config.newer_than),
             "Process files newer than date (YYYY-MM-DD)")
            ("older-than", po::value<std::string>(&config.older_than),
             "Process files older than date (YYYY-MM-DD)");
        
        po::variables_map vm;
        
        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            
            if (vm.count("help")) {
                std::cout << desc << std::endl;
                std::cout << "\nExamples:\n";
                std::cout << "  fileproc -i ./src -o ./processed -p \"*.cpp\" --normalize-lines\n";
                std::cout << "  fileproc -i ./docs -r --search \"TODO\" --replace \"FIXME\" --regex\n";
                std::cout << "  fileproc -i ./logs --older-than 2023-01-01 --verify\n";
                exit(0);
            }
            
            po::notify(vm);
            
            // Handle case sensitivity
            if (vm["ignore-case"].as<bool>()) {
                config.case_sensitive = false;
            }
            
            // Validate configuration
            validate_config(config);
            
        } catch (const po::error& e) {
            std::cerr << "Command line error: " << e.what() << std::endl;
            std::cerr << desc << std::endl;
            exit(1);
        }
        
        return config;
    }
    
private:
    static void validate_config(const ProcessingConfig& config) {
        // Validate input paths
        for (const auto& path : config.input_paths) {
            if (!fs::exists(path)) {
                throw std::runtime_error("Input path does not exist: " + path);
            }
        }
        
        // Validate output path if specified
        if (!config.output_path.empty()) {
            fs::path output_dir = fs::path(config.output_path).parent_path();
            if (!output_dir.empty() && !fs::exists(output_dir)) {
                throw std::runtime_error("Output directory does not exist: " + output_dir.string());
            }
        }
        
        // Validate thread count
        if (config.thread_count < 1 || config.thread_count > 100) {
            throw std::runtime_error("Thread count must be between 1 and 100");
        }
    }
};
```

### Step 3: File Discovery and Filtering

```cpp
class FileFilter {
public:
    FileFilter(const ProcessingConfig& config) : config_(config) {
        // Compile regex pattern if needed
        if (!config_.pattern.empty() && config_.pattern != "*") {
            try {
                // Convert glob pattern to regex
                std::string regex_pattern = glob_to_regex(config_.pattern);
                pattern_regex_ = boost::regex(regex_pattern, 
                    config_.case_sensitive ? 0 : boost::regex::icase);
                use_pattern_ = true;
            } catch (const boost::regex_error& e) {
                throw std::runtime_error("Invalid pattern: " + std::string(e.what()));
            }
        }
        
        // Parse date filters
        if (!config_.newer_than.empty()) {
            newer_than_time_ = parse_date(config_.newer_than);
        }
        if (!config_.older_than.empty()) {
            older_than_time_ = parse_date(config_.older_than);
        }
        
        // Parse size filters
        if (!config_.max_size.empty()) {
            max_size_ = parse_size(config_.max_size);
        }
    }
    
    bool should_process(const fs::path& file_path) const {
        try {
            // Check if file exists and is regular file
            if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) {
                return false;
            }
            
            // Check pattern matching
            if (use_pattern_) {
                std::string filename = file_path.filename().string();
                if (!boost::regex_match(filename, pattern_regex_)) {
                    return false;
                }
            }
            
            // Check extension filters
            std::string extension = file_path.extension().string();
            if (!extension.empty() && extension[0] == '.') {
                extension = extension.substr(1); // Remove leading dot
            }
            
            if (!config_.include_extensions.empty()) {
                auto it = std::find(config_.include_extensions.begin(),
                                  config_.include_extensions.end(), extension);
                if (it == config_.include_extensions.end()) {
                    return false;
                }
            }
            
            if (!config_.exclude_extensions.empty()) {
                auto it = std::find(config_.exclude_extensions.begin(),
                                  config_.exclude_extensions.end(), extension);
                if (it != config_.exclude_extensions.end()) {
                    return false;
                }
            }
            
            // Check file size
            uintmax_t file_size = fs::file_size(file_path);
            if (file_size > max_size_) {
                return false;
            }
            
            // Check file dates
            std::time_t last_write = fs::last_write_time(file_path);
            
            if (newer_than_time_ != std::time_t(-1) && last_write <= newer_than_time_) {
                return false;
            }
            
            if (older_than_time_ != std::time_t(-1) && last_write >= older_than_time_) {
                return false;
            }
            
            return true;
            
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error checking file " << file_path << ": " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    const ProcessingConfig& config_;
    boost::regex pattern_regex_;
    bool use_pattern_ = false;
    std::time_t newer_than_time_ = std::time_t(-1);
    std::time_t older_than_time_ = std::time_t(-1);
    uintmax_t max_size_ = SIZE_MAX;
    
    std::string glob_to_regex(const std::string& glob) const {
        std::string regex;
        regex.reserve(glob.size() * 2);
        
        for (char c : glob) {
            switch (c) {
                case '*':
                    regex += ".*";
                    break;
                case '?':
                    regex += ".";
                    break;
                case '.':
                case '+':
                case '^':
                case '$':
                case '(':
                case ')':
                case '[':
                case ']':
                case '{':
                case '}':
                case '|':
                case '\\':
                    regex += '\\';
                    regex += c;
                    break;
                default:
                    regex += c;
                    break;
            }
        }
        
        return regex;
    }
    
    std::time_t parse_date(const std::string& date_str) const {
        std::tm tm = {};
        std::istringstream ss(date_str);
        ss >> std::get_time(&tm, "%Y-%m-%d");
        
        if (ss.fail()) {
            throw std::runtime_error("Invalid date format: " + date_str + " (expected YYYY-MM-DD)");
        }
        
        return std::mktime(&tm);
    }
    
    uintmax_t parse_size(const std::string& size_str) const {
        if (size_str.empty()) {
            return SIZE_MAX;
        }
        
        std::string number_part;
        std::string unit_part;
        
        size_t i = 0;
        while (i < size_str.length() && (std::isdigit(size_str[i]) || size_str[i] == '.')) {
            number_part += size_str[i];
            ++i;
        }
        
        while (i < size_str.length()) {
            unit_part += std::tolower(size_str[i]);
            ++i;
        }
        
        double size = std::stod(number_part);
        
        if (unit_part == "kb" || unit_part == "k") {
            size *= 1024;
        } else if (unit_part == "mb" || unit_part == "m") {
            size *= 1024 * 1024;
        } else if (unit_part == "gb" || unit_part == "g") {
            size *= 1024 * 1024 * 1024;
        } else if (unit_part == "tb" || unit_part == "t") {
            size *= 1024ULL * 1024 * 1024 * 1024;
        } else if (!unit_part.empty() && unit_part != "b") {
            throw std::runtime_error("Unknown size unit: " + unit_part);
        }
        
        return static_cast<uintmax_t>(size);
    }
};

class FileDiscovery {
public:
    FileDiscovery(const ProcessingConfig& config) 
        : config_(config), filter_(config) {}
    
    std::vector<fs::path> discover_files() {
        std::vector<fs::path> found_files;
        
        for (const auto& input_path : config_.input_paths) {
            fs::path path(input_path);
            
            if (fs::is_regular_file(path)) {
                if (filter_.should_process(path)) {
                    found_files.push_back(path);
                }
            } else if (fs::is_directory(path)) {
                discover_in_directory(path, found_files);
            } else {
                std::cerr << "Warning: Skipping invalid path: " << path << std::endl;
            }
        }
        
        if (config_.verbosity > 0) {
            std::cout << "Discovered " << found_files.size() << " files to process" << std::endl;
        }
        
        return found_files;
    }
    
private:
    const ProcessingConfig& config_;
    FileFilter filter_;
    
    void discover_in_directory(const fs::path& dir_path, std::vector<fs::path>& found_files) {
        try {
            if (config_.recursive) {
                for (fs::recursive_directory_iterator it(dir_path), end; it != end; ++it) {
                    if (fs::is_regular_file(it->path()) && filter_.should_process(it->path())) {
                        found_files.push_back(it->path());
                    }
                }
            } else {
                for (fs::directory_iterator it(dir_path), end; it != end; ++it) {
                    if (fs::is_regular_file(it->path()) && filter_.should_process(it->path())) {
                        found_files.push_back(it->path());
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error reading directory " << dir_path << ": " << e.what() << std::endl;
        }
    }
};
```

### Step 4: Content Processing Engine

```cpp
class EncodingDetector {
public:
    static std::string detect_encoding(const fs::path& file_path) {
        std::ifstream file(file_path.string(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + file_path.string());
        }
        
        // Read first few bytes to detect BOM
        std::vector<unsigned char> buffer(4);
        file.read(reinterpret_cast<char*>(buffer.data()), 4);
        size_t bytes_read = file.gcount();
        
        if (bytes_read >= 3) {
            // UTF-8 BOM
            if (buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF) {
                return "utf-8-bom";
            }
        }
        
        if (bytes_read >= 4) {
            // UTF-32 BOM
            if (buffer[0] == 0xFF && buffer[1] == 0xFE && buffer[2] == 0x00 && buffer[3] == 0x00) {
                return "utf-32le";
            }
            if (buffer[0] == 0x00 && buffer[1] == 0x00 && buffer[2] == 0xFE && buffer[3] == 0xFF) {
                return "utf-32be";
            }
        }
        
        if (bytes_read >= 2) {
            // UTF-16 BOM
            if (buffer[0] == 0xFF && buffer[1] == 0xFE) {
                return "utf-16le";
            }
            if (buffer[0] == 0xFE && buffer[1] == 0xFF) {
                return "utf-16be";
            }
        }
        
        // Heuristic detection for UTF-8 vs ASCII vs Latin-1
        file.seekg(0);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        return detect_encoding_heuristic(content);
    }
    
private:
    static std::string detect_encoding_heuristic(const std::string& content) {
        bool has_high_bytes = false;
        bool valid_utf8 = true;
        
        for (size_t i = 0; i < content.length(); ++i) {
            unsigned char c = content[i];
            
            if (c > 127) {
                has_high_bytes = true;
                
                // Check UTF-8 sequence
                if ((c & 0xE0) == 0xC0) {
                    // 2-byte sequence
                    if (i + 1 >= content.length() || (content[i + 1] & 0xC0) != 0x80) {
                        valid_utf8 = false;
                        break;
                    }
                    i += 1;
                } else if ((c & 0xF0) == 0xE0) {
                    // 3-byte sequence
                    if (i + 2 >= content.length() ||
                        (content[i + 1] & 0xC0) != 0x80 ||
                        (content[i + 2] & 0xC0) != 0x80) {
                        valid_utf8 = false;
                        break;
                    }
                    i += 2;
                } else if ((c & 0xF8) == 0xF0) {
                    // 4-byte sequence
                    if (i + 3 >= content.length() ||
                        (content[i + 1] & 0xC0) != 0x80 ||
                        (content[i + 2] & 0xC0) != 0x80 ||
                        (content[i + 3] & 0xC0) != 0x80) {
                        valid_utf8 = false;
                        break;
                    }
                    i += 3;
                } else {
                    valid_utf8 = false;
                    break;
                }
            }
        }
        
        if (!has_high_bytes) {
            return "ascii";
        } else if (valid_utf8) {
            return "utf-8";
        } else {
            return "latin-1";
        }
    }
};

class ContentProcessor {
public:
    ContentProcessor(const ProcessingConfig& config) : config_(config) {}
    
    bool process_file(const fs::path& input_path, const fs::path& output_path) {
        try {
            if (config_.verbosity > 1) {
                std::cout << "Processing: " << input_path << " -> " << output_path << std::endl;
            }
            
            // Read file content
            std::string content = read_file_content(input_path);
            
            // Apply content transformations
            if (!config_.search_pattern.empty()) {
                content = apply_search_replace(content);
            }
            
            if (config_.normalize_line_endings) {
                content = normalize_line_endings(content);
            }
            
            if (config_.remove_trailing_whitespace) {
                content = remove_trailing_whitespace(content);
            }
            
            // Create backup if requested
            if (config_.create_backups && input_path == output_path) {
                create_backup(input_path);
            }
            
            // Write to output file
            if (!config_.dry_run) {
                write_file_content(output_path, content);
                
                // Preserve file permissions and timestamps
                preserve_file_attributes(input_path, output_path);
                
                // Verify checksum if requested
                if (config_.verify_checksums) {
                    verify_file_integrity(input_path, output_path);
                }
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << input_path << ": " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    const ProcessingConfig& config_;
    
    std::string read_file_content(const fs::path& file_path) {
        std::ifstream file(file_path.string(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading");
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Handle encoding conversion if needed
        if (config_.encoding_from != config_.encoding_to) {
            std::string detected_encoding = config_.encoding_from;
            if (detected_encoding == "auto") {
                detected_encoding = EncodingDetector::detect_encoding(file_path);
            }
            
            if (detected_encoding != config_.encoding_to) {
                content = convert_encoding(content, detected_encoding, config_.encoding_to);
            }
        }
        
        return content;
    }
    
    void write_file_content(const fs::path& file_path, const std::string& content) {
        // Ensure output directory exists
        fs::path parent_dir = file_path.parent_path();
        if (!parent_dir.empty()) {
            fs::create_directories(parent_dir);
        }
        
        std::ofstream file(file_path.string(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing");
        }
        
        file.write(content.c_str(), content.length());
        
        if (!file.good()) {
            throw std::runtime_error("Error writing to file");
        }
    }
    
    std::string apply_search_replace(const std::string& content) {
        if (config_.use_regex) {
            try {
                boost::regex pattern(config_.search_pattern,
                    config_.case_sensitive ? 0 : boost::regex::icase);
                return boost::regex_replace(content, pattern, config_.replace_pattern);
            } catch (const boost::regex_error& e) {
                throw std::runtime_error("Regex error: " + std::string(e.what()));
            }
        } else {
            std::string result = content;
            if (config_.case_sensitive) {
                ba::replace_all(result, config_.search_pattern, config_.replace_pattern);
            } else {
                ba::ireplace_all(result, config_.search_pattern, config_.replace_pattern);
            }
            return result;
        }
    }
    
    std::string normalize_line_endings(const std::string& content) {
        std::string result;
        result.reserve(content.length());
        
#ifdef _WIN32
        const std::string line_ending = "\r\n";
#else
        const std::string line_ending = "\n";
#endif
        
        for (size_t i = 0; i < content.length(); ++i) {
            if (content[i] == '\r') {
                if (i + 1 < content.length() && content[i + 1] == '\n') {
                    // Windows line ending
                    result += line_ending;
                    ++i; // Skip the \n
                } else {
                    // Mac line ending
                    result += line_ending;
                }
            } else if (content[i] == '\n') {
                // Unix line ending
                result += line_ending;
            } else {
                result += content[i];
            }
        }
        
        return result;
    }
    
    std::string remove_trailing_whitespace(const std::string& content) {
        std::vector<std::string> lines;
        ba::split(lines, content, ba::is_any_of("\r\n"), ba::token_compress_on);
        
        for (auto& line : lines) {
            ba::trim_right(line);
        }
        
        return ba::join(lines, "\n");
    }
    
    std::string convert_encoding(const std::string& content, 
                               const std::string& from_encoding,
                               const std::string& to_encoding) {
        // Simplified encoding conversion
        // In a real implementation, you would use a library like ICU
        if (from_encoding == to_encoding) {
            return content;
        }
        
        // For this example, we'll just handle basic cases
        if (from_encoding == "utf-8-bom" && to_encoding == "utf-8") {
            // Remove BOM
            if (content.length() >= 3 && 
                content[0] == '\xEF' && content[1] == '\xBB' && content[2] == '\xBF') {
                return content.substr(3);
            }
        }
        
        // Add more encoding conversions as needed
        return content;
    }
    
    void create_backup(const fs::path& file_path) {
        fs::path backup_path = file_path;
        backup_path += config_.backup_suffix;
        
        if (fs::exists(backup_path)) {
            // Create unique backup name
            int counter = 1;
            fs::path unique_backup;
            do {
                unique_backup = file_path;
                unique_backup += config_.backup_suffix + "." + std::to_string(counter++);
            } while (fs::exists(unique_backup));
            backup_path = unique_backup;
        }
        
        fs::copy_file(file_path, backup_path);
        
        if (config_.verbosity > 1) {
            std::cout << "Created backup: " << backup_path << std::endl;
        }
    }
    
    void preserve_file_attributes(const fs::path& source, const fs::path& target) {
        if (source == target) return;
        
        try {
            // Copy file permissions
            fs::permissions(target, fs::status(source).permissions());
            
            // Copy timestamp
            fs::last_write_time(target, fs::last_write_time(source));
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Warning: Could not preserve file attributes: " << e.what() << std::endl;
        }
    }
    
    void verify_file_integrity(const fs::path& source, const fs::path& target) {
        if (source == target) return;
        
        // Calculate and compare checksums
        std::string source_checksum = calculate_checksum(source);
        std::string target_checksum = calculate_checksum(target);
        
        if (source_checksum != target_checksum) {
            throw std::runtime_error("Checksum verification failed");
        }
        
        if (config_.verbosity > 2) {
            std::cout << "Checksum verified: " << source_checksum << std::endl;
        }
    }
    
    std::string calculate_checksum(const fs::path& file_path) {
        std::ifstream file(file_path.string(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for checksum calculation");
        }
        
        boost::crc_32_type crc;
        
        char buffer[4096];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            crc.process_bytes(buffer, file.gcount());
        }
        
        return (format("%08X") % crc.checksum()).str();
    }
};
```

### Step 5: Progress Tracking and Parallel Processing

```cpp
class ProgressTracker {
public:
    ProgressTracker(size_t total_files) 
        : total_files_(total_files), processed_files_(0), failed_files_(0) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    void file_completed(bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        ++processed_files_;
        if (!success) {
            ++failed_files_;
        }
        
        // Update progress display
        update_progress_display();
    }
    
    void print_summary() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        
        std::cout << "\n--- Processing Summary ---\n";
        std::cout << "Total files: " << total_files_ << "\n";
        std::cout << "Processed successfully: " << (processed_files_ - failed_files_) << "\n";
        std::cout << "Failed: " << failed_files_ << "\n";
        std::cout << "Processing time: " << duration.count() << " ms\n";
        
        if (processed_files_ > 0) {
            double avg_time = static_cast<double>(duration.count()) / processed_files_;
            std::cout << "Average time per file: " << avg_time << " ms\n";
        }
    }
    
private:
    size_t total_files_;
    std::atomic<size_t> processed_files_;
    std::atomic<size_t> failed_files_;
    std::chrono::steady_clock::time_point start_time_;
    std::mutex mutex_;
    
    void update_progress_display() {
        if (total_files_ == 0) return;
        
        double percentage = (static_cast<double>(processed_files_) / total_files_) * 100.0;
        
        std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                  << percentage << "% (" << processed_files_ << "/" << total_files_ << ")";
        std::cout.flush();
    }
};

class FileProcessor {
public:
    FileProcessor(const ProcessingConfig& config) 
        : config_(config), content_processor_(config) {}
    
    void process_files() {
        // Discover files to process
        FileDiscovery discovery(config_);
        auto files = discovery.discover_files();
        
        if (files.empty()) {
            std::cout << "No files found to process." << std::endl;
            return;
        }
        
        // Initialize progress tracking
        ProgressTracker progress(files.size());
        
        if (config_.dry_run) {
            std::cout << "DRY RUN - The following files would be processed:" << std::endl;
            for (const auto& file : files) {
                std::cout << "  " << file << std::endl;
            }
            return;
        }
        
        // Process files
        if (config_.thread_count == 1) {
            process_files_sequential(files, progress);
        } else {
            process_files_parallel(files, progress);
        }
        
        progress.print_summary();
    }
    
private:
    const ProcessingConfig& config_;
    ContentProcessor content_processor_;
    
    void process_files_sequential(const std::vector<fs::path>& files, ProgressTracker& progress) {
        for (const auto& file : files) {
            fs::path output_path = determine_output_path(file);
            bool success = content_processor_.process_file(file, output_path);
            progress.file_completed(success);
        }
    }
    
    void process_files_parallel(const std::vector<fs::path>& files, ProgressTracker& progress) {
        boost::asio::thread_pool pool(config_.thread_count);
        
        for (const auto& file : files) {
            boost::asio::post(pool, [this, file, &progress]() {
                fs::path output_path = determine_output_path(file);
                bool success = content_processor_.process_file(file, output_path);
                progress.file_completed(success);
            });
        }
        
        pool.join();
    }
    
    fs::path determine_output_path(const fs::path& input_path) {
        if (config_.output_path.empty()) {
            // In-place processing
            return input_path;
        }
        
        fs::path output_dir(config_.output_path);
        
        if (config_.preserve_structure) {
            // Find the common base path for all input paths
            fs::path base_path = find_common_base_path();
            fs::path relative_path = fs::relative(input_path, base_path);
            return output_dir / relative_path;
        } else {
            // Flatten all files to output directory
            return output_dir / input_path.filename();
        }
    }
    
    fs::path find_common_base_path() {
        if (config_.input_paths.empty()) {
            return fs::current_path();
        }
        
        if (config_.input_paths.size() == 1) {
            fs::path path(config_.input_paths[0]);
            return fs::is_directory(path) ? path : path.parent_path();
        }
        
        // Find common prefix of all input paths
        fs::path common_path(config_.input_paths[0]);
        
        for (size_t i = 1; i < config_.input_paths.size(); ++i) {
            fs::path current_path(config_.input_paths[i]);
            
            // Find common prefix
            auto common_it = common_path.begin();
            auto current_it = current_path.begin();
            
            fs::path new_common;
            while (common_it != common_path.end() && 
                   current_it != current_path.end() &&
                   *common_it == *current_it) {
                new_common /= *common_it;
                ++common_it;
                ++current_it;
            }
            
            common_path = new_common;
        }
        
        return common_path;
    }
};
```

### Step 6: Main Application

```cpp
class Logger {
public:
    Logger(const std::string& log_file, int verbosity) 
        : verbosity_(verbosity) {
        if (!log_file.empty()) {
            log_stream_.open(log_file, std::ios::app);
            if (!log_stream_) {
                std::cerr << "Warning: Could not open log file: " << log_file << std::endl;
            }
        }
    }
    
    void log(int level, const std::string& message) {
        if (level <= verbosity_) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            std::string timestamp = std::asctime(std::localtime(&time_t));
            timestamp.pop_back(); // Remove newline
            
            std::string log_message = timestamp + " [" + level_to_string(level) + "] " + message;
            
            std::cout << log_message << std::endl;
            
            if (log_stream_.is_open()) {
                log_stream_ << log_message << std::endl;
                log_stream_.flush();
            }
        }
    }
    
private:
    int verbosity_;
    std::ofstream log_stream_;
    
    std::string level_to_string(int level) {
        switch (level) {
            case 0: return "ERROR";
            case 1: return "INFO";
            case 2: return "DEBUG";
            case 3: return "TRACE";
            default: return "UNKNOWN";
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        ProcessingConfig config = CommandLineParser::parse(argc, argv);
        
        // Initialize logger
        Logger logger(config.log_file, config.verbosity);
        
        logger.log(1, "File Processing Tool started");
        logger.log(2, "Configuration loaded successfully");
        
        // Create and run file processor
        FileProcessor processor(config);
        processor.process_files();
        
        logger.log(1, "File processing completed");
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Advanced Features

### Plugin Architecture

```cpp
class ProcessingPlugin {
public:
    virtual ~ProcessingPlugin() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual bool can_process(const fs::path& file_path) const = 0;
    virtual bool process(const fs::path& input_path, const fs::path& output_path) = 0;
};

class PluginManager {
public:
    void load_plugin(std::shared_ptr<ProcessingPlugin> plugin) {
        plugins_[plugin->name()] = plugin;
    }
    
    std::shared_ptr<ProcessingPlugin> find_plugin(const fs::path& file_path) {
        for (auto& pair : plugins_) {
            if (pair.second->can_process(file_path)) {
                return pair.second;
            }
        }
        return nullptr;
    }
    
private:
    std::map<std::string, std::shared_ptr<ProcessingPlugin>> plugins_;
};
```

### Duplicate File Detection

```cpp
class DuplicateDetector {
public:
    struct FileInfo {
        fs::path path;
        uintmax_t size;
        std::string checksum;
        std::time_t modified_time;
    };
    
    std::vector<std::vector<FileInfo>> find_duplicates(const std::vector<fs::path>& files) {
        std::map<std::string, std::vector<FileInfo>> checksum_map;
        
        // Calculate checksums and group by checksum
        for (const auto& file : files) {
            try {
                FileInfo info;
                info.path = file;
                info.size = fs::file_size(file);
                info.modified_time = fs::last_write_time(file);
                info.checksum = calculate_file_checksum(file);
                
                checksum_map[info.checksum].push_back(info);
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << file << ": " << e.what() << std::endl;
            }
        }
        
        // Extract groups with duplicates
        std::vector<std::vector<FileInfo>> duplicates;
        for (const auto& pair : checksum_map) {
            if (pair.second.size() > 1) {
                duplicates.push_back(pair.second);
            }
        }
        
        return duplicates;
    }
    
private:
    std::string calculate_file_checksum(const fs::path& file_path) {
        // Implementation similar to ContentProcessor::calculate_checksum
        // but with stronger hash algorithm for duplicate detection
        std::ifstream file(file_path.string(), std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for checksum");
        }
        
        // Use SHA-256 or similar for better collision resistance
        boost::crc_32_type crc;
        char buffer[8192];
        
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            crc.process_bytes(buffer, file.gcount());
        }
        
        return (format("%08X") % crc.checksum()).str();
    }
};
```

## Testing Framework

### Unit Tests

```cpp
#define BOOST_TEST_MODULE FileProcessorTests
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

BOOST_AUTO_TEST_SUITE(FileFilterTests)

BOOST_AUTO_TEST_CASE(TestPatternMatching) {
    ProcessingConfig config;
    config.pattern = "*.cpp";
    
    FileFilter filter(config);
    
    // Create temporary test files
    fs::path temp_dir = fs::temp_directory_path() / "fileproc_test";
    fs::create_directories(temp_dir);
    
    fs::path cpp_file = temp_dir / "test.cpp";
    fs::path h_file = temp_dir / "test.h";
    
    std::ofstream(cpp_file.string()) << "// C++ file\n";
    std::ofstream(h_file.string()) << "// Header file\n";
    
    BOOST_CHECK(filter.should_process(cpp_file));
    BOOST_CHECK(!filter.should_process(h_file));
    
    // Cleanup
    fs::remove_all(temp_dir);
}

BOOST_AUTO_TEST_CASE(TestSizeFiltering) {
    ProcessingConfig config;
    config.max_size = "1KB";
    
    FileFilter filter(config);
    
    // Test would create files of different sizes and verify filtering
    // Implementation details omitted for brevity
}

BOOST_AUTO_TEST_SUITE_END()
```

## Performance Benchmarks

### Benchmark Script

```cpp
#include <chrono>
#include <random>

class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        size_t files_processed;
        std::chrono::milliseconds total_time;
        std::chrono::milliseconds avg_time_per_file;
        size_t total_bytes_processed;
        double throughput_mbps;
    };
    
    BenchmarkResult run_benchmark(const ProcessingConfig& config, 
                                 const std::vector<fs::path>& test_files) {
        auto start_time = std::chrono::steady_clock::now();
        
        FileProcessor processor(config);
        processor.process_files();
        
        auto end_time = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        BenchmarkResult result;
        result.files_processed = test_files.size();
        result.total_time = total_time;
        result.avg_time_per_file = std::chrono::milliseconds(total_time.count() / test_files.size());
        
        // Calculate total bytes processed
        result.total_bytes_processed = 0;
        for (const auto& file : test_files) {
            try {
                result.total_bytes_processed += fs::file_size(file);
            } catch (...) {}
        }
        
        // Calculate throughput
        double seconds = total_time.count() / 1000.0;
        double megabytes = result.total_bytes_processed / (1024.0 * 1024.0);
        result.throughput_mbps = megabytes / seconds;
        
        return result;
    }
    
    void create_test_dataset(const fs::path& base_dir, size_t num_files, size_t avg_file_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> size_dist(avg_file_size / 2, avg_file_size * 2);
        
        fs::create_directories(base_dir);
        
        for (size_t i = 0; i < num_files; ++i) {
            fs::path file_path = base_dir / ("test_file_" + std::to_string(i) + ".txt");
            size_t file_size = size_dist(gen);
            
            std::ofstream file(file_path.string());
            for (size_t j = 0; j < file_size; ++j) {
                file << static_cast<char>('A' + (j % 26));
                if (j % 80 == 79) file << '\n';
            }
        }
    }
};
```

## Deployment and Usage

### Build Instructions

```cmake
cmake_minimum_required(VERSION 3.10)
project(FileProcessor)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS 
    system filesystem program_options regex thread)

add_executable(fileproc
    src/main.cpp
    src/file_processor.cpp
    src/content_processor.cpp
    src/file_filter.cpp
    src/progress_tracker.cpp
)

target_link_libraries(fileproc 
    ${Boost_LIBRARIES}
    pthread
)

target_include_directories(fileproc PRIVATE 
    ${Boost_INCLUDE_DIRS}
    include/
)
```

### Usage Examples

```bash
# Basic text processing
./fileproc -i ./source_code -o ./processed --search "TODO" --replace "FIXME" -r

# Encoding conversion
./fileproc -i ./documents --encoding-from latin-1 --encoding-to utf-8 -r

# File organization
./fileproc -i ./downloads -o ./organized --include-ext jpg png gif --preserve-structure

# Batch renaming with regex
./fileproc -i ./photos --search "IMG_(\d+)" --replace "Photo_$1" --regex -n

# Performance optimization
./fileproc -i ./large_dataset -o ./processed -t 8 --max-size 100MB -v 2
```

## Assessment Criteria

- [ ] Implements comprehensive file system operations
- [ ] Demonstrates cross-platform compatibility
- [ ] Provides robust command-line interface
- [ ] Handles various file formats and encodings
- [ ] Includes performance optimization techniques
- [ ] Implements proper error handling and recovery
- [ ] Provides comprehensive logging and progress tracking
- [ ] Includes thorough testing suite
- [ ] Achieves performance benchmarks
- [ ] Includes complete documentation

## Deliverables

1. Complete cross-platform file processing application
2. Comprehensive test suite with unit and integration tests
3. Performance benchmarking framework and results
4. User documentation and usage examples
5. Deployment guide for different platforms
6. Plugin development documentation
7. Performance optimization report
