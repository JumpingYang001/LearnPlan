# String Processing and Text Handling with Boost

*Duration: 1 week*

## Overview

String processing and text manipulation are fundamental aspects of modern C++ programming. This comprehensive section covers Boost's powerful string processing libraries, which provide advanced capabilities beyond the standard library. You'll learn to handle complex text processing tasks, implement robust pattern matching, and build efficient text processing pipelines.

### What You'll Learn
- **Advanced Regular Expressions**: Master Boost.Regex for complex pattern matching
- **String Algorithms**: Efficient text manipulation and processing techniques
- **Flexible Tokenization**: Parse structured and unstructured text data
- **Type-Safe Formatting**: Professional string formatting and templating
- **Performance Optimization**: Best practices for high-performance text processing
- **Real-World Applications**: Build practical text processing solutions

### Prerequisites
- Basic C++ knowledge (STL containers, iterators, algorithms)
- Understanding of regular expressions (basic level)
- Familiarity with string handling in C++
- Basic knowledge of character encodings (ASCII, UTF-8)

### Why Boost String Libraries Matter

**Beyond Standard Library Limitations:**
```cpp
// Standard C++ limitations
std::string text = "  Hello World  ";
// No built-in trim function
// Limited regex support in older standards
// No advanced tokenization
// Printf-style formatting safety issues

// Boost provides comprehensive solutions
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>

// Rich, safe, and efficient string processing
std::string trimmed = boost::trim_copy(text);
```

**Real-World Applications:**
- **Web Development**: URL parsing, HTML processing, log analysis
- **Data Processing**: CSV parsing, ETL pipelines, data cleaning
- **Configuration Management**: Config file parsing, template processing
- **Natural Language Processing**: Text analysis, pattern extraction
- **System Administration**: Log parsing, report generation

## Learning Topics

### Boost.Regex - Advanced Pattern Matching

#### Understanding Regular Expressions in Boost
Boost.Regex provides a more mature and feature-rich regular expression engine compared to early versions of `std::regex`. It supports Perl-compatible regular expressions (PCRE) with additional optimizations and features.

**Key Advantages of Boost.Regex:**
- **Mature and Stable**: Battle-tested in production environments
- **PCRE Compatibility**: Full Perl regex syntax support
- **Performance Optimized**: Efficient compilation and execution
- **Rich Feature Set**: Named captures, conditional matching, recursive patterns
- **Thread Safety**: Safe for multi-threaded applications
- **Locale Support**: Unicode and internationalization support

**Architecture Overview:**
```cpp
// Boost.Regex component architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pattern       │───▶│   Compiled      │───▶│   Match         │
│   (string)      │    │   Regex         │    │   Results       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Raw pattern          Optimized FSM           Capture groups
   R"(\d{3}-\d{4})"    Internal state         match[0], match[1]
```

#### Advanced Regular Expressions Beyond std::regex

**Pattern Compilation and Reuse:**
```cpp
#include <boost/regex.hpp>
#include <chrono>
#include <iostream>

void demonstrate_pattern_compilation() {
    const std::string text = "Phone: 123-456-7890, Fax: 987-654-3210, Mobile: 555-123-4567";
    const std::string pattern = R"(\b\d{3}-\d{3}-\d{4}\b)";
    
    // BAD: Compiling pattern multiple times
    auto start_bad = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        boost::regex temp_regex(pattern);  // Expensive compilation each time
        boost::smatch match;
        boost::regex_search(text, match, temp_regex);
    }
    auto end_bad = std::chrono::high_resolution_clock::now();
    
    // GOOD: Compile once, reuse many times
    boost::regex phone_regex(pattern);  // Compile once
    auto start_good = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        boost::smatch match;
        boost::regex_search(text, match, phone_regex);  // Reuse compiled pattern
    }
    auto end_good = std::chrono::high_resolution_clock::now();
    
    std::cout << "Bad approach: " << 
        std::chrono::duration_cast<std::chrono::microseconds>(end_bad - start_bad).count() << "μs\n";
    std::cout << "Good approach: " << 
        std::chrono::duration_cast<std::chrono::microseconds>(end_good - start_good).count() << "μs\n";
}
```

#### Perl-Compatible Regex Syntax

**Advanced Pattern Features:**
```cpp
void demonstrate_advanced_patterns() {
    // Lookahead and lookbehind assertions
    std::string password = "MySecure123!";
    
    // Password validation: at least 8 chars, contains digit, uppercase, lowercase, special
    boost::regex password_pattern(
        R"(^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}$)"
    );
    
    if (boost::regex_match(password, password_pattern)) {
        std::cout << "Strong password!\n";
    }
    
    // Non-greedy matching
    std::string html = "<div>Content 1</div><div>Content 2</div>";
    
    // Greedy: matches entire string
    boost::regex greedy_pattern("<div>.*</div>");
    
    // Non-greedy: matches individual tags
    boost::regex non_greedy_pattern("<div>.*?</div>");
    
    boost::sregex_iterator start(html.begin(), html.end(), non_greedy_pattern);
    boost::sregex_iterator end;
    
    std::cout << "Individual div tags:\n";
    for (auto it = start; it != end; ++it) {
        std::cout << "  " << it->str() << "\n";
    }
}
```

#### Named Captures and Backreferences

**Named Capture Groups:**
```cpp
void demonstrate_named_captures_detailed() {
    // Log parsing with named groups
    std::string log_entries = R"(
        2023-06-24 10:30:15.123 [INFO] User login successful - user_id: 12345
        2023-06-24 10:31:22.456 [ERROR] Database connection failed - error_code: 1042
        2023-06-24 10:32:01.789 [WARN] High memory usage detected - memory: 85%
    )";
    
    // Complex named capture pattern
    boost::regex log_pattern(
        R"((?<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+)"
        R"(\[(?<level>INFO|ERROR|WARN|DEBUG)\]\s+)"
        R"((?<message>.*?)(?:\s+-\s+(?<metadata>\w+:\s*[^-\n]+))?)"
    );
    
    boost::sregex_iterator iter(log_entries.begin(), log_entries.end(), log_pattern);
    boost::sregex_iterator end;
    
    int entry_count = 0;
    for (auto it = iter; it != end; ++it) {
        const boost::smatch& match = *it;
        
        std::cout << "Log Entry " << ++entry_count << ":\n";
        std::cout << "  Timestamp: " << match["timestamp"] << "\n";
        std::cout << "  Level: " << match["level"] << "\n";
        std::cout << "  Message: " << match["message"] << "\n";
        if (match["metadata"].matched) {
            std::cout << "  Metadata: " << match["metadata"] << "\n";
        }
        std::cout << "\n";
    }
}
```

**Backreferences for Pattern Matching:**
```cpp
void demonstrate_backreferences() {
    // Find repeated words
    std::string text = "This is is a test test with repeated repeated words.";
    boost::regex repeated_pattern(R"(\b(\w+)\s+\1\b)");
    
    std::cout << "Repeated words found:\n";
    boost::sregex_iterator start(text.begin(), text.end(), repeated_pattern);
    boost::sregex_iterator end;
    
    for (auto it = start; it != end; ++it) {
        std::cout << "  '" << it->str(1) << "' repeated at position " 
                  << it->position() << "\n";
    }
    
    // HTML tag matching with backreferences
    std::string html = "<p>Paragraph</p><div>Division</div><span>Span text</span>";
    boost::regex tag_pattern(R"(<(\w+)>.*?</\1>)");
    
    std::cout << "\nMatched HTML tags:\n";
    boost::sregex_iterator html_start(html.begin(), html.end(), tag_pattern);
    
    for (auto it = html_start; it != boost::sregex_iterator(); ++it) {
        std::cout << "  Tag: " << it->str(1) << ", Content: " << it->str() << "\n";
    }
}
```

#### Performance Optimization Techniques

**Regex Compilation Options:**
```cpp
void demonstrate_regex_optimization() {
    std::string large_text = /* ... large text content ... */;
    
    // Default compilation
    boost::regex default_regex(R"(\b\w+@\w+\.\w+\b)");
    
    // Optimized compilation with flags
    boost::regex optimized_regex(
        R"(\b\w+@\w+\.\w+\b)",
        boost::regex::optimize | boost::regex::nosubs  // nosubs if captures not needed
    );
    
    // Case-insensitive compilation
    boost::regex case_insensitive(
        R"(\b\w+@\w+\.\w+\b)",
        boost::regex::icase | boost::regex::optimize
    );
    
    // Performance comparison
    auto benchmark_regex = [&](const boost::regex& regex, const std::string& name) {
        auto start = std::chrono::high_resolution_clock::now();
        
        int matches = 0;
        boost::sregex_iterator iter(large_text.begin(), large_text.end(), regex);
        boost::sregex_iterator end;
        
        for (auto it = iter; it != end; ++it) {
            ++matches;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start);
        
        std::cout << name << ": " << matches << " matches in " << duration.count() << "μs\n";
    };
    
    benchmark_regex(default_regex, "Default");
    benchmark_regex(optimized_regex, "Optimized");
    benchmark_regex(case_insensitive, "Case-insensitive");
}
```

**Memory-Efficient Matching:**
```cpp
void demonstrate_memory_efficient_matching() {
    // For large files, use streaming approach instead of loading entire content
    std::ifstream file("large_log_file.txt");
    std::string line;
    boost::regex pattern(R"(ERROR.*?(\d+))");
    
    int error_count = 0;
    while (std::getline(file, line)) {
        boost::smatch match;
        if (boost::regex_search(line, match, pattern)) {
            ++error_count;
            // Process match immediately, don't store
            std::cout << "Error code: " << match[1] << "\n";
        }
    }
    
    std::cout << "Total errors found: " << error_count << "\n";
}
```

### Boost.String_Algo - Comprehensive String Manipulation

#### Understanding String Algorithms
Boost.String_Algo provides a comprehensive collection of string manipulation functions that fill gaps in the standard library. These algorithms are designed for efficiency, safety, and ease of use.

**Algorithm Categories:**
```cpp
// Classification of Boost.String_Algo functions
┌─────────────────────────────────────────────────────────────┐
│                    Boost.String_Algo                        │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Trimming      │   Case          │   Find & Replace        │
│   - trim        │   - to_upper    │   - replace_all         │
│   - trim_left   │   - to_lower    │   - replace_first       │
│   - trim_right  │   - to_title    │   - erase_all          │
├─────────────────┼─────────────────┼─────────────────────────┤
│   Splitting     │   Joining       │   Predicates           │
│   - split       │   - join        │   - starts_with        │
│   - iter_split  │   - join_if     │   - ends_with          │
│   - find_all    │                 │   - contains           │
└─────────────────┴─────────────────┴─────────────────────────┘
```

#### String Manipulation Algorithms

**Advanced Trimming Operations:**
```cpp
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>

void demonstrate_advanced_trimming() {
    std::string data = "   \t\n  Hello World!  \r\n  ";
    
    // Basic trimming
    std::string basic_trim = boost::trim_copy(data);
    std::cout << "Basic trim: '" << basic_trim << "'\n";
    
    // Custom character trimming
    std::string custom_data = "***Hello World!***";
    std::string custom_trim = boost::trim_copy_if(custom_data, boost::is_any_of("*"));
    std::cout << "Custom trim: '" << custom_trim << "'\n";
    
    // Predicate-based trimming
    std::string mixed_data = "123abc456def789";
    std::string digit_trim = boost::trim_copy_if(mixed_data, boost::is_digit());
    std::cout << "Digit trim: '" << digit_trim << "'\n";
    
    // Left and right specific trimming
    std::string asymmetric = "   Left trim only";
    boost::trim_left(asymmetric);
    std::cout << "Left trimmed: '" << asymmetric << "'\n";
    
    // In-place vs copy operations
    std::string original = "  modify me  ";
    std::string copied = boost::trim_copy(original);  // Original unchanged
    boost::trim(original);  // Modifies original
    
    std::cout << "Original after in-place trim: '" << original << "'\n";
    std::cout << "Copy result: '" << copied << "'\n";
}
```

#### Case Conversion with Locale Support

**Comprehensive Case Handling:**
```cpp
void demonstrate_case_conversion() {
    // Basic case conversion
    std::string text = "Hello World! 123 Test";
    
    std::cout << "Original: " << text << "\n";
    std::cout << "Upper: " << boost::to_upper_copy(text) << "\n";
    std::cout << "Lower: " << boost::to_lower_copy(text) << "\n";
    
    // Locale-aware conversion
    std::locale german_locale("de_DE.UTF-8");
    std::string german_text = "Größe";
    
    try {
        std::string german_upper = boost::to_upper_copy(german_text, german_locale);
        std::cout << "German upper: " << german_upper << "\n";
    } catch (const std::exception& e) {
        std::cout << "Locale not available, using default\n";
        std::cout << "German upper: " << boost::to_upper_copy(german_text) << "\n";
    }
    
    // Title case conversion
    std::string title_text = "the quick brown fox jumps over the lazy dog";
    
    // Custom title case implementation
    auto to_title_case = [](const std::string& input) {
        std::string result = boost::to_lower_copy(input);
        bool capitalize_next = true;
        
        for (char& c : result) {
            if (std::isalpha(c) && capitalize_next) {
                c = std::toupper(c);
                capitalize_next = false;
            } else if (std::isspace(c) || std::ispunct(c)) {
                capitalize_next = true;
            }
        }
        return result;
    };
    
    std::cout << "Title case: " << to_title_case(title_text) << "\n";
}
```

#### Find and Replace Operations

**Advanced Find and Replace:**
```cpp
void demonstrate_find_replace_operations() {
    std::string document = R"(
        The quick brown fox jumps over the lazy dog.
        The quick brown fox is very quick indeed.
        The lazy dog sleeps under the quick brown fox.
    )";
    
    // Case-sensitive replace
    std::string result1 = document;
    boost::replace_all(result1, "quick", "FAST");
    std::cout << "Case-sensitive replace:\n" << result1 << "\n";
    
    // Case-insensitive replace
    std::string result2 = document;
    boost::ireplace_all(result2, "QUICK", "speedy");
    std::cout << "Case-insensitive replace:\n" << result2 << "\n";
    
    // First occurrence only
    std::string result3 = document;
    boost::replace_first(result3, "the", "THE");
    std::cout << "Replace first:\n" << result3 << "\n";
    
    // Conditional replace with predicate
    std::string code = "var_name = getValue(); var_temp = getTemp(); var_count = getCount();";
    std::string result4 = code;
    
    // Replace variables starting with "var_" to use camelCase
    boost::regex var_pattern(R"(\bvar_(\w+))");
    result4 = boost::regex_replace(result4, var_pattern, 
        [](const boost::smatch& match) -> std::string {
            std::string var_name = match[1].str();
            // Convert to camelCase
            var_name[0] = std::tolower(var_name[0]);
            return var_name;
        });
    
    std::cout << "Variable rename:\n" << result4 << "\n";
    
    // Erase operations
    std::string result5 = document;
    boost::erase_all(result5, "brown ");
    std::cout << "Erase all 'brown ':\n" << result5 << "\n";
}
```

#### Predicate-Based String Operations

**Advanced Predicates and Custom Operations:**
```cpp
void demonstrate_predicate_operations() {
    std::string filename = "Document_2023.PDF";
    std::string url = "https://www.example.com/path/file.html";
    std::string code = "function validateInput(value) { return value != null; }";
    
    // Built-in predicates
    std::cout << "Filename analysis:\n";
    std::cout << "  Starts with 'Doc': " << boost::istarts_with(filename, "doc") << "\n";
    std::cout << "  Ends with '.pdf': " << boost::iends_with(filename, ".pdf") << "\n";
    std::cout << "  Contains '2023': " << boost::contains(filename, "2023") << "\n";
    
    // URL validation
    std::cout << "\nURL analysis:\n";
    std::cout << "  Is HTTPS: " << boost::starts_with(url, "https://") << "\n";
    std::cout << "  Is HTML: " << boost::ends_with(url, ".html") << "\n";
    
    // Custom predicates
    auto is_camel_case = [](char c) { return std::isupper(c); };
    auto has_camel_case = boost::find_if(filename, is_camel_case) != filename.end();
    std::cout << "  Has camelCase: " << has_camel_case << "\n";
    
    // Find patterns
    auto find_functions = [](const std::string& source) {
        std::vector<std::string> functions;
        boost::regex func_pattern(R"(\bfunction\s+(\w+)\s*\()");
        
        boost::sregex_iterator start(source.begin(), source.end(), func_pattern);
        boost::sregex_iterator end;
        
        for (auto it = start; it != end; ++it) {
            functions.push_back(it->str(1));
        }
        return functions;
    };
    
    auto functions = find_functions(code);
    std::cout << "\nFunctions found in code:\n";
    for (const auto& func : functions) {
        std::cout << "  " << func << "\n";
    }
    
    // All/any/none predicates
    std::string numbers = "12345";
    std::string mixed = "123abc";
    std::string letters = "abcdef";
    
    std::cout << "\nCharacter analysis:\n";
    std::cout << "  All digits in '" << numbers << "': " << 
        boost::all(numbers, boost::is_digit()) << "\n";
    std::cout << "  Any digits in '" << mixed << "': " << 
        boost::any_of(mixed, boost::is_digit()) << "\n";
    std::cout << "  No digits in '" << letters << "': " << 
        !boost::any_of(letters, boost::is_digit()) << "\n";
}
```

### Boost.Tokenizer - Flexible Text Parsing

#### Understanding Tokenization
Tokenization is the process of breaking text into meaningful units (tokens) based on delimiters or patterns. Boost.Tokenizer provides a flexible framework for parsing structured and semi-structured text data.

**Tokenizer Architecture:**
```cpp
// Boost.Tokenizer component hierarchy
┌─────────────────────────────────────────────────────────────┐
│                    boost::tokenizer                         │
├─────────────────────────────────────────────────────────────┤
│  Template Parameters:                                       │
│  - TokenizerFunc: How to split (char_separator, etc.)      │
│  - Iterator: Input iterator type                           │
│  - Type: Token type (usually std::string)                  │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                Tokenizer Functions                         │
├─────────────────┬─────────────────┬─────────────────────────┤
│ char_separator   │ escaped_list    │ offset_separator        │
│ - Simple delim   │ - CSV parsing   │ - Fixed positions       │
│ - Custom rules   │ - Escaped chars │ - Binary data          │
└─────────────────┴─────────────────┴─────────────────────────┘
```

#### Flexible Tokenization Strategies

**Character-Based Separation:**
```cpp
#include <boost/tokenizer.hpp>
#include <iostream>
#include <string>
#include <vector>

void demonstrate_char_separator_strategies() {
    std::string data = "apple,banana;orange:grape|lemon";
    
    // Basic multi-character separator
    boost::char_separator<char> sep(",;:|");
    boost::tokenizer<boost::char_separator<char>> basic_tokens(data, sep);
    
    std::cout << "Basic tokenization:\n";
    for (const auto& token : basic_tokens) {
        std::cout << "  '" << token << "'\n";
    }
    
    // Advanced separator with kept delimiters
    std::string expression = "a+b-c*d/e";
    boost::char_separator<char> math_sep("+-*/", "+-*/");  // Second param: kept separators
    boost::tokenizer<boost::char_separator<char>> math_tokens(expression, math_sep);
    
    std::cout << "\nMath expression tokenization (keeping operators):\n";
    for (const auto& token : math_tokens) {
        std::cout << "  '" << token << "'\n";
    }
    
    // Custom separator behavior
    std::string messy_data = "  token1  ,,  token2  ;  ; token3  ";
    boost::char_separator<char> clean_sep(",; ", "",   // Separators, kept separators
                                         boost::keep_empty_tokens);  // Keep empty tokens
    boost::tokenizer<boost::char_separator<char>> clean_tokens(messy_data, clean_sep);
    
    std::cout << "\nMessy data tokenization (with empty tokens):\n";
    int i = 0;
    for (const auto& token : clean_tokens) {
        std::cout << "  [" << i++ << "] '" << token << "'\n";
    }
    
    // Drop empty tokens (default behavior)
    boost::char_separator<char> drop_empty_sep(",; ");
    boost::tokenizer<boost::char_separator<char>> drop_empty_tokens(messy_data, drop_empty_sep);
    
    std::cout << "\nMessy data tokenization (dropping empty tokens):\n";
    i = 0;
    for (const auto& token : drop_empty_tokens) {
        std::cout << "  [" << i++ << "] '" << token << "'\n";
    }
}
```

#### Custom Token Separators and Escape Sequences

**Advanced Tokenization Scenarios:**
```cpp
void demonstrate_custom_tokenization() {
    // Custom tokenizer for configuration files
    std::string config = R"(
        # Configuration file
        server_host = localhost
        server_port = 8080
        database_url = "postgres://user:pass@localhost/db"
        debug_mode = true
        # End of config
    )";
    
    // Tokenize configuration lines
    boost::char_separator<char> line_sep("\n\r");
    boost::tokenizer<boost::char_separator<char>> lines(config, line_sep);
    
    std::cout << "Configuration parsing:\n";
    for (const auto& line : lines) {
        std::string trimmed_line = boost::trim_copy(line);
        
        // Skip empty lines and comments
        if (trimmed_line.empty() || trimmed_line[0] == '#') {
            continue;
        }
        
        // Parse key-value pairs
        boost::char_separator<char> kv_sep(" = ");
        boost::tokenizer<boost::char_separator<char>> kv_tokens(trimmed_line, kv_sep);
        
        auto it = kv_tokens.begin();
        if (it != kv_tokens.end()) {
            std::string key = *it++;
            std::string value = (it != kv_tokens.end()) ? *it : "";
            
            // Remove quotes if present
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            
            std::cout << "  " << key << " -> " << value << "\n";
        }
    }
}
```

#### CSV Parsing and Structured Data Extraction

**Professional CSV Handling:**
```cpp
void demonstrate_advanced_csv_parsing() {
    // Complex CSV with various edge cases
    std::string csv_data = R"(Name,Age,"Address, City",Phone,"Notes, Comments"
"John Doe",30,"123 Main St, New York",555-1234,"Likes ""pizza"" and pasta"
"Jane Smith",25,"456 Oak Ave, Los Angeles",555-5678,"Has dog, named ""Rex"""
"Bob Johnson",35,"789 Pine Rd, Chicago",555-9012,"Single, no pets"
"Mary Wilson",28,"321 Elm Dr, Miami",555-3456,"Married, 2 kids")";
    
    // Parse CSV with proper quote handling
    std::istringstream csv_stream(csv_data);
    std::string line;
    std::vector<std::vector<std::string>> csv_records;
    
    while (std::getline(csv_stream, line)) {
        boost::tokenizer<boost::escaped_list_separator<char>> csv_tokens(line);
        std::vector<std::string> record(csv_tokens.begin(), csv_tokens.end());
        csv_records.push_back(record);
    }
    
    // Display parsed CSV
    std::cout << "Parsed CSV data:\n";
    for (size_t i = 0; i < csv_records.size(); ++i) {
        std::cout << "Record " << i << ":\n";
        for (size_t j = 0; j < csv_records[i].size(); ++j) {
            std::cout << "  Field " << j << ": '" << csv_records[i][j] << "'\n";
        }
        std::cout << "\n";
    }
    
    // Extract specific data
    if (csv_records.size() > 1) {  // Skip header
        std::cout << "People over 30:\n";
        for (size_t i = 1; i < csv_records.size(); ++i) {
            if (csv_records[i].size() > 1) {
                int age = std::stoi(csv_records[i][1]);
                if (age > 30) {
                    std::cout << "  " << csv_records[i][0] << " (age " << age << ")\n";
                }
            }
        }
    }
}
```

#### Iterator-Based Token Access

**Advanced Iterator Patterns:**
```cpp
void demonstrate_iterator_based_access() {
    std::string data = "token1,token2,token3,token4,token5";
    boost::char_separator<char> sep(",");
    boost::tokenizer<boost::char_separator<char>> tokens(data, sep);
    
    // Random access pattern (collect to vector first)
    std::vector<std::string> token_vector(tokens.begin(), tokens.end());
    
    std::cout << "Random access to tokens:\n";
    std::cout << "  First token: " << token_vector.front() << "\n";
    std::cout << "  Last token: " << token_vector.back() << "\n";
    std::cout << "  Middle token: " << token_vector[token_vector.size() / 2] << "\n";
    
    // Conditional processing
    std::cout << "\nFiltered tokens (length > 6):\n";
    std::copy_if(tokens.begin(), tokens.end(),
                 std::ostream_iterator<std::string>(std::cout, "\n"),
                 [](const std::string& token) { return token.length() > 6; });
    
    // Advanced iterator operations
    auto it = tokens.begin();
    std::advance(it, 2);  // Move to third token
    std::cout << "\nThird token: " << *it << "\n";
    
    // Distance between iterators
    auto start = tokens.begin();
    auto end = tokens.end();
    std::cout << "Total tokens: " << std::distance(start, end) << "\n";
    
    // Reverse iteration (requires collecting to container first)
    std::cout << "\nReverse order:\n";
    for (auto rit = token_vector.rbegin(); rit != token_vector.rend(); ++rit) {
        std::cout << "  " << *rit << "\n";
    }
}
```

### Boost.Format - Type-Safe String Formatting

#### Understanding Format Specifications
Boost.Format provides a printf-style formatting system with type safety and additional features. It bridges the gap between printf's convenience and iostream's safety.

**Format Syntax Overview:**
```cpp
// Boost.Format syntax patterns
┌─────────────────────────────────────────────────────────────┐
│                    Format Specifications                    │
├─────────────────────────────────────────────────────────────┤
│  %[position][flags][width][precision][conversion]          │
│                                                             │
│  Position: %1%, %2%, %3% (1-based indexing)               │
│  Flags: -, +, #, 0, space                                  │
│  Width: minimum field width                                 │
│  Precision: decimal places for floats                      │
│  Conversion: d, x, f, s, etc.                             │
└─────────────────────────────────────────────────────────────┘
```

#### Type-Safe String Formatting

**Basic to Advanced Formatting:**
```cpp
#include <boost/format.hpp>
#include <iostream>
#include <iomanip>

void demonstrate_type_safe_formatting() {
    // Basic type safety
    try {
        boost::format fmt("Number: %1%, String: %2%");
        std::string result = (fmt % 42 % "Hello").str();
        std::cout << result << "\n";
        
        // This would be safe - format validates types
        boost::format safe_fmt("Value: %1%");
        std::cout << (safe_fmt % 3.14159) << "\n";
        
    } catch (const boost::io::format_error& e) {
        std::cout << "Format error: " << e.what() << "\n";
    }
    
    // Type conversion examples
    boost::format conversion_fmt("Int: %1%, Float: %2$8.2f, Hex: %1$#x");
    std::cout << (conversion_fmt % 255 % 3.14159) << "\n";
    
    // Boolean formatting
    boost::format bool_fmt("Success: %1%, Enabled: %2%");
    std::cout << (bool_fmt % true % false) << "\n";
    
    // Custom formatting for user types
    struct Point {
        double x, y;
        Point(double x, double y) : x(x), y(y) {}
    };
    
    // Custom formatter for Point
    auto format_point = [](const Point& p) {
        boost::format fmt("Point(%1$.2f, %2$.2f)");
        return (fmt % p.x % p.y).str();
    };
    
    Point p(3.14159, 2.71828);
    std::cout << "Custom type: " << format_point(p) << "\n";
}
```

#### Positional and Named Parameters

**Advanced Parameter Handling:**
```cpp
void demonstrate_advanced_parameters() {
    // Positional parameters with reordering
    boost::format reorder_fmt("Last: %3%, First: %1%, Middle: %2%");
    std::cout << (reorder_fmt % "John" % "Q" % "Doe") << "\n";
    
    // Repeated parameters
    boost::format repeat_fmt("Value %1% appears %2% times, %1% is important");
    std::cout << (repeat_fmt % "X" % 5) << "\n";
    
    // Complex formatting with multiple types
    boost::format complex_fmt(
        "Report: %1% - Date: %2%, Items: %3%, Total: $%4$,.2f, Status: %5%"
    );
    
    std::string report = (complex_fmt 
        % "Monthly Sales"
        % "2023-06-24"
        % 150
        % 12345.67
        % "Complete").str();
    
    std::cout << report << "\n";
    
    // Conditional formatting
    auto format_grade = [](int score) {
        boost::format grade_fmt("Score: %1% (%2%)");
        std::string grade = (score >= 90) ? "A" :
                           (score >= 80) ? "B" :
                           (score >= 70) ? "C" :
                           (score >= 60) ? "D" : "F";
        return (grade_fmt % score % grade).str();
    };
    
    std::cout << format_grade(95) << "\n";
    std::cout << format_grade(73) << "\n";
    std::cout << format_grade(45) << "\n";
}
```

#### Custom Format Specifications

**Professional Formatting Techniques:**
```cpp
void demonstrate_custom_format_specs() {
    // Table formatting
    std::vector<std::tuple<std::string, int, double>> products = {
        {"Widget A", 150, 29.99},
        {"Super Gadget Pro", 75, 149.50},
        {"Tool", 200, 15.25}
    };
    
    // Table header
    boost::format header_fmt("| %1$-20s | %2$8s | %3$10s |");
    boost::format row_fmt("| %1$-20s | %2$8d | %3$10.2f |");
    boost::format separator("+%1$s+%2$s+%3$s+");
    
    std::string sep_line = (separator % std::string(22, '-') 
                                    % std::string(10, '-') 
                                    % std::string(12, '-')).str();
    
    std::cout << sep_line << "\n";
    std::cout << (header_fmt % "Product" % "Quantity" % "Price") << "\n";
    std::cout << sep_line << "\n";
    
    for (const auto& product : products) {
        std::cout << (row_fmt % std::get<0>(product) 
                              % std::get<1>(product) 
                              % std::get<2>(product)) << "\n";
    }
    std::cout << sep_line << "\n";
    
    // Financial formatting
    std::vector<double> amounts = {1234.56, -567.89, 999999.99, 0.01};
    boost::format money_fmt("Amount: %1$+12,.2f");
    
    std::cout << "\nFinancial formatting:\n";
    for (double amount : amounts) {
        std::cout << (money_fmt % amount) << "\n";
    }
    
    // Scientific notation
    std::vector<double> scientific = {0.0000123, 1.23e-8, 1.23e15, 9.87654321};
    boost::format sci_fmt("Value: %1$15.3e");
    
    std::cout << "\nScientific notation:\n";
    for (double value : scientific) {
        std::cout << (sci_fmt % value) << "\n";
    }
}
```

#### Integration with Streams and Internationalization

**Advanced Integration Patterns:**
```cpp
void demonstrate_stream_integration() {
    // Stream integration
    std::ostringstream oss;
    boost::format fmt("Processing item %1% of %2%: %3%");
    
    for (int i = 1; i <= 5; ++i) {
        oss << (fmt % i % 5 % ("Item_" + std::to_string(i))) << "\n";
    }
    
    std::cout << "Stream integration:\n" << oss.str() << "\n";
    
    // Locale-aware formatting
    try {
        std::locale german_locale("de_DE.UTF-8");
        boost::format german_fmt("Preis: %1$,.2f €");
        german_fmt.imbue(german_locale);
        
        std::cout << "German formatting: " << (german_fmt % 1234.56) << "\n";
    } catch (const std::exception& e) {
        std::cout << "German locale not available, using default\n";
        boost::format default_fmt("Price: $%1$,.2f");
        std::cout << "Default formatting: " << (default_fmt % 1234.56) << "\n";
    }
    
    // Custom number formatting
    auto format_bytes = [](size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        double size = static_cast<double>(bytes);
        int unit_index = 0;
        
        while (size >= 1024.0 && unit_index < 4) {
            size /= 1024.0;
            ++unit_index;
        }
        
        boost::format byte_fmt("%1$.2f %2%");
        return (byte_fmt % size % units[unit_index]).str();
    };
    
    std::cout << "\nByte formatting:\n";
    std::cout << "  " << format_bytes(1024) << "\n";
    std::cout << "  " << format_bytes(1048576) << "\n";
    std::cout << "  " << format_bytes(1073741824) << "\n";
}
```

## Real-World Applications and Examples

### Text Processing Pipeline - Complete Implementation

**Professional Text Processing System:**
```cpp
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/tokenizer.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>

class AdvancedTextProcessor {
private:
    // Pre-compiled regex patterns for performance
    boost::regex email_pattern_;
    boost::regex url_pattern_;
    boost::regex phone_pattern_;
    boost::regex html_tag_pattern_;
    boost::regex whitespace_pattern_;
    
public:
    AdvancedTextProcessor() {
        // Compile patterns once for reuse
        email_pattern_ = boost::regex(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
        url_pattern_ = boost::regex(R"(https?://[^\s<>"{}|\\^`\[\]]+)");
        phone_pattern_ = boost::regex(R"(\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b)");
        html_tag_pattern_ = boost::regex(R"(<[^>]*>)");
        whitespace_pattern_ = boost::regex(R"(\s+)");
    }
    
    // Comprehensive text cleaning
    std::string cleanText(const std::string& input) {
        std::string result = input;
        
        // Remove HTML tags
        result = boost::regex_replace(result, html_tag_pattern_, " ");
        
        // Normalize whitespace
        result = boost::regex_replace(result, whitespace_pattern_, " ");
        
        // Trim and clean
        boost::trim(result);
        
        return result;
    }
    
    // Extract structured data
    struct ExtractedData {
        std::vector<std::string> emails;
        std::vector<std::string> urls;
        std::vector<std::string> phones;
        std::unordered_map<std::string, int> word_count;
        int total_words = 0;
        int total_chars = 0;
    };
    
    ExtractedData extractData(const std::string& text) {
        ExtractedData data;
        
        // Extract emails
        boost::sregex_iterator email_iter(text.begin(), text.end(), email_pattern_);
        boost::sregex_iterator email_end;
        for (auto it = email_iter; it != email_end; ++it) {
            data.emails.push_back(it->str());
        }
        
        // Extract URLs
        boost::sregex_iterator url_iter(text.begin(), text.end(), url_pattern_);
        boost::sregex_iterator url_end;
        for (auto it = url_iter; it != url_end; ++it) {
            data.urls.push_back(it->str());
        }
        
        // Extract phone numbers
        boost::sregex_iterator phone_iter(text.begin(), text.end(), phone_pattern_);
        boost::sregex_iterator phone_end;
        for (auto it = phone_iter; it != phone_end; ++it) {
            // Format phone number consistently
            boost::format phone_fmt("(%1%) %2%-%3%");
            std::string formatted_phone = (phone_fmt % it->str(1) % it->str(2) % it->str(3)).str();
            data.phones.push_back(formatted_phone);
        }
        
        // Word frequency analysis
        std::string clean_text = cleanText(text);
        data.total_chars = clean_text.length();
        
        // Tokenize words
        boost::char_separator<char> word_sep(" \t\n\r.,!?;:");
        boost::tokenizer<boost::char_separator<char>> words(clean_text, word_sep);
        
        for (const auto& word : words) {
            if (!word.empty()) {
                std::string lower_word = boost::to_lower_copy(word);
                data.word_count[lower_word]++;
                data.total_words++;
            }
        }
        
        return data;
    }
    
    // Generate comprehensive report
    std::string generateReport(const ExtractedData& data, const std::string& source_name = "Unknown") {
        boost::format report_fmt(
            "Text Analysis Report for: %1%\n"
            "================================%2%\n"
            "Statistics:\n"
            "  Total Characters: %3%\n"
            "  Total Words: %4%\n"
            "  Unique Words: %5%\n"
            "\n"
            "Extracted Information:\n"
            "  Email Addresses: %6%\n"
            "  URLs: %7%\n" 
            "  Phone Numbers: %8%\n"
            "\n"
        );
        
        std::string separator(source_name.length(), '=');
        std::string basic_report = (report_fmt 
            % source_name % separator
            % data.total_chars % data.total_words % data.word_count.size()
            % data.emails.size() % data.urls.size() % data.phones.size()).str();
        
        // Add detailed listings
        if (!data.emails.empty()) {
            basic_report += "Email Addresses Found:\n";
            for (const auto& email : data.emails) {
                basic_report += "  • " + email + "\n";
            }
            basic_report += "\n";
        }
        
        if (!data.urls.empty()) {
            basic_report += "URLs Found:\n";
            for (const auto& url : data.urls) {
                basic_report += "  • " + url + "\n";
            }
            basic_report += "\n";
        }
        
        if (!data.phones.empty()) {
            basic_report += "Phone Numbers Found:\n";
            for (const auto& phone : data.phones) {
                basic_report += "  • " + phone + "\n";
            }
            basic_report += "\n";
        }
        
        // Top 10 most frequent words
        std::vector<std::pair<std::string, int>> word_freq(data.word_count.begin(), data.word_count.end());
        std::sort(word_freq.begin(), word_freq.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        basic_report += "Top 10 Most Frequent Words:\n";
        for (size_t i = 0; i < std::min(size_t(10), word_freq.size()); ++i) {
            boost::format word_fmt("  %1$2d. %2$-15s (%3% occurrences)\n");
            basic_report += (word_fmt % (i + 1) % word_freq[i].first % word_freq[i].second).str();
        }
        
        return basic_report;
    }
};

// Example usage and benchmarking
void demonstrate_complete_text_processing() {
    std::string sample_document = R"(
        <html><body>
        <h1>Company Information</h1>
        <p>Welcome to TechCorp! Contact us at support@techcorp.com or sales@techcorp.com.</p>
        <p>Visit our website at https://www.techcorp.com for more information.</p>
        <p>Call us at (555) 123-4567 or +1-800-555-0199 for immediate assistance.</p>
        <p>Our office is located at 1234 Tech Street, Silicon Valley, CA 94000.</p>
        <p>We specialize in innovative technology solutions and cutting-edge software development.</p>
        <p>Our team of expert developers and engineers work tirelessly to deliver quality products.</p>
        </body></html>
    )";
    
    AdvancedTextProcessor processor;
    
    // Benchmark processing time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto extracted_data = processor.extractData(sample_document);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::string report = processor.generateReport(extracted_data, "Sample Document");
    std::cout << report << std::endl;
    std::cout << "Processing time: " << duration.count() << " microseconds\n";
}
```

### Log File Analysis System

**Production-Ready Log Parser:**
```cpp
class LogAnalyzer {
private:
    boost::regex apache_log_pattern_;
    boost::regex error_log_pattern_;
    
public:
    LogAnalyzer() {
        // Apache Common Log Format
        apache_log_pattern_ = boost::regex(
            R"(^(\S+)\s+\S+\s+\S+\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)\s*"?([^"]*)"?\s*"?([^"]*)"?.*$)"
        );
        
        // Generic error log pattern
        error_log_pattern_ = boost::regex(
            R"((\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.*))"
        );
    }
    
    struct LogEntry {
        std::string ip_address;
        std::string timestamp;
        std::string request;
        int status_code = 0;
        int response_size = 0;
        std::string user_agent;
        std::string referer;
    };
    
    struct LogStatistics {
        std::unordered_map<std::string, int> ip_counts;
        std::unordered_map<int, int> status_counts;
        std::unordered_map<std::string, int> user_agent_counts;
        int total_requests = 0;
        int total_bytes = 0;
        std::vector<std::string> error_messages;
    };
    
    std::vector<LogEntry> parseApacheLogs(const std::string& log_content) {
        std::vector<LogEntry> entries;
        
        boost::char_separator<char> line_sep("\n\r");
        boost::tokenizer<boost::char_separator<char>> lines(log_content, line_sep);
        
        for (const auto& line : lines) {
            if (line.empty()) continue;
            
            boost::smatch match;
            if (boost::regex_match(line, match, apache_log_pattern_)) {
                LogEntry entry;
                entry.ip_address = match[1].str();
                entry.timestamp = match[2].str();
                entry.request = match[3].str();
                entry.status_code = std::stoi(match[4].str());
                
                if (!match[5].str().empty() && match[5].str() != "-") {
                    entry.response_size = std::stoi(match[5].str());
                }
                
                entry.referer = match[6].str();
                entry.user_agent = match[7].str();
                
                entries.push_back(entry);
            }
        }
        
        return entries;
    }
    
    LogStatistics analyzeEntries(const std::vector<LogEntry>& entries) {
        LogStatistics stats;
        
        for (const auto& entry : entries) {
            stats.ip_counts[entry.ip_address]++;
            stats.status_counts[entry.status_code]++;
            stats.user_agent_counts[entry.user_agent]++;
            stats.total_requests++;
            stats.total_bytes += entry.response_size;
            
            // Identify errors
            if (entry.status_code >= 400) {
                boost::format error_fmt("Error %1%: %2% from %3%");
                stats.error_messages.push_back(
                    (error_fmt % entry.status_code % entry.request % entry.ip_address).str()
                );
            }
        }
        
        return stats;
    }
    
    std::string generateLogReport(const LogStatistics& stats) {
        boost::format report_header(
            "Log Analysis Report\n"
            "==================\n"
            "Total Requests: %1%\n"
            "Total Bytes Served: %2$,d\n"
            "Unique IP Addresses: %3%\n"
            "Error Count: %4%\n\n"
        );
        
        std::string report = (report_header 
            % stats.total_requests 
            % stats.total_bytes 
            % stats.ip_counts.size() 
            % stats.error_messages.size()).str();
        
        // Top IPs
        std::vector<std::pair<std::string, int>> top_ips(stats.ip_counts.begin(), stats.ip_counts.end());
        std::sort(top_ips.begin(), top_ips.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        report += "Top 10 IP Addresses:\n";
        for (size_t i = 0; i < std::min(size_t(10), top_ips.size()); ++i) {
            boost::format ip_fmt("  %1$-16s: %2$6d requests\n");
            report += (ip_fmt % top_ips[i].first % top_ips[i].second).str();
        }
        
        // Status code distribution
        report += "\nHTTP Status Code Distribution:\n";
        for (const auto& status : stats.status_counts) {
            boost::format status_fmt("  %1$3d: %2$6d requests\n");
            report += (status_fmt % status.first % status.second).str();
        }
        
        // Recent errors
        if (!stats.error_messages.empty()) {
            report += "\nRecent Errors (last 10):\n";
            size_t start = stats.error_messages.size() > 10 ? 
                          stats.error_messages.size() - 10 : 0;
            for (size_t i = start; i < stats.error_messages.size(); ++i) {
                report += "  • " + stats.error_messages[i] + "\n";
            }
        }
        
        return report;
    }
};
```

### Configuration File Parser

**Robust Configuration Management:**
```cpp
class ConfigurationParser {
private:
    boost::regex section_pattern_;
    boost::regex key_value_pattern_;
    boost::regex comment_pattern_;
    
public:
    ConfigurationParser() {
        section_pattern_ = boost::regex(R"^\[([^\]]+)\]$");
        key_value_pattern_ = boost::regex(R"^([^=]+)=(.*)$");
        comment_pattern_ = boost::regex(R"^[#;].*$");
    }
    
    struct ConfigValue {
        std::string value;
        std::string section;
        
        // Type conversion helpers
        template<typename T>
        T as() const;
        
        bool asBool() const {
            std::string lower_val = boost::to_lower_copy(boost::trim_copy(value));
            return lower_val == "true" || lower_val == "yes" || lower_val == "1" || lower_val == "on";
        }
        
        int asInt() const { return std::stoi(value); }
        double asDouble() const { return std::stod(value); }
        std::string asString() const { return boost::trim_copy(value); }
        
        std::vector<std::string> asList(const std::string& delimiter = ",") const {
            std::vector<std::string> result;
            boost::split(result, value, boost::is_any_of(delimiter));
            
            // Trim each element
            for (auto& item : result) {
                boost::trim(item);
            }
            
            return result;
        }
    };
    
    std::unordered_map<std::string, ConfigValue> parseConfigFile(const std::string& content) {
        std::unordered_map<std::string, ConfigValue> config;
        std::string current_section = "default";
        
        boost::char_separator<char> line_sep("\n\r");
        boost::tokenizer<boost::char_separator<char>> lines(content, line_sep);
        
        for (const auto& line : lines) {
            std::string trimmed_line = boost::trim_copy(line);
            
            // Skip empty lines and comments
            if (trimmed_line.empty() || boost::regex_match(trimmed_line, comment_pattern_)) {
                continue;
            }
            
            boost::smatch match;
            
            // Check for section header
            if (boost::regex_match(trimmed_line, match, section_pattern_)) {
                current_section = match[1].str();
                boost::trim(current_section);
                continue;
            }
            
            // Check for key-value pair
            if (boost::regex_match(trimmed_line, match, key_value_pattern_)) {
                std::string key = boost::trim_copy(match[1].str());
                std::string value = boost::trim_copy(match[2].str());
                
                // Remove quotes if present
                if (value.size() >= 2) {
                    if ((value.front() == '"' && value.back() == '"') ||
                        (value.front() == '\'' && value.back() == '\'')) {
                        value = value.substr(1, value.size() - 2);
                    }
                }
                
                // Create full key with section
                std::string full_key = current_section + "." + key;
                config[full_key] = {value, current_section};
            }
        }
        
        return config;
    }
    
    std::string generateConfigReport(const std::unordered_map<std::string, ConfigValue>& config) {
        std::unordered_map<std::string, std::vector<std::string>> sections;
        
        // Group by sections
        for (const auto& pair : config) {
            sections[pair.second.section].push_back(pair.first);
        }
        
        boost::format header_fmt("Configuration Report\n====================\nTotal settings: %1%\nSections: %2%\n\n");
        std::string report = (header_fmt % config.size() % sections.size()).str();
        
        // Display by sections
        for (const auto& section_pair : sections) {
            boost::format section_fmt("[%1%]\n%2%\n");
            std::string section_separator(section_pair.first.length() + 2, '-');
            
            report += (section_fmt % section_pair.first % section_separator).str();
            
            for (const auto& key : section_pair.second) {
                auto it = config.find(key);
                if (it != config.end()) {
                    std::string simple_key = key.substr(section_pair.first.length() + 1);
                    boost::format key_fmt("  %1$-20s = %2%\n");
                    report += (key_fmt % simple_key % it->second.asString()).str();
                }
            }
            report += "\n";
        }
        
        return report;
    }
};
```

## Practical Exercises and Projects

### Exercise 1: Advanced Log File Parser
**Objective**: Build a comprehensive log analysis tool that can handle multiple log formats.

**Requirements**:
```cpp
// Implement a LogAnalyzer class that can:
// 1. Parse Apache, Nginx, and custom application logs
// 2. Extract IP addresses, timestamps, HTTP methods, status codes
// 3. Generate statistical reports with charts (ASCII art)
// 4. Identify security threats (brute force, injection attempts)
// 5. Handle large files efficiently (streaming)

class LogAnalyzer {
public:
    enum class LogFormat { APACHE, NGINX, CUSTOM, AUTO_DETECT };
    
    struct SecurityThreat {
        std::string type;           // "brute_force", "sql_injection", etc.
        std::string source_ip;
        int frequency;
        std::vector<std::string> sample_requests;
    };
    
    struct LogReport {
        int total_requests;
        std::map<int, int> status_distribution;
        std::map<std::string, int> ip_frequency;
        std::vector<SecurityThreat> threats;
        double requests_per_second;
        std::string time_range;
    };
    
    // TODO: Implement these methods
    LogFormat detectFormat(const std::string& sample_line);
    LogReport analyzeLogFile(const std::string& filename, LogFormat format = LogFormat::AUTO_DETECT);
    std::string generateHTMLReport(const LogReport& report);
    std::vector<SecurityThreat> detectThreats(const std::vector<LogEntry>& entries);
    void streamAnalyze(const std::string& filename, std::function<void(const LogEntry&)> callback);
};
```

**Sample Input**:
```
127.0.0.1 - - [24/Jun/2023:10:30:15 +0000] "GET /index.html HTTP/1.1" 200 2326
127.0.0.1 - - [24/Jun/2023:10:30:16 +0000] "POST /login HTTP/1.1" 401 0
192.168.1.100 - - [24/Jun/2023:10:30:17 +0000] "GET /admin' OR '1'='1 HTTP/1.1" 403 0
```

**Expected Features**:
- Automatic format detection
- Real-time streaming analysis for large files
- Security threat detection
- Performance metrics calculation
- HTML report generation with embedded CSS

### Exercise 2: CSV Data Processor and ETL Pipeline
**Objective**: Create a robust CSV processing system for data cleaning and transformation.

**Requirements**:
```cpp
class CSVProcessor {
public:
    struct ColumnInfo {
        std::string name;
        enum Type { STRING, INTEGER, DOUBLE, BOOLEAN, DATE } type;
        bool required = false;
        std::string default_value;
        std::function<bool(const std::string&)> validator;
    };
    
    struct ProcessingRule {
        std::string column_name;
        enum Action { TRIM, UPPER, LOWER, REPLACE, VALIDATE, TRANSFORM } action;
        std::string parameter;  // For replace: "old,new", for transform: custom function
    };
    
    struct ValidationResult {
        bool is_valid;
        std::vector<std::string> errors;
        int processed_rows;
        int error_rows;
    };
    
    // TODO: Implement these methods
    bool loadSchema(const std::string& schema_json);
    ValidationResult processCSV(const std::string& input_file, const std::string& output_file);
    void addProcessingRule(const ProcessingRule& rule);
    std::string generateQualityReport(const ValidationResult& result);
    bool exportToJSON(const std::string& csv_file, const std::string& json_file);
    bool exportToXML(const std::string& csv_file, const std::string& xml_file);
};
```

**Sample Schema** (JSON):
```json
{
  "columns": [
    {"name": "id", "type": "INTEGER", "required": true},
    {"name": "email", "type": "STRING", "required": true, "validator": "email"},
    {"name": "age", "type": "INTEGER", "default": "0"},
    {"name": "salary", "type": "DOUBLE", "validator": "positive"}
  ],
  "rules": [
    {"column": "email", "action": "LOWER"},
    {"column": "name", "action": "TRIM"},
    {"column": "phone", "action": "REPLACE", "parameter": "[^0-9],,"}
  ]
}
```

### Exercise 3: Text Template Engine
**Objective**: Build a flexible template system for generating dynamic content.

**Requirements**:
```cpp
class TemplateEngine {
public:
    // Template syntax: {{variable}}, {{#if condition}}, {{#for item in list}}
    struct TemplateContext {
        std::unordered_map<std::string, std::string> variables;
        std::unordered_map<std::string, std::vector<std::string>> lists;
        std::unordered_map<std::string, bool> conditions;
    };
    
    // TODO: Implement these methods
    std::string loadTemplate(const std::string& template_file);
    std::string renderTemplate(const std::string& template_str, const TemplateContext& context);
    bool validateTemplate(const std::string& template_str, std::vector<std::string>& errors);
    void registerHelper(const std::string& name, std::function<std::string(const std::vector<std::string>&)> helper);
    void registerFilter(const std::string& name, std::function<std::string(const std::string&)> filter);
    
private:
    // Helper methods for parsing
    std::string processVariables(const std::string& text, const TemplateContext& context);
    std::string processConditionals(const std::string& text, const TemplateContext& context);
    std::string processLoops(const std::string& text, const TemplateContext& context);
};
```

**Sample Template**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{page_title}}</title>
</head>
<body>
    <h1>Welcome {{user_name}}!</h1>
    
    {{#if show_products}}
    <h2>Our Products:</h2>
    <ul>
    {{#for product in products}}
        <li>{{product}} - ${{price|currency}}</li>
    {{/for}}
    </ul>
    {{/if}}
    
    {{#if show_footer}}
    <footer>
        <p>© 2023 {{company_name}}. Generated on {{current_date|date_format}}</p>
    </footer>
    {{/if}}
</body>
</html>
```

### Exercise 4: Natural Language Processing Toolkit
**Objective**: Create a basic NLP toolkit for text analysis and processing.

**Requirements**:
```cpp
class NLPToolkit {
public:
    struct TextStatistics {
        int character_count;
        int word_count;
        int sentence_count;
        int paragraph_count;
        double avg_word_length;
        double avg_sentence_length;
        int unique_words;
        std::map<std::string, int> word_frequency;
        std::vector<std::string> most_common_words;
    };
    
    struct SentimentResult {
        enum Sentiment { POSITIVE, NEGATIVE, NEUTRAL } sentiment;
        double confidence;
        std::vector<std::string> positive_words;
        std::vector<std::string> negative_words;
    };
    
    struct NamedEntity {
        std::string text;
        enum Type { PERSON, ORGANIZATION, LOCATION, DATE, EMAIL, PHONE } type;
        size_t start_position;
        size_t end_position;
    };
    
    // TODO: Implement these methods
    TextStatistics analyzeText(const std::string& text);
    SentimentResult analyzeSentiment(const std::string& text);
    std::vector<NamedEntity> extractNamedEntities(const std::string& text);
    std::vector<std::string> extractKeywords(const std::string& text, int max_keywords = 10);
    std::string summarizeText(const std::string& text, int max_sentences = 3);
    double calculateReadabilityScore(const std::string& text);  // Flesch-Kincaid
    std::vector<std::string> detectLanguages(const std::string& text);
    
private:
    void loadSentimentDictionary(const std::string& positive_file, const std::string& negative_file);
    std::vector<std::string> tokenizeSentences(const std::string& text);
    std::vector<std::string> tokenizeWords(const std::string& text);
    bool isStopWord(const std::string& word);
};
```

## Performance Considerations and Optimization

### Regex Performance Optimization

**Pattern Compilation Strategy:**
```cpp
class OptimizedRegexProcessor {
private:
    // Pre-compiled patterns for reuse
    static const boost::regex email_regex_;
    static const boost::regex url_regex_;
    static const boost::regex phone_regex_;
    
    // Pattern cache for dynamic patterns
    mutable std::unordered_map<std::string, boost::regex> pattern_cache_;
    mutable std::mutex cache_mutex_;
    
public:
    // Compile-time pattern optimization
    template<typename Pattern>
    static constexpr auto compile_pattern(Pattern&& pattern) {
        return boost::regex(std::forward<Pattern>(pattern), 
                           boost::regex::optimize | boost::regex::nosubs);
    }
    
    // Runtime pattern caching
    const boost::regex& getOrCompilePattern(const std::string& pattern) const {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = pattern_cache_.find(pattern);
        if (it != pattern_cache_.end()) {
            return it->second;
        }
        
        // Compile with optimization flags
        auto result = pattern_cache_.emplace(pattern, 
            boost::regex(pattern, boost::regex::optimize));
        return result.first->second;
    }
};

// Performance benchmarking
void benchmark_regex_performance() {
    const std::string large_text = /* load large text file */;
    const int iterations = 1000;
    
    // Benchmark 1: Pattern recompilation (BAD)
    auto start_bad = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        boost::regex pattern(R"(\b\w+@\w+\.\w+\b)");  // Recompile each time
        boost::sregex_iterator iter(large_text.begin(), large_text.end(), pattern);
        int count = std::distance(iter, boost::sregex_iterator());
    }
    auto end_bad = std::chrono::high_resolution_clock::now();
    
    // Benchmark 2: Pattern reuse (GOOD)
    boost::regex compiled_pattern(R"(\b\w+@\w+\.\w+\b)", boost::regex::optimize);
    auto start_good = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        boost::sregex_iterator iter(large_text.begin(), large_text.end(), compiled_pattern);
        int count = std::distance(iter, boost::sregex_iterator());
    }
    auto end_good = std::chrono::high_resolution_clock::now();
    
    std::cout << "Pattern recompilation: " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_bad - start_bad).count() << "ms\n";
    std::cout << "Pattern reuse: " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(end_good - start_good).count() << "ms\n";
}
```

### String Algorithm Efficiency

**Memory-Efficient String Processing:**
```cpp
class EfficientStringProcessor {
public:
    // In-place operations when possible
    void processLargeText(std::string& text) {
        // Reserve capacity to avoid reallocations
        text.reserve(text.size() * 1.2);  // Assume 20% growth
        
        // In-place trimming
        boost::trim(text);
        
        // In-place case conversion
        boost::to_lower(text);
        
        // Batch replacements to minimize passes
        std::vector<std::pair<std::string, std::string>> replacements = {
            {"\\t", " "}, {"\\n", " "}, {"\\r", " "}, {"  ", " "}
        };
        
        for (const auto& replacement : replacements) {
            boost::replace_all(text, replacement.first, replacement.second);
        }
    }
    
    // Stream processing for very large files
    void processLargeFile(const std::string& input_file, const std::string& output_file) {
        std::ifstream input(input_file);
        std::ofstream output(output_file);
        
        std::string line;
        line.reserve(1024);  // Reserve typical line size
        
        boost::regex pattern(R"(\b\w+@\w+\.\w+\b)");
        
        while (std::getline(input, line)) {
            // Process line in chunks to avoid memory spikes
            if (line.length() > 10000) {
                processLineInChunks(line, pattern, output);
            } else {
                std::string processed = processLine(line, pattern);
                output << processed << "\n";
            }
        }
    }
    
private:
    void processLineInChunks(const std::string& line, const boost::regex& pattern, std::ofstream& output) {
        const size_t chunk_size = 8192;
        for (size_t i = 0; i < line.length(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, line.length());
            std::string chunk = line.substr(i, end - i);
            std::string processed = processLine(chunk, pattern);
            output << processed;
        }
        output << "\n";
    }
    
    std::string processLine(const std::string& line, const boost::regex& pattern) {
        // Implement line processing logic
        return boost::regex_replace(line, pattern, "[EMAIL]");
    }
};
```

### Memory Management for Large Text Processing

**Smart Memory Usage Patterns:**
```cpp
class MemoryEfficientTextProcessor {
public:
    // Use string views when possible (C++17)
    std::vector<std::string_view> tokenizeView(std::string_view text, char delimiter) {
        std::vector<std::string_view> tokens;
        size_t start = 0;
        size_t end = text.find(delimiter);
        
        while (end != std::string_view::npos) {
            tokens.emplace_back(text.substr(start, end - start));
            start = end + 1;
            end = text.find(delimiter, start);
        }
        
        if (start < text.length()) {
            tokens.emplace_back(text.substr(start));
        }
        
        return tokens;
    }
    
    // Memory pool for frequent small allocations
    class StringPool {
    private:
        std::vector<std::unique_ptr<char[]>> pools_;
        size_t current_pool_size_ = 4096;
        size_t current_offset_ = 0;
        
    public:
        std::string_view allocateString(const std::string& str) {
            if (current_offset_ + str.length() + 1 > current_pool_size_) {
                // Allocate new pool
                current_pool_size_ = std::max(current_pool_size_ * 2, str.length() + 1);
                pools_.emplace_back(std::make_unique<char[]>(current_pool_size_));
                current_offset_ = 0;
            }
            
            char* ptr = pools_.back().get() + current_offset_;
            std::memcpy(ptr, str.c_str(), str.length() + 1);
            current_offset_ += str.length() + 1;
            
            return std::string_view(ptr, str.length());
        }
    };
    
    // Lazy evaluation for expensive operations
    class LazyTextAnalysis {
    private:
        std::string text_;
        mutable std::optional<int> word_count_;
        mutable std::optional<std::map<std::string, int>> word_frequency_;
        
    public:
        explicit LazyTextAnalysis(std::string text) : text_(std::move(text)) {}
        
        int getWordCount() const {
            if (!word_count_) {
                word_count_ = calculateWordCount();
            }
            return *word_count_;
        }
        
        const std::map<std::string, int>& getWordFrequency() const {
            if (!word_frequency_) {
                word_frequency_ = calculateWordFrequency();
            }
            return *word_frequency_;
        }
        
    private:
        int calculateWordCount() const {
            boost::char_separator<char> sep(" \t\n\r");
            boost::tokenizer<boost::char_separator<char>> tokens(text_, sep);
            return std::distance(tokens.begin(), tokens.end());
        }
        
        std::map<std::string, int> calculateWordFrequency() const {
            std::map<std::string, int> frequency;
            boost::char_separator<char> sep(" \t\n\r");
            boost::tokenizer<boost::char_separator<char>> tokens(text_, sep);
            
            for (const auto& token : tokens) {
                std::string word = boost::to_lower_copy(token);
                frequency[word]++;
            }
            
            return frequency;
        }
    };
};
```

## Best Practices and Design Patterns

### Regular Expression Design Principles

**1. Specificity and Precision:**
```cpp
// BAD: Too generic, matches too much
boost::regex bad_email(R"(.+@.+)");

// GOOD: Specific and accurate
boost::regex good_email(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4})");

// BETTER: Handle edge cases
boost::regex better_email(R"(^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$)");
```

**2. Performance-Oriented Pattern Design:**
```cpp
class PatternDesignBestPractices {
public:
    // Use anchors to avoid unnecessary backtracking
    static const boost::regex anchored_pattern_;
    static const boost::regex unanchored_pattern_;
    
    void demonstrateAnchoringPerformance() {
        std::string text = "This is a long text with an email address user@example.com at the end";
        
        // Without anchors - scans entire string
        boost::regex slow_pattern(R"(\w+@\w+\.\w+)");
        
        // With word boundaries - more efficient
        boost::regex fast_pattern(R"(\b\w+@\w+\.\w+\b)");
        
        // Benchmark the difference
        auto benchmark = [&](const boost::regex& pattern, const std::string& name) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10000; ++i) {
                boost::smatch match;
                boost::regex_search(text, match, pattern);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << name << ": " << duration.count() << " microseconds\n";
        };
        
        benchmark(slow_pattern, "Without boundaries");
        benchmark(fast_pattern, "With boundaries");
    }
    
    // Use non-capturing groups when captures aren't needed
    boost::regex optimized_pattern() {
        // BAD: Unnecessary capturing groups
        boost::regex bad(R"((\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}))");
        
        // GOOD: Non-capturing groups for better performance
        return boost::regex(R"((?:\d{4})-(?:\d{2})-(?:\d{2}) (?:\d{2}):(?:\d{2}):(?:\d{2}))");
    }
    
    // Atomic groups to prevent catastrophic backtracking
    boost::regex atomic_groups_pattern() {
        // Prevents catastrophic backtracking on malformed input
        return boost::regex(R"((?>a+)a*b)");
    }
};
```

**3. Internationalization and Unicode Support:**
```cpp
class InternationalTextProcessor {
public:
    // Handle Unicode properly
    void processUnicodeText() {
        std::string utf8_text = u8"Café, naïve, résumé, 北京, Москва";
        
        // Use Unicode-aware patterns
        boost::regex unicode_word(R"(\b\w+\b)");
        
        try {
            boost::sregex_iterator start(utf8_text.begin(), utf8_text.end(), unicode_word);
            boost::sregex_iterator end;
            
            std::cout << "Unicode words found:\n";
            for (auto it = start; it != end; ++it) {
                std::cout << "  '" << it->str() << "'\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Unicode processing error: " << e.what() << "\n";
        }
    }
    
    // Locale-aware string operations
    void localeAwareProcessing() {
        try {
            // Set locale for proper sorting and comparison
            std::locale turkish_locale("tr_TR.UTF-8");
            
            std::vector<std::string> words = {"İstanbul", "ankara", "İzmir", "bursa"};
            
            // Sort using Turkish collation rules
            std::sort(words.begin(), words.end(), 
                [&](const std::string& a, const std::string& b) {
                    return std::use_facet<std::collate<char>>(turkish_locale)
                           .compare(a.data(), a.data() + a.size(),
                                   b.data(), b.data() + b.size()) < 0;
                });
                
            std::cout << "Turkish-sorted words:\n";
            for (const auto& word : words) {
                std::cout << "  " << word << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Locale not available, using default sorting\n";
        }
    }
};
```

### String Processing Pipeline Design

**1. Builder Pattern for Complex Processing:**
```cpp
class TextProcessingPipeline {
public:
    class Builder {
    private:
        std::vector<std::function<std::string(const std::string&)>> processors_;
        
    public:
        Builder& trim() {
            processors_.emplace_back([](const std::string& text) {
                return boost::trim_copy(text);
            });
            return *this;
        }
        
        Builder& toLowerCase() {
            processors_.emplace_back([](const std::string& text) {
                return boost::to_lower_copy(text);
            });
            return *this;
        }
        
        Builder& replaceAll(const std::string& from, const std::string& to) {
            processors_.emplace_back([from, to](const std::string& text) {
                std::string result = text;
                boost::replace_all(result, from, to);
                return result;
            });
            return *this;
        }
        
        Builder& removeHtmlTags() {
            processors_.emplace_back([](const std::string& text) {
                static const boost::regex html_tags(R"(<[^>]*>)");
                return boost::regex_replace(text, html_tags, "");
            });
            return *this;
        }
        
        Builder& normalizeWhitespace() {
            processors_.emplace_back([](const std::string& text) {
                static const boost::regex multi_space(R"(\s+)");
                return boost::regex_replace(text, multi_space, " ");
            });
            return *this;
        }
        
        Builder& customProcessor(std::function<std::string(const std::string&)> processor) {
            processors_.emplace_back(std::move(processor));
            return *this;
        }
        
        TextProcessingPipeline build() {
            return TextProcessingPipeline(std::move(processors_));
        }
    };
    
    std::string process(const std::string& input) const {
        std::string result = input;
        for (const auto& processor : processors_) {
            result = processor(result);
        }
        return result;
    }
    
    // Batch processing with progress callback
    std::vector<std::string> processBatch(const std::vector<std::string>& inputs,
                                         std::function<void(size_t, size_t)> progress_callback = nullptr) const {
        std::vector<std::string> results;
        results.reserve(inputs.size());
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            results.push_back(process(inputs[i]));
            
            if (progress_callback && (i + 1) % 100 == 0) {
                progress_callback(i + 1, inputs.size());
            }
        }
        
        return results;
    }
    
private:
    explicit TextProcessingPipeline(std::vector<std::function<std::string(const std::string&)>> processors)
        : processors_(std::move(processors)) {}
    
    std::vector<std::function<std::string(const std::string&)>> processors_;
};

// Usage example
void demonstrate_pipeline_usage() {
    auto pipeline = TextProcessingPipeline::Builder()
        .removeHtmlTags()
        .normalizeWhitespace()
        .trim()
        .toLowerCase()
        .replaceAll("&nbsp;", " ")
        .customProcessor([](const std::string& text) {
            // Custom processing logic
            return boost::regex_replace(text, boost::regex(R"(\b\d+\b)"), "[NUMBER]");
        })
        .build();
    
    std::string html_text = "<p>Hello <b>World</b>! &nbsp; The year is 2023.</p>";
    std::string result = pipeline.process(html_text);
    
    std::cout << "Original: " << html_text << "\n";
    std::cout << "Processed: " << result << "\n";
}
```

**2. Strategy Pattern for Different Processing Approaches:**
```cpp
class TextProcessingStrategy {
public:
    virtual ~TextProcessingStrategy() = default;
    virtual std::string process(const std::string& text) = 0;
    virtual std::string getName() const = 0;
};

class HTMLCleaningStrategy : public TextProcessingStrategy {
public:
    std::string process(const std::string& text) override {
        static const boost::regex html_pattern(R"(<[^>]*>)");
        std::string result = boost::regex_replace(text, html_pattern, "");
        
        // Decode HTML entities
        boost::replace_all(result, "&lt;", "<");
        boost::replace_all(result, "&gt;", ">");
        boost::replace_all(result, "&amp;", "&");
        boost::replace_all(result, "&quot;", "\"");
        boost::replace_all(result, "&#39;", "'");
        boost::replace_all(result, "&nbsp;", " ");
        
        return boost::trim_copy(result);
    }
    
    std::string getName() const override { return "HTML Cleaning"; }
};

class MarkdownCleaningStrategy : public TextProcessingStrategy {
public:
    std::string process(const std::string& text) override {
        std::string result = text;
        
        // Remove markdown formatting
        static const std::vector<std::pair<boost::regex, std::string>> markdown_patterns = {
            {boost::regex(R"(\*\*([^*]+)\*\*)"), "$1"},      // Bold
            {boost::regex(R"(\*([^*]+)\*)"), "$1"},          // Italic
            {boost::regex(R"(`([^`]+)`)"), "$1"},            // Code
            {boost::regex(R"(\[([^\]]+)\]\([^)]+\))"), "$1"}, // Links
            {boost::regex(R"(^#{1,6}\s*)"), ""},             // Headers
            {boost::regex(R"(^\s*[-*+]\s*)"), ""},           // List items
        };
        
        for (const auto& pattern : markdown_patterns) {
            result = boost::regex_replace(result, pattern.first, pattern.second);
        }
        
        return boost::trim_copy(result);
    }
    
    std::string getName() const override { return "Markdown Cleaning"; }
};

class TextProcessingContext {
private:
    std::unique_ptr<TextProcessingStrategy> strategy_;
    
public:
    void setStrategy(std::unique_ptr<TextProcessingStrategy> strategy) {
        strategy_ = std::move(strategy);
    }
    
    std::string processText(const std::string& text) {
        if (!strategy_) {
            throw std::runtime_error("No processing strategy set");
        }
        
        std::cout << "Using strategy: " << strategy_->getName() << "\n";
        return strategy_->process(text);
    }
};
```

### Format String Safety and Validation

**1. Compile-Time Format String Validation:**
```cpp
template<typename... Args>
class SafeFormatter {
private:
    boost::format format_;
    static constexpr size_t expected_args = sizeof...(Args);
    
public:
    explicit SafeFormatter(const std::string& format_str) : format_(format_str) {
        // Validate format string at construction
        validateFormat(format_str);
    }
    
    std::string operator()(Args... args) const {
        static_assert(sizeof...(args) == expected_args, 
                     "Argument count mismatch with template parameters");
        
        boost::format fmt = format_;
        return applyArgs(fmt, args...).str();
    }
    
private:
    void validateFormat(const std::string& format_str) {
        // Count placeholders
        boost::regex placeholder_pattern(R"(%\d+%)");
        boost::sregex_iterator start(format_str.begin(), format_str.end(), placeholder_pattern);
        boost::sregex_iterator end;
        
        size_t placeholder_count = std::distance(start, end);
        if (placeholder_count != expected_args) {
            throw std::invalid_argument(
                "Format string expects " + std::to_string(placeholder_count) + 
                " arguments, but " + std::to_string(expected_args) + " provided");
        }
    }
    
    template<typename T>
    boost::format& applyArgs(boost::format& fmt, T&& arg) const {
        return fmt % std::forward<T>(arg);
    }
    
    template<typename T, typename... Rest>
    boost::format& applyArgs(boost::format& fmt, T&& first, Rest&&... rest) const {
        return applyArgs(fmt % std::forward<T>(first), std::forward<Rest>(rest)...);
    }
};

// Usage with compile-time safety
void demonstrate_safe_formatting() {
    // This will compile and work correctly
    SafeFormatter<std::string, int> name_age_formatter("Hello %1%, you are %2% years old!");
    std::cout << name_age_formatter("Alice", 25) << "\n";
    
    // This would cause a compile-time error:
    // SafeFormatter<std::string> wrong_formatter("Hello %1%, you are %2% years old!");
    // std::cout << wrong_formatter("Alice") << "\n";  // Too few arguments
}
```

**2. Runtime Format Validation and Error Handling:**
```cpp
class RobustFormatter {
public:
    struct FormatResult {
        bool success;
        std::string result;
        std::string error_message;
    };
    
    template<typename... Args>
    static FormatResult formatSafe(const std::string& format_str, Args&&... args) {
        try {
            boost::format fmt(format_str);
            
            // Apply arguments with error checking
            auto result = applyArgsSafe(fmt, std::forward<Args>(args)...);
            if (!result.success) {
                return result;
            }
            
            return {true, fmt.str(), ""};
            
        } catch (const boost::io::format_error& e) {
            return {false, "", "Format error: " + std::string(e.what())};
        } catch (const std::exception& e) {
            return {false, "", "Error: " + std::string(e.what())};
        }
    }
    
private:
    template<typename T>
    static FormatResult applyArgsSafe(boost::format& fmt, T&& arg) {
        try {
            fmt % std::forward<T>(arg);
            return {true, "", ""};
        } catch (const std::exception& e) {
            return {false, "", "Argument application error: " + std::string(e.what())};
        }
    }
    
    template<typename T, typename... Rest>
    static FormatResult applyArgsSafe(boost::format& fmt, T&& first, Rest&&... rest) {
        auto result = applyArgsSafe(fmt, std::forward<T>(first));
        if (!result.success) {
            return result;
        }
        
        return applyArgsSafe(fmt, std::forward<Rest>(rest)...);
    }
};

// Usage with runtime safety
void demonstrate_robust_formatting() {
    auto result1 = RobustFormatter::formatSafe("Hello %1%, you are %2% years old!", "Bob", 30);
    if (result1.success) {
        std::cout << result1.result << "\n";
    } else {
        std::cout << "Format error: " << result1.error_message << "\n";
    }
    
    // This will handle the error gracefully
    auto result2 = RobustFormatter::formatSafe("Hello %1%, you are %2% years old!", "Alice");
    if (!result2.success) {
        std::cout << "Expected error: " << result2.error_message << "\n";
    }
}
```

## Learning Assessment and Mastery Validation

### Self-Assessment Checklist

Before proceeding to the next section, ensure you can confidently:

**Basic Understanding:**
□ Explain the differences between Boost.Regex and std::regex  
□ Create and compile regular expression patterns efficiently  
□ Use basic string algorithms for trimming, case conversion, and splitting  
□ Implement simple tokenization for CSV and delimited data  
□ Apply basic formatting with Boost.Format  

**Intermediate Skills:**
□ Design complex regex patterns with named captures and backreferences  
□ Optimize regex performance through pattern compilation and caching  
□ Build robust CSV parsers handling edge cases and validation  
□ Create flexible tokenization strategies for different data formats  
□ Implement type-safe formatting with positional parameters  
□ Handle Unicode and internationalization in text processing  

**Advanced Competencies:**
□ Build complete text processing pipelines with error handling  
□ Implement streaming text processing for large files  
□ Design pattern-based text processors using Strategy pattern  
□ Create memory-efficient text processing systems  
□ Build domain-specific parsers (logs, configuration files, etc.)  
□ Integrate multiple Boost string libraries in complex applications  

### Practical Skills Validation

**Code Challenge 1: Email Validation System**
```cpp
// Implement a comprehensive email validation system
class EmailValidator {
public:
    enum class ValidationLevel { BASIC, STANDARD, STRICT };
    
    struct ValidationResult {
        bool is_valid;
        std::string normalized_email;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
    };
    
    // TODO: Implement these methods
    ValidationResult validate(const std::string& email, ValidationLevel level = ValidationLevel::STANDARD);
    std::string normalizeEmail(const std::string& email);
    bool isDisposableEmail(const std::string& email);
    std::vector<std::string> extractEmailsFromText(const std::string& text);
    std::string suggestCorrection(const std::string& invalid_email);
};

// Expected behavior:
// - Handle different email formats and edge cases
// - Normalize emails (lowercase, remove dots from Gmail, etc.)
// - Detect disposable email providers
// - Suggest corrections for common typos
// - Extract emails from free-form text
```

**Code Challenge 2: Configuration File Parser**
```cpp
// Build a robust configuration parser supporting multiple formats
class ConfigParser {
public:
    enum class Format { INI, JSON, YAML, TOML, AUTO_DETECT };
    
    struct ConfigSection {
        std::string name;
        std::map<std::string, std::string> values;
        std::vector<ConfigSection> subsections;
    };
    
    // TODO: Implement complete configuration system
    bool loadFromFile(const std::string& filename, Format format = Format::AUTO_DETECT);
    bool loadFromString(const std::string& content, Format format);
    std::string getValue(const std::string& key, const std::string& default_value = "");
    bool setValue(const std::string& key, const std::string& value);
    bool saveToFile(const std::string& filename, Format format);
    std::vector<std::string> validateConfiguration(const std::map<std::string, std::string>& schema);
};

// Test with this configuration:
/*
[database]
host = localhost
port = 5432
username = admin
password = "secret with spaces"
ssl_enabled = true

[logging]
level = INFO
file = "/var/log/app.log"
max_size = 10MB
*/
```

**Code Challenge 3: Text Analysis Engine**
```cpp
// Create a comprehensive text analysis system
class TextAnalysisEngine {
public:
    struct AnalysisReport {
        // Basic statistics
        int character_count;
        int word_count;
        int sentence_count;
        int paragraph_count;
        
        // Language analysis
        double readability_score;
        std::string detected_language;
        std::vector<std::string> keywords;
        
        // Content analysis
        std::map<std::string, int> word_frequency;
        std::vector<std::string> named_entities;
        std::vector<std::string> email_addresses;
        std::vector<std::string> urls;
        std::vector<std::string> phone_numbers;
        
        // Sentiment analysis
        enum Sentiment { POSITIVE, NEGATIVE, NEUTRAL } sentiment;
        double sentiment_confidence;
    };
    
    // TODO: Build complete analysis system
    AnalysisReport analyzeText(const std::string& text);
    std::string generateHTMLReport(const AnalysisReport& report);
    bool exportToJSON(const AnalysisReport& report, const std::string& filename);
    std::vector<std::string> extractKeyPhrases(const std::string& text, int max_phrases = 10);
    std::string summarizeText(const std::string& text, int max_sentences = 3);
};
```

### Knowledge Transfer Projects

**Project 1: Web Log Analyzer**
Build a production-ready web server log analyzer that can:
- Parse multiple log formats (Apache, Nginx, IIS)
- Generate statistical reports with charts
- Detect security threats and anomalies
- Export results in multiple formats (HTML, JSON, CSV)
- Handle large log files efficiently

**Project 2: Document Processing System**
Create a document processing pipeline that can:
- Extract text from various formats (HTML, Markdown, plain text)
- Clean and normalize content
- Extract structured data (contacts, dates, numbers)
- Generate summaries and keyword extraction
- Support batch processing with progress tracking

**Project 3: Configuration Management Tool**
Develop a configuration management system that:
- Supports multiple configuration formats
- Provides schema validation
- Offers environment-specific configurations
- Includes template processing capabilities
- Features a command-line interface for management operations

### Performance Benchmarking Exercises

**Exercise 1: Regex Performance Analysis**
Compare the performance of different regex approaches:
```cpp
// Benchmark these scenarios:
// 1. Pattern recompilation vs. pattern caching
// 2. Greedy vs. non-greedy matching
// 3. Anchored vs. unanchored patterns
// 4. Capturing vs. non-capturing groups
// 5. Different regex engines (Boost vs. std::regex)

void benchmark_regex_performance();
```

**Exercise 2: String Processing Optimization**
Optimize text processing for large datasets:
```cpp
// Test these optimization techniques:
// 1. In-place operations vs. copy operations
// 2. Memory pre-allocation strategies
// 3. String view usage for read-only operations
// 4. Batch processing vs. individual processing
// 5. Memory pooling for frequent allocations

void benchmark_string_processing();
```

**Exercise 3: Memory Usage Analysis**
Analyze memory usage patterns:
```cpp
// Profile memory usage for:
// 1. Large text file processing
// 2. Regex pattern compilation and storage
// 3. String tokenization and storage
// 4. Format string caching
// 5. Unicode text processing

void analyze_memory_usage();
```

### Integration with Real-World Systems

**Database Integration Example:**
```cpp
class DatabaseTextProcessor {
public:
    // Process text data from database
    void processTextColumns(const std::string& connection_string, 
                           const std::string& table_name,
                           const std::vector<std::string>& text_columns);
    
    // Clean and normalize database text
    void cleanDatabaseText(const std::string& connection_string,
                          const std::map<std::string, std::string>& table_column_map);
    
    // Extract structured data and store in separate tables
    void extractStructuredData(const std::string& connection_string,
                              const std::string& source_table,
                              const std::string& target_table);
};
```

**Web Service Integration Example:**
```cpp
class WebServiceTextProcessor {
public:
    // REST API for text processing
    std::string processTextAPI(const std::string& input_text, 
                              const std::string& operation);
    
    // Batch processing endpoint
    std::string processBatchAPI(const std::vector<std::string>& texts,
                               const std::string& operation);
    
    // Streaming processing for large uploads
    void processStreamAPI(std::istream& input, std::ostream& output,
                         const std::string& operation);
};
```

## Next Steps and Advanced Topics

### Recommended Learning Path

1. **Master the Fundamentals** (Week 1-2)
   - Complete all basic exercises
   - Understand performance implications
   - Practice with different data formats

2. **Build Real Projects** (Week 3-4)
   - Implement one major project from the suggestions
   - Focus on error handling and edge cases
   - Add comprehensive testing

3. **Performance Optimization** (Week 5)
   - Profile your implementations
   - Optimize for memory and speed
   - Compare with alternative approaches

4. **Integration and Deployment** (Week 6)
   - Integrate with real systems
   - Add monitoring and logging
   - Create documentation and examples

### Advanced Topics to Explore

**1. Unicode and Internationalization**
- ICU library integration
- Unicode normalization
- Locale-specific text processing
- Character encoding conversion

**2. Machine Learning Integration**
- Text classification
- Named entity recognition
- Sentiment analysis with ML models
- Document clustering and similarity

**3. High-Performance Computing**
- Parallel text processing
- GPU-accelerated string operations
- Distributed text processing
- Memory-mapped file processing

**4. Domain-Specific Applications**
- Bioinformatics sequence analysis
- Financial data parsing
- Legal document processing
- Medical text analysis

### Preparation for Next Module

Before moving to **Date and Time Utilities**, ensure you have:

□ Completed at least 2 major projects  
□ Achieved satisfactory performance benchmarks  
□ Implemented error handling and validation  
□ Created comprehensive test suites  
□ Documented your implementations  
□ Reviewed and optimized code for production use  

**Transition Skills Required:**
- String-to-date parsing using regex patterns
- Formatted output generation for temporal data
- Text processing for log timestamps
- Configuration parsing for time-based settings

The skills learned in this module will directly support the next module's focus on temporal data processing and formatting.

## Next Steps

Move on to [Date and Time Utilities](05_Date_Time_Utilities.md) to explore Boost's temporal processing capabilities.
