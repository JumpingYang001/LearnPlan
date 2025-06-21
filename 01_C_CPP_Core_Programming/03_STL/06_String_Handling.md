# String Handling

*Part of STL Learning Track - 1 week*

## Overview

String handling in C++ involves multiple classes and utilities provided by the STL. The main components include std::string, std::wstring, std::string_view (C++17), and various string algorithms. This module covers comprehensive string manipulation, processing, and optimization techniques.

## std::string and std::wstring

### Basic String Operations
```cpp
#include <string>
#include <iostream>
#include <algorithm>

void basic_string_operations() {
    // Construction
    std::string str1;                          // Empty string
    std::string str2("Hello World");           // From C-string
    std::string str3(10, 'A');                 // 10 'A' characters
    std::string str4(str2, 6);                 // Substring from position 6
    std::string str5(str2, 0, 5);              // Substring "Hello"
    
    std::cout << "str1: '" << str1 << "'" << std::endl;
    std::cout << "str2: '" << str2 << "'" << std::endl;
    std::cout << "str3: '" << str3 << "'" << std::endl;
    std::cout << "str4: '" << str4 << "'" << std::endl;
    std::cout << "str5: '" << str5 << "'" << std::endl;
    
    // Assignment
    str1 = "New value";
    str1 += " appended";
    
    std::cout << "After assignment and append: " << str1 << std::endl;
    
    // Size and capacity
    std::cout << "Length: " << str2.length() << std::endl;
    std::cout << "Size: " << str2.size() << std::endl;
    std::cout << "Capacity: " << str2.capacity() << std::endl;
    std::cout << "Empty: " << str1.empty() << std::endl;
    
    // Element access
    std::cout << "First character: " << str2[0] << std::endl;
    std::cout << "First character (safe): " << str2.at(0) << std::endl;
    std::cout << "Last character: " << str2.back() << std::endl;
    
    // Modification
    str2.push_back('!');
    str2.pop_back();
    str2.insert(5, " Beautiful");
    str2.erase(5, 10); // Remove " Beautiful"
    
    std::cout << "After modifications: " << str2 << std::endl;
}
```

### String Comparison and Searching
```cpp
#include <string>
#include <iostream>

void string_comparison_searching() {
    std::string str1 = "Hello World";
    std::string str2 = "Hello";
    std::string str3 = "World";
    
    // Comparison
    std::cout << "str1 == str2: " << (str1 == str2) << std::endl;
    std::cout << "str1.compare(str2): " << str1.compare(str2) << std::endl;
    std::cout << "str1 < str2: " << (str1 < str2) << std::endl;
    
    // Finding
    size_t pos = str1.find("World");
    if (pos != std::string::npos) {
        std::cout << "'World' found at position: " << pos << std::endl;
    }
    
    // Find from position
    pos = str1.find('o', 5); // Find 'o' starting from position 5
    std::cout << "'o' found at position: " << pos << std::endl;
    
    // Reverse find
    pos = str1.rfind('o');
    std::cout << "Last 'o' found at position: " << pos << std::endl;
    
    // Find first/last of character set
    pos = str1.find_first_of("aeiou");
    std::cout << "First vowel at position: " << pos << std::endl;
    
    pos = str1.find_last_of("aeiou");
    std::cout << "Last vowel at position: " << pos << std::endl;
    
    // Find first/last not of character set
    pos = str1.find_first_not_of("Hello ");
    std::cout << "First non-'Hello ' character at position: " << pos << std::endl;
    
    pos = str1.find_last_not_of("ld");
    std::cout << "Last non-'ld' character at position: " << pos << std::endl;
}
```

### Substring and Replacement
```cpp
#include <string>
#include <iostream>

void substring_replacement() {
    std::string text = "The quick brown fox jumps over the lazy dog";
    
    // Substring
    std::string sub1 = text.substr(4, 5);  // "quick"
    std::string sub2 = text.substr(10);    // from position 10 to end
    
    std::cout << "Original: " << text << std::endl;
    std::cout << "Substring (4, 5): " << sub1 << std::endl;
    std::cout << "Substring from 10: " << sub2 << std::endl;
    
    // Replace
    std::string replaced = text;
    replaced.replace(4, 5, "slow");  // Replace "quick" with "slow"
    std::cout << "After replace: " << replaced << std::endl;
    
    // Replace with find
    std::string text2 = text;
    size_t pos = text2.find("fox");
    if (pos != std::string::npos) {
        text2.replace(pos, 3, "cat");
    }
    std::cout << "Replace fox with cat: " << text2 << std::endl;
    
    // Multiple replacements
    std::string text3 = "apple apple apple";
    pos = 0;
    while ((pos = text3.find("apple", pos)) != std::string::npos) {
        text3.replace(pos, 5, "orange");
        pos += 6; // Length of "orange"
    }
    std::cout << "Replace all apples: " << text3 << std::endl;
}
```

### Wide Strings (std::wstring)
```cpp
#include <string>
#include <iostream>
#include <locale>
#include <codecvt>

void wide_string_examples() {
    // Wide strings for Unicode support
    std::wstring wstr = L"Hello 世界";
    std::wcout << L"Wide string: " << wstr << std::endl;
    
    // Size in wide characters
    std::wcout << L"Length: " << wstr.length() << std::endl;
    
    // Wide string operations
    wstr += L" Unicode";
    std::wcout << L"After append: " << wstr << std::endl;
    
    // Find in wide string
    size_t pos = wstr.find(L"世界");
    if (pos != std::wstring::npos) {
        std::wcout << L"Found at position: " << pos << std::endl;
    }
    
    // Convert between narrow and wide strings (C++11/14)
    // Note: std::wstring_convert is deprecated in C++17
    std::string narrow_str = "Hello World";
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    
    try {
        std::wstring wide_from_narrow = converter.from_bytes(narrow_str);
        std::string narrow_from_wide = converter.to_bytes(wstr);
        
        std::wcout << L"Converted to wide: " << wide_from_narrow << std::endl;
        std::cout << "Converted to narrow: " << narrow_from_wide << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Conversion error: " << e.what() << std::endl;
    }
}
```

## std::string_view (C++17)

### Basic string_view Usage
```cpp
#include <string_view>
#include <string>
#include <iostream>

// Function that accepts string_view (more efficient)
void print_string_info(std::string_view sv) {
    std::cout << "String: '" << sv << "'" << std::endl;
    std::cout << "Length: " << sv.length() << std::endl;
    std::cout << "First char: " << sv.front() << std::endl;
    std::cout << "Last char: " << sv.back() << std::endl;
}

void string_view_basic() {
    std::cout << "\n=== string_view Basic Usage ===" << std::endl;
    
    // Can be constructed from various sources
    const char* c_str = "Hello World";
    std::string std_str = "STL String";
    
    // All of these work without copying
    print_string_info("String literal");
    print_string_info(c_str);
    print_string_info(std_str);
    
    // Substring without copying
    std::string_view sv(c_str, 5); // First 5 characters
    std::cout << "Substring view: " << sv << std::endl;
    
    // Remove prefix/suffix
    std::string_view full = "  Hello World  ";
    full.remove_prefix(2);  // Remove first 2 characters
    full.remove_suffix(2);  // Remove last 2 characters
    std::cout << "Trimmed view: '" << full << "'" << std::endl;
}
```

### string_view Operations
```cpp
#include <string_view>
#include <iostream>
#include <algorithm>

void string_view_operations() {
    std::cout << "\n=== string_view Operations ===" << std::endl;
    
    std::string_view text = "The quick brown fox jumps over the lazy dog";
    
    // Substring
    std::string_view word = text.substr(4, 5); // "quick"
    std::cout << "Substring: " << word << std::endl;
    
    // Find operations
    size_t pos = text.find("fox");
    if (pos != std::string_view::npos) {
        std::cout << "'fox' found at position: " << pos << std::endl;
    }
    
    // Find with string_view
    std::string_view search_term = "brown";
    pos = text.find(search_term);
    std::cout << "'" << search_term << "' found at position: " << pos << std::endl;
    
    // Starts with / ends with (C++20 features, but can be implemented)
    auto starts_with = [](std::string_view text, std::string_view prefix) {
        return text.substr(0, prefix.size()) == prefix;
    };
    
    auto ends_with = [](std::string_view text, std::string_view suffix) {
        return text.size() >= suffix.size() && 
               text.substr(text.size() - suffix.size()) == suffix;
    };
    
    std::cout << "Starts with 'The': " << starts_with(text, "The") << std::endl;
    std::cout << "Ends with 'dog': " << ends_with(text, "dog") << std::endl;
    
    // Compare
    std::string_view sv1 = "apple";
    std::string_view sv2 = "banana";
    std::cout << "Compare 'apple' vs 'banana': " << sv1.compare(sv2) << std::endl;
}
```

### string_view vs string Performance
```cpp
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <iostream>

// Functions for performance comparison
void process_with_string(const std::string& str) {
    // Simulated processing
    for (char c : str) {
        volatile char temp = c; // Prevent optimization
        (void)temp;
    }
}

void process_with_string_view(std::string_view sv) {
    // Simulated processing
    for (char c : sv) {
        volatile char temp = c; // Prevent optimization
        (void)temp;
    }
}

void string_view_performance() {
    std::cout << "\n=== string_view Performance Comparison ===" << std::endl;
    
    std::vector<std::string> strings;
    for (int i = 0; i < 10000; ++i) {
        strings.push_back("This is test string number " + std::to_string(i));
    }
    
    // Test with std::string (involves copying for substrings)
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& str : strings) {
        if (str.length() > 10) {
            std::string sub = str.substr(5, 10); // Creates copy
            process_with_string(sub);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_string = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test with std::string_view (no copying)
    start = std::chrono::high_resolution_clock::now();
    
    for (const auto& str : strings) {
        if (str.length() > 10) {
            std::string_view sub = std::string_view(str).substr(5, 10); // No copy
            process_with_string_view(sub);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto duration_string_view = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "std::string processing: " << duration_string.count() << " microseconds" << std::endl;
    std::cout << "std::string_view processing: " << duration_string_view.count() << " microseconds" << std::endl;
    
    if (duration_string_view.count() < duration_string.count()) {
        std::cout << "string_view is " << (duration_string.count() / static_cast<double>(duration_string_view.count())) 
                  << "x faster" << std::endl;
    }
}
```

## String Algorithms and Operations

### String Trimming
```cpp
#include <string>
#include <algorithm>
#include <cctype>
#include <iostream>

// Trim whitespace from left
std::string ltrim(std::string str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return str;
}

// Trim whitespace from right
std::string rtrim(std::string str) {
    str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), str.end());
    return str;
}

// Trim whitespace from both ends
std::string trim(std::string str) {
    return ltrim(rtrim(str));
}

void string_trimming_examples() {
    std::cout << "\n=== String Trimming ===" << std::endl;
    
    std::string text = "   Hello World   ";
    
    std::cout << "Original: '" << text << "'" << std::endl;
    std::cout << "Left trim: '" << ltrim(text) << "'" << std::endl;
    std::cout << "Right trim: '" << rtrim(text) << "'" << std::endl;
    std::cout << "Both trim: '" << trim(text) << "'" << std::endl;
    
    // In-place trimming with string_view
    std::string_view sv = "  \t\n  Hello World  \t\n  ";
    
    // Remove leading whitespace
    while (!sv.empty() && std::isspace(sv.front())) {
        sv.remove_prefix(1);
    }
    
    // Remove trailing whitespace
    while (!sv.empty() && std::isspace(sv.back())) {
        sv.remove_suffix(1);
    }
    
    std::cout << "string_view trimmed: '" << sv << "'" << std::endl;
}
```

### String Splitting
```cpp
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::string::size_type start = 0;
    std::string::size_type end = 0;
    
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    
    tokens.push_back(str.substr(start));
    return tokens;
}

std::vector<std::string> split_whitespace(const std::string& str) {
    std::istringstream iss(str);
    std::vector<std::string> tokens;
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

void string_splitting_examples() {
    std::cout << "\n=== String Splitting ===" << std::endl;
    
    std::string csv = "apple,banana,cherry,date";
    auto fruits = split(csv, ',');
    
    std::cout << "CSV split: ";
    for (const auto& fruit : fruits) {
        std::cout << "'" << fruit << "' ";
    }
    std::cout << std::endl;
    
    std::string path = "home/user/documents/file.txt";
    auto path_parts = split(path, '/');
    
    std::cout << "Path split: ";
    for (const auto& part : path_parts) {
        std::cout << "'" << part << "' ";
    }
    std::cout << std::endl;
    
    std::string text = "word1::word2::word3";
    auto words = split_string(text, "::");
    
    std::cout << "Multi-char delimiter split: ";
    for (const auto& word : words) {
        std::cout << "'" << word << "' ";
    }
    std::cout << std::endl;
    
    std::string sentence = "  The   quick  brown   fox  ";
    auto sentence_words = split_whitespace(sentence);
    
    std::cout << "Whitespace split: ";
    for (const auto& word : sentence_words) {
        std::cout << "'" << word << "' ";
    }
    std::cout << std::endl;
}
```

### String Case Conversion
```cpp
#include <string>
#include <algorithm>
#include <cctype>
#include <iostream>

std::string to_upper(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), 
                   [](unsigned char c) { return std::toupper(c); });
    return str;
}

std::string to_lower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), 
                   [](unsigned char c) { return std::tolower(c); });
    return str;
}

std::string capitalize(std::string str) {
    if (!str.empty()) {
        str[0] = std::toupper(str[0]);
        for (size_t i = 1; i < str.length(); ++i) {
            str[i] = std::tolower(str[i]);
        }
    }
    return str;
}

std::string title_case(std::string str) {
    bool capitalize_next = true;
    
    for (char& c : str) {
        if (std::isalpha(c)) {
            if (capitalize_next) {
                c = std::toupper(c);
                capitalize_next = false;
            } else {
                c = std::tolower(c);
            }
        } else {
            capitalize_next = true;
        }
    }
    
    return str;
}

void string_case_conversion() {
    std::cout << "\n=== String Case Conversion ===" << std::endl;
    
    std::string text = "Hello World From C++ STL";
    
    std::cout << "Original: " << text << std::endl;
    std::cout << "Upper: " << to_upper(text) << std::endl;
    std::cout << "Lower: " << to_lower(text) << std::endl;
    std::cout << "Capitalize: " << capitalize(text) << std::endl;
    std::cout << "Title Case: " << title_case("hello world from c++ stl") << std::endl;
}
```

### String Pattern Matching and Replacement
```cpp
#include <string>
#include <regex>
#include <iostream>

void string_pattern_matching() {
    std::cout << "\n=== String Pattern Matching ===" << std::endl;
    
    std::string text = "Contact us at support@example.com or sales@company.org";
    
    // Simple pattern matching without regex
    auto contains_email = [](const std::string& str) {
        return str.find('@') != std::string::npos && str.find('.') != std::string::npos;
    };
    
    std::cout << "Contains email: " << contains_email(text) << std::endl;
    
    // Using regex for more complex patterns
    std::regex email_pattern(R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)");
    
    // Find all matches
    std::sregex_iterator start(text.begin(), text.end(), email_pattern);
    std::sregex_iterator end;
    
    std::cout << "Email addresses found:" << std::endl;
    for (std::sregex_iterator i = start; i != end; ++i) {
        std::smatch match = *i;
        std::cout << "  " << match.str() << std::endl;
    }
    
    // Replace emails with placeholder
    std::string masked = std::regex_replace(text, email_pattern, "[EMAIL]");
    std::cout << "Masked text: " << masked << std::endl;
    
    // Validate string format
    auto is_phone_number = [](const std::string& str) {
        std::regex phone_pattern(R"(\(\d{3}\) \d{3}-\d{4})");
        return std::regex_match(str, phone_pattern);
    };
    
    std::cout << "Valid phone (123) 456-7890: " << is_phone_number("(123) 456-7890") << std::endl;
    std::cout << "Valid phone 123-456-7890: " << is_phone_number("123-456-7890") << std::endl;
}
```

## String Conversions

### Numeric Conversions
```cpp
#include <string>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <iomanip>

void numeric_conversions() {
    std::cout << "\n=== String Numeric Conversions ===" << std::endl;
    
    // String to number
    std::string int_str = "12345";
    std::string float_str = "123.45";
    std::string hex_str = "0xFF";
    std::string bin_str = "0b1010";
    
    try {
        // Basic conversions
        int i = std::stoi(int_str);
        long l = std::stol(int_str);
        float f = std::stof(float_str);
        double d = std::stod(float_str);
        
        std::cout << "String to int: " << i << std::endl;
        std::cout << "String to long: " << l << std::endl;
        std::cout << "String to float: " << f << std::endl;
        std::cout << "String to double: " << d << std::endl;
        
        // Hex conversion
        int hex_val = std::stoi(hex_str, nullptr, 16);
        std::cout << "Hex string to int: " << hex_val << std::endl;
        
        // Binary conversion (manual)
        int bin_val = std::stoi(bin_str.substr(2), nullptr, 2);
        std::cout << "Binary string to int: " << bin_val << std::endl;
        
        // With error position
        size_t pos;
        double val = std::stod("123.45abc", &pos);
        std::cout << "Parsed: " << val << ", stopped at position: " << pos << std::endl;
        
    } catch (const std::invalid_argument& e) {
        std::cout << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Out of range: " << e.what() << std::endl;
    }
    
    // Number to string
    int num = 42;
    double pi = 3.14159265359;
    
    std::string num_str = std::to_string(num);
    std::string pi_str = std::to_string(pi);
    
    std::cout << "Int to string: " << num_str << std::endl;
    std::cout << "Double to string: " << pi_str << std::endl;
    
    // Formatted conversion using stringstream
    std::ostringstream oss;
    oss << "Number: " << std::setw(5) << std::setfill('0') << num;
    oss << ", Pi: " << std::fixed << std::setprecision(3) << pi;
    
    std::cout << "Formatted: " << oss.str() << std::endl;
}
```

### Custom String Conversions
```cpp
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

// Convert any type to string if it supports operator<<
template<typename T>
std::string to_string_custom(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

// Convert string to any type if it supports operator>>
template<typename T>
T from_string(const std::string& str) {
    std::istringstream iss(str);
    T value;
    if (!(iss >> value)) {
        throw std::invalid_argument("Cannot convert string to type");
    }
    return value;
}

// Convert container to string
template<typename Container>
std::string container_to_string(const Container& container, const std::string& delimiter = ", ") {
    std::ostringstream oss;
    auto it = container.begin();
    if (it != container.end()) {
        oss << *it;
        ++it;
    }
    
    for (; it != container.end(); ++it) {
        oss << delimiter << *it;
    }
    
    return oss.str();
}

void custom_conversions() {
    std::cout << "\n=== Custom String Conversions ===" << std::endl;
    
    // Test custom to_string
    int value = 42;
    double pi = 3.14159;
    std::cout << "Custom int to string: " << to_string_custom(value) << std::endl;
    std::cout << "Custom double to string: " << to_string_custom(pi) << std::endl;
    
    // Test custom from_string
    try {
        int parsed_int = from_string<int>("123");
        double parsed_double = from_string<double>("45.67");
        
        std::cout << "Parsed int: " << parsed_int << std::endl;
        std::cout << "Parsed double: " << parsed_double << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Parsing error: " << e.what() << std::endl;
    }
    
    // Test container to string
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Vector to string: " << container_to_string(numbers) << std::endl;
    std::cout << "Vector to string (custom delimiter): " << container_to_string(numbers, " | ") << std::endl;
}
```

## Advanced String Processing

### String Builder Pattern
```cpp
#include <string>
#include <sstream>
#include <iostream>

class StringBuilder {
private:
    std::ostringstream stream_;
    
public:
    StringBuilder& append(const std::string& str) {
        stream_ << str;
        return *this;
    }
    
    StringBuilder& append(char c) {
        stream_ << c;
        return *this;
    }
    
    template<typename T>
    StringBuilder& append(const T& value) {
        stream_ << value;
        return *this;
    }
    
    StringBuilder& appendLine(const std::string& str = "") {
        stream_ << str << '\n';
        return *this;
    }
    
    StringBuilder& clear() {
        stream_.str("");
        stream_.clear();
        return *this;
    }
    
    std::string toString() const {
        return stream_.str();
    }
    
    size_t length() const {
        return stream_.str().length();
    }
    
    bool empty() const {
        return stream_.str().empty();
    }
};

void string_builder_example() {
    std::cout << "\n=== String Builder Pattern ===" << std::endl;
    
    StringBuilder sb;
    
    sb.append("Hello ")
      .append("World")
      .append('!')
      .appendLine()
      .append("Number: ")
      .append(42)
      .appendLine()
      .append("Pi: ")
      .append(3.14159);
    
    std::cout << "Built string:" << std::endl;
    std::cout << sb.toString() << std::endl;
    std::cout << "Length: " << sb.length() << std::endl;
}
```

### String Pool for Memory Efficiency
```cpp
#include <string>
#include <unordered_set>
#include <iostream>
#include <memory>

class StringPool {
private:
    std::unordered_set<std::string> pool_;
    
public:
    const std::string& intern(const std::string& str) {
        auto result = pool_.insert(str);
        return *result.first;
    }
    
    const std::string& intern(std::string&& str) {
        auto result = pool_.insert(std::move(str));
        return *result.first;
    }
    
    size_t size() const {
        return pool_.size();
    }
    
    void clear() {
        pool_.clear();
    }
};

void string_pool_example() {
    std::cout << "\n=== String Pool Example ===" << std::endl;
    
    StringPool pool;
    
    // Intern strings
    const std::string& str1 = pool.intern("Hello");
    const std::string& str2 = pool.intern("World");
    const std::string& str3 = pool.intern("Hello"); // Same as str1
    
    std::cout << "str1: " << str1 << " (address: " << &str1 << ")" << std::endl;
    std::cout << "str2: " << str2 << " (address: " << &str2 << ")" << std::endl;
    std::cout << "str3: " << str3 << " (address: " << &str3 << ")" << std::endl;
    
    std::cout << "str1 and str3 same object: " << (&str1 == &str3) << std::endl;
    std::cout << "Pool size: " << pool.size() << std::endl;
}
```

### String Algorithms Performance
```cpp
#include <string>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

void string_performance_comparison() {
    std::cout << "\n=== String Performance Comparison ===" << std::endl;
    
    const int iterations = 100000;
    
    // Test string concatenation methods
    auto test_string_concat = [iterations]() {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string result;
        for (int i = 0; i < iterations; ++i) {
            result += "test";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };
    
    auto test_stringstream_concat = [iterations]() {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ostringstream oss;
        for (int i = 0; i < iterations; ++i) {
            oss << "test";
        }
        std::string result = oss.str();
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };
    
    auto test_reserve_concat = [iterations]() {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::string result;
        result.reserve(iterations * 4); // Pre-allocate
        for (int i = 0; i < iterations; ++i) {
            result += "test";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    };
    
    auto time_concat = test_string_concat();
    auto time_stringstream = test_stringstream_concat();
    auto time_reserve = test_reserve_concat();
    
    std::cout << "String concatenation: " << time_concat << " microseconds" << std::endl;
    std::cout << "Stringstream: " << time_stringstream << " microseconds" << std::endl;
    std::cout << "String with reserve: " << time_reserve << " microseconds" << std::endl;
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <string>
#include <string_view>

int main() {
    std::cout << "=== String Handling Examples ===" << std::endl;
    
    std::cout << "\n--- Basic String Operations ---" << std::endl;
    basic_string_operations();
    string_comparison_searching();
    substring_replacement();
    wide_string_examples();
    
    std::cout << "\n--- string_view ---" << std::endl;
    string_view_basic();
    string_view_operations();
    string_view_performance();
    
    std::cout << "\n--- String Algorithms ---" << std::endl;
    string_trimming_examples();
    string_splitting_examples();
    string_case_conversion();
    string_pattern_matching();
    
    std::cout << "\n--- String Conversions ---" << std::endl;
    numeric_conversions();
    custom_conversions();
    
    std::cout << "\n--- Advanced Processing ---" << std::endl;
    string_builder_example();
    string_pool_example();
    string_performance_comparison();
    
    return 0;
}
```

## Best Practices

1. **Use std::string_view for read-only operations**: Avoids unnecessary copying
2. **Reserve capacity when building strings**: Reduces reallocations
3. **Use appropriate string search methods**: find(), find_first_of(), etc.
4. **Be careful with string lifetimes**: Especially with string_view
5. **Consider locale for case conversions**: For international applications
6. **Use regex for complex pattern matching**: More reliable than manual parsing
7. **Profile string operations**: Understand performance characteristics

## Common Pitfalls

1. **string_view dangling references**: Don't store string_view to temporary strings
2. **Character encoding issues**: Be aware of ASCII vs UTF-8 vs wide characters
3. **Performance of operator+**: Can create many temporary objects
4. **find() return value**: Check against npos
5. **Empty string handling**: Consider edge cases

## Key Concepts Summary

1. **std::string**: Mutable string class with rich interface
2. **std::wstring**: Wide character strings for Unicode
3. **std::string_view**: Non-owning string view for efficiency
4. **String algorithms**: Trimming, splitting, case conversion
5. **Conversions**: Between strings and numeric types
6. **Performance**: Consider memory allocation patterns

## Exercises

1. Implement a CSV parser using string operations
2. Create a template-based string formatter like printf
3. Build a simple text template engine with variable substitution
4. Implement a string similarity algorithm (Levenshtein distance)
5. Create a word wrap function that handles different line lengths
6. Build a simple markup language processor
7. Implement a string compression algorithm using repeated substring detection
8. Create a fast string search algorithm (like Boyer-Moore or KMP)
