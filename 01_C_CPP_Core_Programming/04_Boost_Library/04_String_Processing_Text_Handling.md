# String Processing and Text Handling

*Duration: 1 week*

## Overview

This section covers Boost's comprehensive string processing libraries, including regular expressions, string algorithms, tokenization, and formatting.

## Learning Topics

### Boost.Regex
- Advanced regular expressions beyond std::regex
- Perl-compatible regex syntax
- Named captures and backreferences
- Performance optimization techniques

### Boost.String_Algo
- String manipulation algorithms
- Case conversion, trimming, splitting
- Find and replace operations
- Predicate-based string operations

### Boost.Tokenizer
- Flexible tokenization strategies
- Custom token separators and escape sequences
- CSV parsing and structured data extraction
- Iterator-based token access

### Boost.Format
- Type-safe string formatting
- Positional and named parameters
- Custom format specifications
- Integration with streams and internationalization

## Code Examples

### Boost.Regex - Basic Usage
```cpp
#include <boost/regex.hpp>
#include <iostream>
#include <string>
#include <vector>

void demonstrate_basic_regex() {
    std::string text = "Email: john.doe@example.com, Phone: (555) 123-4567";
    
    // Email extraction
    boost::regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    boost::smatch email_match;
    
    if (boost::regex_search(text, email_match, email_pattern)) {
        std::cout << "Found email: " << email_match[0] << "\n";
    }
    
    // Phone number extraction with groups
    boost::regex phone_pattern(R"(\((\d{3})\)\s*(\d{3})-(\d{4}))");
    boost::smatch phone_match;
    
    if (boost::regex_search(text, phone_match, phone_pattern)) {
        std::cout << "Phone: " << phone_match[0] << "\n";
        std::cout << "Area code: " << phone_match[1] << "\n";
        std::cout << "Exchange: " << phone_match[2] << "\n";
        std::cout << "Number: " << phone_match[3] << "\n";
    }
}
```

### Named Captures and Advanced Regex
```cpp
#include <boost/regex.hpp>
#include <iostream>
#include <string>

void demonstrate_named_captures() {
    std::string log_line = "2023-06-21 14:30:25 [ERROR] Failed to connect to database";
    
    // Named capture groups
    boost::regex log_pattern(
        R"((?<date>\d{4}-\d{2}-\d{2})\s+)"
        R"((?<time>\d{2}:\d{2}:\d{2})\s+)"
        R"(\[(?<level>\w+)\]\s+)"
        R"((?<message>.+))"
    );
    
    boost::smatch match;
    if (boost::regex_match(log_line, match, log_pattern)) {
        std::cout << "Date: " << match["date"] << "\n";
        std::cout << "Time: " << match["time"] << "\n";
        std::cout << "Level: " << match["level"] << "\n";
        std::cout << "Message: " << match["message"] << "\n";
    }
}

void demonstrate_regex_replace() {
    std::string html = "<p>Hello <b>world</b>!</p>";
    
    // Remove HTML tags
    boost::regex tag_pattern("<[^>]*>");
    std::string plain_text = boost::regex_replace(html, tag_pattern, "");
    std::cout << "Plain text: " << plain_text << "\n";
    
    // Replace with capture groups
    std::string text = "Date: 2023/06/21";
    boost::regex date_pattern(R"((\d{4})/(\d{2})/(\d{2}))");
    std::string formatted = boost::regex_replace(text, date_pattern, "$3-$2-$1");
    std::cout << "Formatted: " << formatted << "\n";
}
```

### Boost.String_Algo Examples
```cpp
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <vector>

void demonstrate_string_algorithms() {
    std::string text = "  Hello World, How Are You?  ";
    
    // Trimming
    std::string trimmed = boost::trim_copy(text);
    std::cout << "Trimmed: '" << trimmed << "'\n";
    
    // Case conversion
    std::string upper = boost::to_upper_copy(text);
    std::string lower = boost::to_lower_copy(text);
    std::cout << "Upper: " << upper << "\n";
    std::cout << "Lower: " << lower << "\n";
    
    // Splitting
    std::vector<std::string> words;
    boost::split(words, trimmed, boost::is_any_of(" ,"), boost::token_compress_on);
    
    std::cout << "Words:\n";
    for (const auto& word : words) {
        if (!word.empty()) {
            std::cout << "  '" << word << "'\n";
        }
    }
    
    // Joining
    std::string joined = boost::join(words, " | ");
    std::cout << "Joined: " << joined << "\n";
}

void demonstrate_string_predicates() {
    std::string filename = "document.PDF";
    
    // Case-insensitive operations
    if (boost::iends_with(filename, ".pdf")) {
        std::cout << "File is a PDF\n";
    }
    
    if (boost::istarts_with(filename, "doc")) {
        std::cout << "Filename starts with 'doc'\n";
    }
    
    // Find and replace
    std::string text = "The quick brown fox jumps over the lazy dog";
    boost::replace_all(text, "the", "THE");
    std::cout << "Replaced: " << text << "\n";
    
    // Case-insensitive replace
    boost::ireplace_all(text, "fox", "CAT");
    std::cout << "Case-insensitive replace: " << text << "\n";
}

void demonstrate_custom_predicates() {
    std::string text = "abc123def456ghi";
    
    // Find first numeric sequence
    auto it = boost::find_if(text, boost::is_digit());
    if (it != text.end()) {
        std::cout << "First digit at position: " << std::distance(text.begin(), it) << "\n";
    }
    
    // Split on digits
    std::vector<std::string> parts;
    boost::split(parts, text, boost::is_digit(), boost::token_compress_on);
    
    std::cout << "Non-digit parts:\n";
    for (const auto& part : parts) {
        if (!part.empty()) {
            std::cout << "  '" << part << "'\n";
        }
    }
}
```

### Boost.Tokenizer Examples
```cpp
#include <boost/tokenizer.hpp>
#include <iostream>
#include <string>

void demonstrate_basic_tokenizer() {
    std::string text = "apple,banana;orange:grape";
    
    // Simple character separator
    boost::char_separator<char> sep(",;:");
    boost::tokenizer<boost::char_separator<char>> tokens(text, sep);
    
    std::cout << "Tokens:\n";
    for (const auto& token : tokens) {
        std::cout << "  '" << token << "'\n";
    }
}

void demonstrate_csv_tokenizer() {
    std::string csv_line = "\"John Doe\",25,\"Engineer, Software\",\"New York, NY\"";
    
    // CSV tokenizer handles quoted fields
    boost::tokenizer<boost::escaped_list_separator<char>> csv_tokens(csv_line);
    
    std::cout << "CSV fields:\n";
    int field_num = 1;
    for (const auto& token : csv_tokens) {
        std::cout << "  Field " << field_num++ << ": '" << token << "'\n";
    }
}

void demonstrate_custom_tokenizer() {
    std::string data = "key1=value1&key2=value2&key3=value with spaces";
    
    // Custom separator for URL parameters
    boost::char_separator<char> sep("&");
    boost::tokenizer<boost::char_separator<char>> param_tokens(data, sep);
    
    std::cout << "URL parameters:\n";
    for (const auto& param : param_tokens) {
        // Further split each parameter
        boost::char_separator<char> eq_sep("=");
        boost::tokenizer<boost::char_separator<char>> kv_tokens(param, eq_sep);
        
        auto it = kv_tokens.begin();
        if (it != kv_tokens.end()) {
            std::string key = *it++;
            std::string value = (it != kv_tokens.end()) ? *it : "";
            std::cout << "  " << key << " -> " << value << "\n";
        }
    }
}
```

### Boost.Format Examples
```cpp
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <iomanip>

void demonstrate_basic_format() {
    // Basic formatting
    boost::format fmt("Hello %1%, you are %2% years old!");
    std::string result = (fmt % "Alice" % 25).str();
    std::cout << result << "\n";
    
    // Reusable format object
    boost::format number_fmt("Number: %1$5d, Hex: %1$#x, Float: %2$8.2f");
    
    std::cout << number_fmt % 42 % 3.14159 << "\n";
    std::cout << number_fmt % 255 % 2.71828 << "\n";
}

void demonstrate_format_specifications() {
    // Width and alignment
    boost::format table_fmt("| %1$-15s | %2$8d | %3$10.2f |\n");
    
    std::cout << "Product Table:\n";
    std::cout << table_fmt % "Name" % "Quantity" % "Price";
    std::cout << std::string(40, '-') << "\n";
    std::cout << table_fmt % "Widget A" % 150 % 29.99;
    std::cout << table_fmt % "Gadget B" % 75 % 149.50;
    std::cout << table_fmt % "Tool C" % 200 % 15.25;
}

void demonstrate_advanced_format() {
    // Named arguments (using position specifiers)
    auto create_message = [](const std::string& name, int age, double salary) {
        boost::format fmt("Employee: %2$s, Age: %3$d, Salary: $%1$,.2f");
        return (fmt % salary % name % age).str();
    };
    
    std::cout << create_message("Bob", 30, 75000.50) << "\n";
    
    // Conditional formatting
    auto format_status = [](bool success, const std::string& operation) {
        boost::format fmt("Operation '%2$s': %1$s");
        return (fmt % (success ? "SUCCESS" : "FAILED") % operation).str();
    };
    
    std::cout << format_status(true, "Database connection") << "\n";
    std::cout << format_status(false, "File upload") << "\n";
}
```

### Text Processing Pipeline
```cpp
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <vector>

class TextProcessor {
public:
    // Clean and normalize text
    std::string normalize(const std::string& text) {
        std::string result = text;
        
        // Remove extra whitespace
        boost::trim(result);
        boost::replace_all(result, "\\t", " ");
        boost::replace_all(result, "\\n", " ");
        
        // Normalize multiple spaces
        boost::regex multi_space("\\s+");
        result = boost::regex_replace(result, multi_space, " ");
        
        return result;
    }
    
    // Extract email addresses
    std::vector<std::string> extractEmails(const std::string& text) {
        std::vector<std::string> emails;
        boost::regex email_pattern(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
        
        boost::sregex_iterator start(text.begin(), text.end(), email_pattern);
        boost::sregex_iterator end;
        
        for (auto it = start; it != end; ++it) {
            emails.push_back(it->str());
        }
        
        return emails;
    }
    
    // Generate report
    std::string generateReport(const std::string& text) {
        std::string normalized = normalize(text);
        auto emails = extractEmails(normalized);
        
        std::vector<std::string> words;
        boost::split(words, normalized, boost::is_space(), boost::token_compress_on);
        
        boost::format report_fmt(
            "Text Analysis Report\\n"
            "===================\\n"
            "Characters: %1%\\n"
            "Words: %2%\\n"
            "Emails found: %3%\\n"
            "Email addresses: %4%\\n"
        );
        
        std::string email_list = boost::join(emails, ", ");
        if (email_list.empty()) email_list = "None";
        
        return (report_fmt % normalized.length() % words.size() % emails.size() % email_list).str();
    }
};

void demonstrate_text_processing_pipeline() {
    std::string sample_text = 
        "   Contact   us at  support@example.com   or   \\n\\n"
        "   sales@company.org   for   more   information.   \\t\\t"
        "   Invalid email: not-an-email   ";
    
    TextProcessor processor;
    std::cout << processor.generateReport(sample_text) << "\n";
}
```

## Practical Exercises

1. **Log File Parser**
   - Parse web server log files using regex
   - Extract timestamps, IP addresses, and HTTP status codes
   - Generate statistics reports

2. **CSV Data Processor**
   - Read CSV files with complex quoting and escaping
   - Clean and validate data fields
   - Export processed data in different formats

3. **Text Template Engine**
   - Create a simple template system using Boost.Format
   - Support conditional sections and loops
   - Handle user-provided templates safely

4. **Natural Language Processor**
   - Implement basic text analysis (word count, sentiment)
   - Extract named entities using regex patterns
   - Generate summary statistics

## Performance Considerations

### Regex Performance
- Compile regex patterns once, reuse multiple times
- Use non-capturing groups when captures aren't needed
- Consider regex alternatives for simple string operations

### String Algorithm Efficiency
- Use in-place operations when possible
- Reserve string capacity for large concatenations
- Choose appropriate algorithms for specific use cases

### Memory Management
- Avoid unnecessary string copies
- Use string views when available
- Consider streaming for large text files

## Best Practices

1. **Regular Expression Design**
   - Make patterns as specific as possible
   - Test regex patterns thoroughly
   - Document complex patterns
   - Consider internationalization

2. **String Processing Pipeline**
   - Chain operations efficiently
   - Handle encoding issues consistently
   - Validate input at boundaries
   - Provide meaningful error messages

3. **Format String Safety**
   - Validate format strings at compile time when possible
   - Handle missing or extra arguments gracefully
   - Use type-safe formatting approaches

## Assessment

- Can design efficient regex patterns for complex text processing
- Understands string algorithm performance characteristics
- Can build robust text processing pipelines
- Implements safe and flexible formatting systems

## Next Steps

Move on to [Date and Time Utilities](05_Date_Time_Utilities.md) to explore Boost's temporal processing capabilities.
