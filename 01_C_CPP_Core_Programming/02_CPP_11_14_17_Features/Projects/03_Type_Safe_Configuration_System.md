# Type-Safe Configuration System Project

## Project Overview

Build a comprehensive configuration system using C++11/14/17 features including type traits, variadic templates, SFINAE, constexpr, std::optional, and std::variant to create a flexible, type-safe configuration management system.

## Learning Objectives

- Master template metaprogramming techniques
- Understand type traits and SFINAE
- Use variadic templates effectively
- Implement compile-time validation with constexpr
- Handle optional and variant types safely
- Create flexible, extensible APIs

## Project Structure

```
config_system_project/
├── src/
│   ├── main.cpp
│   ├── config_manager.cpp
│   ├── config_parser.cpp
│   ├── config_validator.cpp
│   └── config_serializer.cpp
├── include/
│   ├── config_manager.h
│   ├── config_types.h
│   ├── config_traits.h
│   ├── config_parser.h
│   ├── config_validator.h
│   ├── config_serializer.h
│   └── config_utilities.h
├── examples/
│   ├── basic_usage.cpp
│   ├── advanced_features.cpp
│   └── real_world_config.cpp
├── tests/
│   ├── test_config_types.cpp
│   ├── test_config_validation.cpp
│   └── test_config_serialization.cpp
├── data/
│   ├── sample_config.json
│   ├── sample_config.ini
│   └── sample_config.xml
└── CMakeLists.txt
```

## Core Components

### 1. Type System and Traits

```cpp
// include/config_types.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <type_traits>
#include <chrono>
#include <functional>

namespace config {

// Forward declarations
template<typename T> class ConfigValue;
class ConfigNode;

// Basic value types that can be stored in configuration
using ConfigInt = int64_t;
using ConfigFloat = double;
using ConfigString = std::string;
using ConfigBool = bool;

// Container types
template<typename T>
using ConfigArray = std::vector<T>;

template<typename K, typename V>
using ConfigMap = std::map<K, V>;

// Special types
using ConfigDuration = std::chrono::milliseconds;
using ConfigPath = std::string;  // File system path
using ConfigURL = std::string;   // URL string

// Variant type that can hold any basic configuration value
using ConfigVariant = std::variant<
    ConfigInt,
    ConfigFloat,
    ConfigString,
    ConfigBool,
    ConfigDuration,
    ConfigPath,
    ConfigURL,
    ConfigArray<ConfigInt>,
    ConfigArray<ConfigFloat>,
    ConfigArray<ConfigString>,
    ConfigArray<ConfigBool>,
    ConfigMap<ConfigString, ConfigString>
>;

// Optional variant for nullable values
using OptionalConfigVariant = std::optional<ConfigVariant>;

// Configuration node that can contain nested structures
class ConfigNode {
private:
    std::map<std::string, OptionalConfigVariant> values_;
    std::map<std::string, ConfigNode> children_;
    
public:
    // Value accessors
    template<typename T>
    std::optional<T> get(const std::string& key) const;
    
    template<typename T>
    T get_or(const std::string& key, const T& default_value) const;
    
    template<typename T>
    void set(const std::string& key, const T& value);
    
    // Node accessors
    ConfigNode& get_child(const std::string& key);
    const ConfigNode& get_child(const std::string& key) const;
    
    bool has_value(const std::string& key) const;
    bool has_child(const std::string& key) const;
    
    // Iteration
    auto begin() { return values_.begin(); }
    auto end() { return values_.end(); }
    auto begin() const { return values_.begin(); }
    auto end() const { return values_.end(); }
    
    auto children_begin() { return children_.begin(); }
    auto children_end() { return children_.end(); }
    auto children_begin() const { return children_.begin(); }
    auto children_end() const { return children_.end(); }
    
    // Utility
    void clear();
    bool empty() const;
    size_t size() const;
    std::vector<std::string> get_keys() const;
    std::vector<std::string> get_child_keys() const;
};

// Wrapper for typed configuration values with validation
template<typename T>
class ConfigValue {
private:
    std::optional<T> value_;
    std::function<bool(const T&)> validator_;
    T default_value_;
    std::string description_;
    bool required_;
    
public:
    ConfigValue() = default;
    
    ConfigValue(const T& default_val, const std::string& desc = "", bool req = false)
        : default_value_(default_val), description_(desc), required_(req) {}
    
    template<typename Validator>
    ConfigValue(const T& default_val, Validator&& validator, const std::string& desc = "", bool req = false)
        : validator_(std::forward<Validator>(validator))
        , default_value_(default_val)
        , description_(desc)
        , required_(req) {}
    
    // Value access
    const T& get() const {
        return value_.has_value() ? *value_ : default_value_;
    }
    
    bool has_value() const { return value_.has_value(); }
    bool is_required() const { return required_; }
    const std::string& description() const { return description_; }
    
    // Value setting with validation
    bool set(const T& value) {
        if (validator_ && !validator_(value)) {
            return false;
        }
        value_ = value;
        return true;
    }
    
    bool try_set(const T& value) noexcept {
        try {
            return set(value);
        } catch (...) {
            return false;
        }
    }
    
    void reset() { value_.reset(); }
    
    // Validation
    bool is_valid() const {
        if (required_ && !value_.has_value()) {
            return false;
        }
        if (value_.has_value() && validator_) {
            return validator_(*value_);
        }
        return true;
    }
};

} // namespace config
```

### 2. Type Traits and Template Metaprogramming

```cpp
// include/config_traits.h
#pragma once
#include "config_types.h"
#include <type_traits>
#include <string>
#include <sstream>
#include <chrono>

namespace config {
namespace traits {

// Primary template for checking if a type is supported by the config system
template<typename T>
struct is_config_type : std::false_type {};

// Specializations for supported types
template<> struct is_config_type<ConfigInt> : std::true_type {};
template<> struct is_config_type<ConfigFloat> : std::true_type {};
template<> struct is_config_type<ConfigString> : std::true_type {};
template<> struct is_config_type<ConfigBool> : std::true_type {};
template<> struct is_config_type<ConfigDuration> : std::true_type {};
template<> struct is_config_type<ConfigPath> : std::true_type {};
template<> struct is_config_type<ConfigURL> : std::true_type {};

// Array types
template<typename T> 
struct is_config_type<ConfigArray<T>> : is_config_type<T> {};

// Map types
template<typename K, typename V>
struct is_config_type<ConfigMap<K, V>> : std::conjunction<is_config_type<K>, is_config_type<V>> {};

// Helper variable template
template<typename T>
constexpr bool is_config_type_v = is_config_type<T>::value;

// Check if type can be converted from string
template<typename T>
struct is_string_convertible {
    template<typename U>
    static auto test(int) -> decltype(std::declval<std::stringstream&>() >> std::declval<U&>(), std::true_type{});
    
    template<typename>
    static std::false_type test(...);
    
    using type = decltype(test<T>(0));
    static constexpr bool value = type::value;
};

template<typename T>
constexpr bool is_string_convertible_v = is_string_convertible<T>::value;

// Specializations for types that need special handling
template<> struct is_string_convertible<ConfigString> : std::true_type {};
template<> struct is_string_convertible<ConfigBool> : std::true_type {};
template<> struct is_string_convertible<ConfigDuration> : std::true_type {};

// Check if type is numeric
template<typename T>
struct is_numeric : std::is_arithmetic<T> {};

template<typename T>
constexpr bool is_numeric_v = is_numeric<T>::value;

// Check if type is container
template<typename T>
struct is_container : std::false_type {};

template<typename T>
struct is_container<std::vector<T>> : std::true_type {};

template<typename K, typename V>
struct is_container<std::map<K, V>> : std::true_type {};

template<typename T>
constexpr bool is_container_v = is_container<T>::value;

// Get the value type of a container
template<typename T>
struct container_value_type {};

template<typename T>
struct container_value_type<std::vector<T>> {
    using type = T;
};

template<typename K, typename V>
struct container_value_type<std::map<K, V>> {
    using type = V;
};

template<typename T>
using container_value_type_t = typename container_value_type<T>::type;

// Type conversion utilities
template<typename T>
struct type_name {
    static std::string get() { return typeid(T).name(); }
};

// Specializations for better type names
template<> struct type_name<int> { static std::string get() { return "int"; } };
template<> struct type_name<double> { static std::string get() { return "double"; } };
template<> struct type_name<std::string> { static std::string get() { return "string"; } };
template<> struct type_name<bool> { static std::string get() { return "bool"; } };

template<typename T>
struct type_name<std::vector<T>> {
    static std::string get() { return "array<" + type_name<T>::get() + ">"; }
};

template<typename K, typename V>
struct type_name<std::map<K, V>> {
    static std::string get() { 
        return "map<" + type_name<K>::get() + ", " + type_name<V>::get() + ">"; 
    }
};

} // namespace traits

// String conversion functions using SFINAE
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, std::optional<T>>
from_string(const std::string& str) {
    std::stringstream ss(str);
    T value;
    if (ss >> value && ss.eof()) {
        return value;
    }
    return std::nullopt;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, bool>, std::optional<T>>
from_string(const std::string& str) {
    std::string lower_str = str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    
    if (lower_str == "true" || lower_str == "1" || lower_str == "yes" || lower_str == "on") {
        return true;
    } else if (lower_str == "false" || lower_str == "0" || lower_str == "no" || lower_str == "off") {
        return false;
    }
    return std::nullopt;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, std::string>, std::optional<T>>
from_string(const std::string& str) {
    return str;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, ConfigDuration>, std::optional<T>>
from_string(const std::string& str) {
    // Parse duration strings like "1000ms", "5s", "2m", "1h"
    std::regex duration_regex(R"((\d+)(ms|s|m|h)?)");
    std::smatch match;
    
    if (std::regex_match(str, match, duration_regex)) {
        int value = std::stoi(match[1].str());
        std::string unit = match[2].str();
        
        if (unit.empty() || unit == "ms") {
            return std::chrono::milliseconds(value);
        } else if (unit == "s") {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(value));
        } else if (unit == "m") {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::minutes(value));
        } else if (unit == "h") {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::hours(value));
        }
    }
    return std::nullopt;
}

// Convert to string
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, std::string>
to_string(const T& value) {
    return std::to_string(value);
}

template<typename T>
std::enable_if_t<std::is_same_v<T, bool>, std::string>
to_string(const T& value) {
    return value ? "true" : "false";
}

template<typename T>
std::enable_if_t<std::is_same_v<T, std::string>, std::string>
to_string(const T& value) {
    return value;
}

template<typename T>
std::enable_if_t<std::is_same_v<T, ConfigDuration>, std::string>
to_string(const T& value) {
    return std::to_string(value.count()) + "ms";
}

} // namespace config
```

### 3. Configuration Manager with Template Magic

```cpp
// include/config_manager.h
#pragma once
#include "config_types.h"
#include "config_traits.h"
#include <memory>
#include <functional>
#include <regex>

namespace config {

class ConfigError : public std::exception {
private:
    std::string message_;
    
public:
    explicit ConfigError(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
};

// Configuration schema definition
class ConfigSchema {
public:
    struct FieldSchema {
        std::string name;
        std::string type_name;
        std::string description;
        bool required;
        std::string default_value;
        std::function<bool(const std::string&)> validator;
        
        FieldSchema(const std::string& n, const std::string& t, const std::string& d = "", 
                   bool req = false, const std::string& def = "")
            : name(n), type_name(t), description(d), required(req), default_value(def) {}
    };
    
private:
    std::map<std::string, FieldSchema> fields_;
    std::map<std::string, std::unique_ptr<ConfigSchema>> nested_schemas_;
    
public:
    // Add field to schema
    template<typename T>
    ConfigSchema& add_field(const std::string& name, const std::string& description = "", 
                           bool required = false, const T& default_value = T{}) {
        static_assert(traits::is_config_type_v<T>, "Type not supported by config system");
        
        fields_.emplace(name, FieldSchema{
            name,
            traits::type_name<T>::get(),
            description,
            required,
            to_string(default_value)
        });
        
        return *this;
    }
    
    // Add field with validator
    template<typename T, typename Validator>
    ConfigSchema& add_field_with_validator(const std::string& name, Validator&& validator,
                                          const std::string& description = "", 
                                          bool required = false, const T& default_value = T{}) {
        static_assert(traits::is_config_type_v<T>, "Type not supported by config system");
        
        auto field = FieldSchema{
            name,
            traits::type_name<T>::get(),
            description,
            required,
            to_string(default_value)
        };
        
        field.validator = [validator = std::forward<Validator>(validator)](const std::string& str) -> bool {
            auto parsed = from_string<T>(str);
            return parsed && validator(*parsed);
        };
        
        fields_.emplace(name, std::move(field));
        return *this;
    }
    
    // Add nested schema
    ConfigSchema& add_nested_schema(const std::string& name, std::unique_ptr<ConfigSchema> schema) {
        nested_schemas_[name] = std::move(schema);
        return *this;
    }
    
    // Validation
    std::vector<std::string> validate(const ConfigNode& node) const;
    
    // Access
    const std::map<std::string, FieldSchema>& get_fields() const { return fields_; }
    const std::map<std::string, std::unique_ptr<ConfigSchema>>& get_nested_schemas() const { 
        return nested_schemas_; 
    }
};

// Main configuration manager
class ConfigManager {
private:
    ConfigNode root_;
    std::unique_ptr<ConfigSchema> schema_;
    std::vector<std::string> validation_errors_;
    
    // Path resolution
    std::vector<std::string> split_path(const std::string& path) const;
    ConfigNode* navigate_to_node(const std::string& path);
    const ConfigNode* navigate_to_node(const std::string& path) const;
    
public:
    ConfigManager() = default;
    explicit ConfigManager(std::unique_ptr<ConfigSchema> schema) : schema_(std::move(schema)) {}
    
    // Schema management
    void set_schema(std::unique_ptr<ConfigSchema> schema) { schema_ = std::move(schema); }
    const ConfigSchema* get_schema() const { return schema_.get(); }
    
    // Generic value access with path support (e.g., "database.host")
    template<typename T>
    std::optional<T> get(const std::string& path) {
        static_assert(traits::is_config_type_v<T>, "Type not supported by config system");
        
        const ConfigNode* node = navigate_to_node(path);
        if (!node) {
            return std::nullopt;
        }
        
        auto path_parts = split_path(path);
        const std::string& key = path_parts.back();
        
        return node->get<T>(key);
    }
    
    template<typename T>
    T get_or(const std::string& path, const T& default_value) {
        auto value = get<T>(path);
        return value ? *value : default_value;
    }
    
    template<typename T>
    void set(const std::string& path, const T& value) {
        static_assert(traits::is_config_type_v<T>, "Type not supported by config system");
        
        auto path_parts = split_path(path);
        const std::string& key = path_parts.back();
        
        // Navigate to parent node, creating if necessary
        ConfigNode* node = &root_;
        for (size_t i = 0; i < path_parts.size() - 1; ++i) {
            node = &node->get_child(path_parts[i]);
        }
        
        node->set(key, value);
    }
    
    // Typed configuration object support
    template<typename ConfigStruct>
    std::enable_if_t<std::is_class_v<ConfigStruct>, void>
    load_struct(ConfigStruct& config_obj, const std::string& prefix = "") {
        load_struct_impl(config_obj, prefix);
    }
    
    template<typename ConfigStruct>
    std::enable_if_t<std::is_class_v<ConfigStruct>, void>
    save_struct(const ConfigStruct& config_obj, const std::string& prefix = "") {
        save_struct_impl(config_obj, prefix);
    }
    
    // Validation
    bool validate();
    const std::vector<std::string>& get_validation_errors() const { return validation_errors_; }
    
    // File I/O
    bool load_from_file(const std::string& filename);
    bool save_to_file(const std::string& filename) const;
    
    // JSON support
    bool load_from_json(const std::string& json_string);
    std::string to_json(bool pretty = true) const;
    
    // INI support
    bool load_from_ini(const std::string& ini_string);
    std::string to_ini() const;
    
    // Environment variable support
    void load_from_env(const std::string& prefix = "");
    
    // Utility
    void clear() { root_.clear(); validation_errors_.clear(); }
    const ConfigNode& get_root() const { return root_; }
    
    // Debug information
    void print_structure() const;
    std::map<std::string, std::string> get_all_values() const;
    
private:
    template<typename T>
    void load_struct_impl(T& obj, const std::string& prefix);
    
    template<typename T>
    void save_struct_impl(const T& obj, const std::string& prefix);
};

// Compile-time configuration struct support using reflection-like techniques
#define CONFIG_FIELD(type, name, ...) \
    type name __VA_ARGS__; \
    static constexpr auto _config_field_##name() { \
        return std::make_tuple(#name, #type, offsetof(std::remove_reference_t<decltype(*this)>, name)); \
    }

// Helper macro for defining configuration structs
#define CONFIG_STRUCT(name) \
    struct name { \
        template<typename F> \
        void for_each_field(F&& func) const { \
            for_each_field_impl(std::forward<F>(func)); \
        } \
        template<typename F> \
        void for_each_field(F&& func) { \
            for_each_field_impl(std::forward<F>(func)); \
        } \
        private: \
        template<typename F> \
        void for_each_field_impl(F&& func) const

// Usage example for compile-time reflection
struct DatabaseConfig {
    CONFIG_FIELD(std::string, host, = "localhost");
    CONFIG_FIELD(int, port, = 5432);
    CONFIG_FIELD(std::string, username, = "user");
    CONFIG_FIELD(std::string, password);
    CONFIG_FIELD(std::string, database, = "mydb");
    CONFIG_FIELD(ConfigDuration, timeout, = std::chrono::seconds(30));
    CONFIG_FIELD(bool, ssl_enabled, = false);
    
    // Field iteration for reflection
    template<typename F>
    void for_each_field(F&& func) {
        func("host", host);
        func("port", port);
        func("username", username);
        func("password", password);
        func("database", database);
        func("timeout", timeout);
        func("ssl_enabled", ssl_enabled);
    }
    
    template<typename F>
    void for_each_field(F&& func) const {
        func("host", host);
        func("port", port);
        func("username", username);
        func("password", password);
        func("database", database);
        func("timeout", timeout);
        func("ssl_enabled", ssl_enabled);
    }
};

} // namespace config
```

### 4. Configuration Validation System

```cpp
// include/config_validator.h
#pragma once
#include "config_types.h"
#include <functional>
#include <regex>
#include <limits>

namespace config {
namespace validators {

// Range validators
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::function<bool(const T&)>>
range(const T& min_val, const T& max_val) {
    return [min_val, max_val](const T& value) {
        return value >= min_val && value <= max_val;
    };
}

template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::function<bool(const T&)>>
min_value(const T& min_val) {
    return [min_val](const T& value) {
        return value >= min_val;
    };
}

template<typename T>
std::enable_if_t<std::is_arithmetic_v<T>, std::function<bool(const T&)>>
max_value(const T& max_val) {
    return [max_val](const T& value) {
        return value <= max_val;
    };
}

// String validators
inline std::function<bool(const std::string&)> min_length(size_t min_len) {
    return [min_len](const std::string& value) {
        return value.length() >= min_len;
    };
}

inline std::function<bool(const std::string&)> max_length(size_t max_len) {
    return [max_len](const std::string& value) {
        return value.length() <= max_len;
    };
}

inline std::function<bool(const std::string&)> regex_match(const std::string& pattern) {
    return [regex = std::regex(pattern)](const std::string& value) {
        return std::regex_match(value, regex);
    };
}

inline std::function<bool(const std::string&)> not_empty() {
    return [](const std::string& value) {
        return !value.empty();
    };
}

// Email validator
inline std::function<bool(const std::string&)> email() {
    static const std::regex email_regex(
        R"(^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$)"
    );
    return [](const std::string& value) {
        return std::regex_match(value, email_regex);
    };
}

// URL validator
inline std::function<bool(const std::string&)> url() {
    static const std::regex url_regex(
        R"(^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?$)"
    );
    return [](const std::string& value) {
        return std::regex_match(value, url_regex);
    };
}

// IP address validator
inline std::function<bool(const std::string&)> ip_address() {
    static const std::regex ip_regex(
        R"(^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)"
    );
    return [](const std::string& value) {
        return std::regex_match(value, ip_regex);
    };
}

// Port number validator
inline std::function<bool(int)> port_number() {
    return range<int>(1, 65535);
}

// Duration validators
inline std::function<bool(const ConfigDuration&)> max_duration(const ConfigDuration& max_dur) {
    return [max_dur](const ConfigDuration& value) {
        return value <= max_dur;
    };
}

inline std::function<bool(const ConfigDuration&)> min_duration(const ConfigDuration& min_dur) {
    return [min_dur](const ConfigDuration& value) {
        return value >= min_dur;
    };
}

// Container validators
template<typename Container>
std::function<bool(const Container&)> min_size(size_t min_sz) {
    return [min_sz](const Container& container) {
        return container.size() >= min_sz;
    };
}

template<typename Container>
std::function<bool(const Container&)> max_size(size_t max_sz) {
    return [max_sz](const Container& container) {
        return container.size() <= max_sz;
    };
}

template<typename Container>
std::function<bool(const Container&)> not_empty_container() {
    return [](const Container& container) {
        return !container.empty();
    };
}

// Composite validators
template<typename... Validators>
auto all_of(Validators&&... validators) {
    return [validators...](const auto& value) {
        return (validators(value) && ...);
    };
}

template<typename... Validators>
auto any_of(Validators&&... validators) {
    return [validators...](const auto& value) {
        return (validators(value) || ...);
    };
}

template<typename Validator>
auto not_validator(Validator&& validator) {
    return [validator = std::forward<Validator>(validator)](const auto& value) {
        return !validator(value);
    };
}

// Custom validator helper
template<typename T, typename F>
std::function<bool(const T&)> custom(F&& func) {
    return std::forward<F>(func);
}

} // namespace validators
} // namespace config
```

### 5. Usage Examples

```cpp
// examples/basic_usage.cpp
#include <iostream>
#include "config_manager.h"
#include "config_validator.h"

using namespace config;

// Example configuration structure
struct ServerConfig {
    std::string host = "localhost";
    int port = 8080;
    ConfigDuration timeout = std::chrono::seconds(30);
    bool ssl_enabled = false;
    std::vector<std::string> allowed_origins;
    
    // Reflection-like field iteration
    template<typename F>
    void for_each_field(F&& func) {
        func("host", host);
        func("port", port);
        func("timeout", timeout);
        func("ssl_enabled", ssl_enabled);
        func("allowed_origins", allowed_origins);
    }
    
    template<typename F>
    void for_each_field(F&& func) const {
        func("host", host);
        func("port", port);
        func("timeout", timeout);
        func("ssl_enabled", ssl_enabled);
        func("allowed_origins", allowed_origins);
    }
};

void demonstrate_basic_usage() {
    std::cout << "\n=== Basic Configuration Usage ===" << std::endl;
    
    // Create configuration manager
    ConfigManager config;
    
    // Set some values
    config.set("server.host", std::string("0.0.0.0"));
    config.set("server.port", 9090);
    config.set("server.timeout", std::chrono::seconds(60));
    config.set("server.ssl_enabled", true);
    
    config.set("database.host", std::string("db.example.com"));
    config.set("database.port", 5432);
    config.set("database.username", std::string("admin"));
    config.set("database.password", std::string("secret123"));
    
    // Get values with type safety
    auto server_host = config.get<std::string>("server.host");
    auto server_port = config.get<int>("server.port");
    auto ssl_enabled = config.get<bool>("server.ssl_enabled");
    
    std::cout << "Server configuration:" << std::endl;
    std::cout << "  Host: " << server_host.value_or("unknown") << std::endl;
    std::cout << "  Port: " << server_port.value_or(0) << std::endl;
    std::cout << "  SSL: " << std::boolalpha << ssl_enabled.value_or(false) << std::endl;
    
    // Get values with defaults
    auto cache_ttl = config.get_or("cache.ttl", std::chrono::minutes(5));
    auto max_connections = config.get_or("server.max_connections", 100);
    
    std::cout << "Defaults:" << std::endl;
    std::cout << "  Cache TTL: " << cache_ttl.count() << "ms" << std::endl;
    std::cout << "  Max connections: " << max_connections << std::endl;
}

void demonstrate_schema_validation() {
    std::cout << "\n=== Schema Validation ===" << std::endl;
    
    // Create schema
    auto schema = std::make_unique<ConfigSchema>();
    
    schema->add_field<std::string>("server.host", "Server hostname", true, "localhost")
           .add_field_with_validator<int>("server.port", validators::port_number(), 
                                         "Server port number", true, 8080)
           .add_field<ConfigDuration>("server.timeout", "Connection timeout", false, 
                                     std::chrono::seconds(30))
           .add_field_with_validator<std::string>("database.host", validators::not_empty(), 
                                                "Database hostname", true)
           .add_field_with_validator<int>("database.port", validators::range(1, 65535), 
                                        "Database port", true, 5432);
    
    ConfigManager config(std::move(schema));
    
    // Set valid values
    config.set("server.host", std::string("api.example.com"));
    config.set("server.port", 443);
    config.set("database.host", std::string("db.example.com"));
    config.set("database.port", 5432);
    
    // Validate
    if (config.validate()) {
        std::cout << "Configuration is valid!" << std::endl;
    } else {
        std::cout << "Configuration validation failed:" << std::endl;
        for (const auto& error : config.get_validation_errors()) {
            std::cout << "  " << error << std::endl;
        }
    }
    
    // Test invalid values
    config.set("server.port", 70000);  // Invalid port
    config.set("database.host", std::string(""));  // Empty string
    
    if (!config.validate()) {
        std::cout << "\nValidation errors after setting invalid values:" << std::endl;
        for (const auto& error : config.get_validation_errors()) {
            std::cout << "  " << error << std::endl;
        }
    }
}

void demonstrate_struct_integration() {
    std::cout << "\n=== Struct Integration ===" << std::endl;
    
    ConfigManager config;
    
    // Set configuration values
    config.set("host", std::string("production.example.com"));
    config.set("port", 443);
    config.set("timeout", std::chrono::minutes(2));
    config.set("ssl_enabled", true);
    
    // Load into struct
    ServerConfig server_config;
    config.load_struct(server_config);
    
    std::cout << "Loaded server configuration:" << std::endl;
    std::cout << "  Host: " << server_config.host << std::endl;
    std::cout << "  Port: " << server_config.port << std::endl;
    std::cout << "  Timeout: " << server_config.timeout.count() << "ms" << std::endl;
    std::cout << "  SSL: " << std::boolalpha << server_config.ssl_enabled << std::endl;
    
    // Modify struct and save back
    server_config.port = 8443;
    server_config.timeout = std::chrono::seconds(45);
    
    config.save_struct(server_config);
    
    std::cout << "\nAfter modifying struct:" << std::endl;
    std::cout << "  Port in config: " << config.get_or("port", 0) << std::endl;
    std::cout << "  Timeout in config: " << config.get_or("timeout", std::chrono::seconds(0)).count() << "ms" << std::endl;
}

void demonstrate_file_formats() {
    std::cout << "\n=== File Format Support ===" << std::endl;
    
    ConfigManager config;
    
    // Set some configuration
    config.set("app.name", std::string("MyApplication"));
    config.set("app.version", std::string("1.0.0"));
    config.set("server.host", std::string("localhost"));
    config.set("server.port", 8080);
    config.set("logging.level", std::string("INFO"));
    config.set("logging.file", std::string("/var/log/app.log"));
    
    // Export to JSON
    std::string json = config.to_json(true);
    std::cout << "JSON format:" << std::endl;
    std::cout << json << std::endl;
    
    // Export to INI
    std::string ini = config.to_ini();
    std::cout << "\nINI format:" << std::endl;
    std::cout << ini << std::endl;
    
    // Test loading from JSON
    ConfigManager config2;
    if (config2.load_from_json(json)) {
        std::cout << "\nSuccessfully loaded from JSON" << std::endl;
        std::cout << "App name: " << config2.get_or("app.name", std::string("unknown")) << std::endl;
    }
}

int main() {
    try {
        demonstrate_basic_usage();
        demonstrate_schema_validation();
        demonstrate_struct_integration();
        demonstrate_file_formats();
        
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
project(TypeSafeConfig)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/config_manager.cpp
    src/config_parser.cpp
    src/config_validator.cpp
    src/config_serializer.cpp
)

# Create library
add_library(config_lib ${SOURCES})

# Examples
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage config_lib)

add_executable(advanced_features examples/advanced_features.cpp)
target_link_libraries(advanced_features config_lib)

# Tests
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(config_tests
        tests/test_config_types.cpp
        tests/test_config_validation.cpp
        tests/test_config_serialization.cpp
    )
    target_link_libraries(config_tests GTest::gtest_main config_lib)
    
    enable_testing()
    add_test(NAME Config_Tests COMMAND config_tests)
endif()

# Compiler options
if(MSVC)
    target_compile_options(config_lib PRIVATE /W4 /std:c++17)
else()
    target_compile_options(config_lib PRIVATE -Wall -Wextra -Wpedantic -std=c++17)
endif()
```

## Expected Learning Outcomes

After completing this project, you should master:

1. **Template Metaprogramming**
   - Type traits and SFINAE techniques
   - Variadic templates and parameter packs
   - Compile-time computation with constexpr

2. **Modern C++ Features**
   - std::optional for nullable values
   - std::variant for type-safe unions
   - Perfect forwarding and universal references

3. **API Design Principles**
   - Type safety at compile time
   - Fluent interfaces and method chaining
   - Error handling strategies

4. **Generic Programming**
   - Creating reusable, type-safe components
   - Template specialization techniques
   - Concept-like constraints using SFINAE

## Extensions and Improvements

1. **Advanced Features**
   - Hot configuration reloading
   - Configuration change notifications
   - Encrypted configuration values
   - Remote configuration sources

2. **Integration Support**
   - Command-line argument parsing
   - Docker/Kubernetes integration
   - Cloud configuration services

3. **Performance Optimizations**
   - Memory-efficient storage
   - Lazy evaluation of configurations
   - Configuration caching strategies

This project demonstrates advanced C++ template programming techniques while creating a practical, reusable configuration system.
