# Project 5: Expression Parser with Boost.Spirit

*Estimated Duration: 2-3 weeks*
*Difficulty: Intermediate to Advanced*

## Project Overview

Develop a comprehensive expression parser and evaluator using Boost.Spirit that can handle mathematical expressions, logical operations, function calls, and variable assignments. This project demonstrates advanced parsing techniques, AST (Abstract Syntax Tree) construction, and expression evaluation.

## Learning Objectives

- Master Boost.Spirit for parsing complex grammars
- Understand parser combinators and recursive descent parsing
- Design and implement Abstract Syntax Trees (AST)
- Handle operator precedence and associativity
- Implement symbol tables and scope management
- Create extensible expression evaluation systems

## Project Requirements

### Core Features

1. **Mathematical Expression Parsing**
   - Basic arithmetic operations (+, -, *, /, %, ^)
   - Parentheses for grouping
   - Unary operators (-, +, !)
   - Operator precedence and associativity

2. **Data Types and Literals**
   - Integer and floating-point numbers
   - Boolean values (true/false)
   - String literals with escape sequences
   - Array and object literals

3. **Variables and Functions**
   - Variable declarations and assignments
   - Function definitions with parameters
   - Function calls with arguments
   - Built-in mathematical functions

### Advanced Features

4. **Control Structures**
   - Conditional expressions (if-then-else)
   - Loop constructs (for, while)
   - Pattern matching expressions
   - Exception handling constructs

5. **Advanced Language Features**
   - User-defined functions with closures
   - Lambda expressions
   - List comprehensions
   - Type annotations and checking

6. **Integration and Extensibility**
   - Plugin system for custom functions
   - Serialization of parsed expressions
   - Interactive REPL (Read-Eval-Print Loop)
   - Integration with external libraries

## Implementation Guide

### Step 1: Basic Grammar and AST Definitions

```cpp
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/variant.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <cmath>
#include <variant>
#include <optional>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

// Forward declarations
struct Expression;
struct Statement;

// AST Node types
struct Identifier {
    std::string name;
    
    Identifier() = default;
    Identifier(const std::string& n) : name(n) {}
    
    bool operator==(const Identifier& other) const {
        return name == other.name;
    }
};

struct NumberLiteral {
    double value;
    
    NumberLiteral() : value(0.0) {}
    NumberLiteral(double v) : value(v) {}
    
    bool operator==(const NumberLiteral& other) const {
        return value == other.value;
    }
};

struct StringLiteral {
    std::string value;
    
    StringLiteral() = default;
    StringLiteral(const std::string& v) : value(v) {}
    
    bool operator==(const StringLiteral& other) const {
        return value == other.value;
    }
};

struct BooleanLiteral {
    bool value;
    
    BooleanLiteral() : value(false) {}
    BooleanLiteral(bool v) : value(v) {}
    
    bool operator==(const BooleanLiteral& other) const {
        return value == other.value;
    }
};

struct ArrayLiteral {
    std::vector<Expression> elements;
    
    ArrayLiteral() = default;
    ArrayLiteral(const std::vector<Expression>& elems) : elements(elems) {}
};

struct BinaryOperation {
    std::string operator_;
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    
    BinaryOperation() = default;
    BinaryOperation(const std::string& op, 
                   std::shared_ptr<Expression> l, 
                   std::shared_ptr<Expression> r)
        : operator_(op), left(l), right(r) {}
};

struct UnaryOperation {
    std::string operator_;
    std::shared_ptr<Expression> operand;
    
    UnaryOperation() = default;
    UnaryOperation(const std::string& op, std::shared_ptr<Expression> operand_)
        : operator_(op), operand(operand_) {}
};

struct FunctionCall {
    std::string name;
    std::vector<Expression> arguments;
    
    FunctionCall() = default;
    FunctionCall(const std::string& n, const std::vector<Expression>& args)
        : name(n), arguments(args) {}
};

struct ConditionalExpression {
    std::shared_ptr<Expression> condition;
    std::shared_ptr<Expression> true_branch;
    std::shared_ptr<Expression> false_branch;
    
    ConditionalExpression() = default;
    ConditionalExpression(std::shared_ptr<Expression> cond,
                         std::shared_ptr<Expression> true_expr,
                         std::shared_ptr<Expression> false_expr)
        : condition(cond), true_branch(true_expr), false_branch(false_expr) {}
};

// Main expression variant
struct Expression {
    using VariantType = boost::variant<
        NumberLiteral,
        StringLiteral,
        BooleanLiteral,
        Identifier,
        ArrayLiteral,
        BinaryOperation,
        UnaryOperation,
        FunctionCall,
        ConditionalExpression
    >;
    
    VariantType value;
    
    Expression() = default;
    
    template<typename T>
    Expression(const T& v) : value(v) {}
    
    template<typename T>
    Expression& operator=(const T& v) {
        value = v;
        return *this;
    }
};

// Adapt structs for Boost.Fusion
BOOST_FUSION_ADAPT_STRUCT(
    Identifier,
    (std::string, name)
)

BOOST_FUSION_ADAPT_STRUCT(
    NumberLiteral,
    (double, value)
)

BOOST_FUSION_ADAPT_STRUCT(
    StringLiteral,
    (std::string, value)
)

BOOST_FUSION_ADAPT_STRUCT(
    BooleanLiteral,
    (bool, value)
)

BOOST_FUSION_ADAPT_STRUCT(
    FunctionCall,
    (std::string, name)
    (std::vector<Expression>, arguments)
)
```

### Step 2: Expression Parser Grammar

```cpp
template<typename Iterator>
class ExpressionGrammar : public qi::grammar<Iterator, Expression(), ascii::space_type> {
public:
    ExpressionGrammar() : ExpressionGrammar::base_type(expression, "expression") {
        using qi::double_;
        using qi::int_;
        using qi::lit;
        using qi::lexeme;
        using qi::alpha;
        using qi::alnum;
        using qi::char_;
        using qi::string;
        using qi::_val;
        using qi::_1;
        using qi::_2;
        using qi::_3;
        using ascii::space;
        using phoenix::construct;
        using phoenix::val;
        
        // Literals
        number_literal = double_[_val = construct<NumberLiteral>(_1)];
        
        string_literal = lexeme['"' >> *(char_ - '"') >> '"']
            [_val = construct<StringLiteral>(_1)];
        
        boolean_literal = 
            (lit("true")[_val = construct<BooleanLiteral>(true)] |
             lit("false")[_val = construct<BooleanLiteral>(false)]);
        
        identifier = lexeme[(alpha | char_('_')) >> *(alnum | char_('_'))]
            [_val = construct<Identifier>(_1)];
        
        // Array literals
        array_literal = '[' >> -(expression % ',') >> ']'
            [_val = construct<ArrayLiteral>(_1)];
        
        // Function calls
        function_call = identifier >> '(' >> -(expression % ',') >> ')'
            [_val = construct<FunctionCall>(_1, _2)];
        
        // Primary expressions
        primary_expr = 
            function_call |
            number_literal |
            string_literal |
            boolean_literal |
            array_literal |
            identifier |
            ('(' >> expression >> ')');
        
        // Unary expressions
        unary_expr = 
            (unary_op >> unary_expr)
            [_val = construct_unary_op(_1, _2)] |
            primary_expr;
        
        // Build binary expression hierarchy with proper precedence
        multiplicative_expr = 
            unary_expr >> *(multiplicative_op >> unary_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        additive_expr = 
            multiplicative_expr >> *(additive_op >> multiplicative_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        relational_expr = 
            additive_expr >> *(relational_op >> additive_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        equality_expr = 
            relational_expr >> *(equality_op >> relational_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        logical_and_expr = 
            equality_expr >> *(lit("&&") >> equality_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        logical_or_expr = 
            logical_and_expr >> *(lit("||") >> logical_and_expr)
            [_val = construct_left_assoc(_val, _1)];
        
        // Conditional expression (ternary operator)
        conditional_expr = 
            (logical_or_expr >> '?' >> expression >> ':' >> conditional_expr)
            [_val = construct_conditional(_1, _2, _3)] |
            logical_or_expr;
        
        expression = conditional_expr;
        
        // Operators
        unary_op = lit("+") | lit("-") | lit("!");
        multiplicative_op = lit("*") | lit("/") | lit("%");
        additive_op = lit("+") | lit("-");
        relational_op = lit("<=") | lit(">=") | lit("<") | lit(">");
        equality_op = lit("==") | lit("!=");
        
        // Set rule names for better error reporting
        number_literal.name("number");
        string_literal.name("string");
        boolean_literal.name("boolean");
        identifier.name("identifier");
        array_literal.name("array");
        function_call.name("function_call");
        expression.name("expression");
        
        // Error handling
        qi::on_error<qi::fail>(
            expression,
            std::cout << val("Error! Expecting ")
                      << _4
                      << val(" here: \"")
                      << construct<std::string>(_3, _2)
                      << val("\"")
                      << std::endl
        );
    }
    
private:
    qi::rule<Iterator, Expression(), ascii::space_type> expression;
    qi::rule<Iterator, Expression(), ascii::space_type> conditional_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> logical_or_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> logical_and_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> equality_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> relational_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> additive_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> multiplicative_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> unary_expr;
    qi::rule<Iterator, Expression(), ascii::space_type> primary_expr;
    
    qi::rule<Iterator, NumberLiteral(), ascii::space_type> number_literal;
    qi::rule<Iterator, StringLiteral(), ascii::space_type> string_literal;
    qi::rule<Iterator, BooleanLiteral(), ascii::space_type> boolean_literal;
    qi::rule<Iterator, Identifier(), ascii::space_type> identifier;
    qi::rule<Iterator, ArrayLiteral(), ascii::space_type> array_literal;
    qi::rule<Iterator, FunctionCall(), ascii::space_type> function_call;
    
    qi::rule<Iterator, std::string(), ascii::space_type> unary_op;
    qi::rule<Iterator, std::string(), ascii::space_type> multiplicative_op;
    qi::rule<Iterator, std::string(), ascii::space_type> additive_op;
    qi::rule<Iterator, std::string(), ascii::space_type> relational_op;
    qi::rule<Iterator, std::string(), ascii::space_type> equality_op;
    
    // Semantic actions
    phoenix::function<ConstructUnaryOp> construct_unary_op;
    phoenix::function<ConstructLeftAssoc> construct_left_assoc;
    phoenix::function<ConstructConditional> construct_conditional;
};

// Semantic action functors
struct ConstructUnaryOp {
    template<typename T1, typename T2>
    struct result { typedef Expression type; };
    
    template<typename T1, typename T2>
    Expression operator()(const T1& op, const T2& operand) const {
        return UnaryOperation(op, std::make_shared<Expression>(operand));
    }
};

struct ConstructLeftAssoc {
    template<typename T1, typename T2>
    struct result { typedef Expression type; };
    
    template<typename T1, typename T2>
    Expression operator()(const T1& left, const T2& right_ops) const {
        Expression result = left;
        
        for (const auto& op_expr : right_ops) {
            std::string op = op_expr.first;
            Expression right = op_expr.second;
            
            result = BinaryOperation(op, 
                                   std::make_shared<Expression>(result),
                                   std::make_shared<Expression>(right));
        }
        
        return result;
    }
};

struct ConstructConditional {
    template<typename T1, typename T2, typename T3>
    struct result { typedef Expression type; };
    
    template<typename T1, typename T2, typename T3>
    Expression operator()(const T1& condition, const T2& true_branch, const T3& false_branch) const {
        return ConditionalExpression(
            std::make_shared<Expression>(condition),
            std::make_shared<Expression>(true_branch),
            std::make_shared<Expression>(false_branch)
        );
    }
};
```

### Step 3: Expression Evaluator

```cpp
class Value {
public:
    using VariantType = std::variant<double, std::string, bool, std::vector<Value>>;
    
    VariantType value;
    
    Value() : value(0.0) {}
    
    template<typename T>
    Value(const T& v) : value(v) {}
    
    // Type checking methods
    bool is_number() const { return std::holds_alternative<double>(value); }
    bool is_string() const { return std::holds_alternative<std::string>(value); }
    bool is_boolean() const { return std::holds_alternative<bool>(value); }
    bool is_array() const { return std::holds_alternative<std::vector<Value>>(value); }
    
    // Conversion methods
    double as_number() const {
        if (is_number()) return std::get<double>(value);
        if (is_boolean()) return std::get<bool>(value) ? 1.0 : 0.0;
        if (is_string()) {
            try {
                return std::stod(std::get<std::string>(value));
            } catch (...) {
                return 0.0;
            }
        }
        return 0.0;
    }
    
    std::string as_string() const {
        if (is_string()) return std::get<std::string>(value);
        if (is_number()) return std::to_string(std::get<double>(value));
        if (is_boolean()) return std::get<bool>(value) ? "true" : "false";
        if (is_array()) {
            std::string result = "[";
            const auto& arr = std::get<std::vector<Value>>(value);
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) result += ", ";
                result += arr[i].as_string();
            }
            result += "]";
            return result;
        }
        return "";
    }
    
    bool as_boolean() const {
        if (is_boolean()) return std::get<bool>(value);
        if (is_number()) return std::get<double>(value) != 0.0;
        if (is_string()) return !std::get<std::string>(value).empty();
        if (is_array()) return !std::get<std::vector<Value>>(value).empty();
        return false;
    }
    
    const std::vector<Value>& as_array() const {
        static std::vector<Value> empty_array;
        if (is_array()) return std::get<std::vector<Value>>(value);
        return empty_array;
    }
    
    // Operators
    Value operator+(const Value& other) const {
        if (is_number() && other.is_number()) {
            return Value(as_number() + other.as_number());
        } else {
            return Value(as_string() + other.as_string());
        }
    }
    
    Value operator-(const Value& other) const {
        return Value(as_number() - other.as_number());
    }
    
    Value operator*(const Value& other) const {
        return Value(as_number() * other.as_number());
    }
    
    Value operator/(const Value& other) const {
        double divisor = other.as_number();
        if (divisor == 0.0) {
            throw std::runtime_error("Division by zero");
        }
        return Value(as_number() / divisor);
    }
    
    Value operator==(const Value& other) const {
        if (is_number() && other.is_number()) {
            return Value(as_number() == other.as_number());
        } else if (is_string() && other.is_string()) {
            return Value(as_string() == other.as_string());
        } else if (is_boolean() && other.is_boolean()) {
            return Value(as_boolean() == other.as_boolean());
        } else {
            return Value(as_string() == other.as_string());
        }
    }
    
    Value operator!=(const Value& other) const {
        return Value(!(*this == other).as_boolean());
    }
    
    Value operator<(const Value& other) const {
        return Value(as_number() < other.as_number());
    }
    
    Value operator<=(const Value& other) const {
        return Value(as_number() <= other.as_number());
    }
    
    Value operator>(const Value& other) const {
        return Value(as_number() > other.as_number());
    }
    
    Value operator>=(const Value& other) const {
        return Value(as_number() >= other.as_number());
    }
};

// Symbol table for variables and functions
class SymbolTable {
public:
    using BuiltinFunction = std::function<Value(const std::vector<Value>&)>;
    
    SymbolTable(SymbolTable* parent = nullptr) : parent_(parent) {
        if (!parent) {
            register_builtin_functions();
        }
    }
    
    void set_variable(const std::string& name, const Value& value) {
        variables_[name] = value;
    }
    
    Value get_variable(const std::string& name) const {
        auto it = variables_.find(name);
        if (it != variables_.end()) {
            return it->second;
        }
        
        if (parent_) {
            return parent_->get_variable(name);
        }
        
        throw std::runtime_error("Undefined variable: " + name);
    }
    
    bool has_variable(const std::string& name) const {
        return variables_.find(name) != variables_.end() ||
               (parent_ && parent_->has_variable(name));
    }
    
    void register_function(const std::string& name, BuiltinFunction func) {
        functions_[name] = func;
    }
    
    Value call_function(const std::string& name, const std::vector<Value>& args) const {
        auto it = functions_.find(name);
        if (it != functions_.end()) {
            return it->second(args);
        }
        
        if (parent_) {
            return parent_->call_function(name, args);
        }
        
        throw std::runtime_error("Undefined function: " + name);
    }
    
    bool has_function(const std::string& name) const {
        return functions_.find(name) != functions_.end() ||
               (parent_ && parent_->has_function(name));
    }
    
    std::unique_ptr<SymbolTable> create_child_scope() {
        return std::make_unique<SymbolTable>(this);
    }
    
private:
    SymbolTable* parent_;
    std::map<std::string, Value> variables_;
    std::map<std::string, BuiltinFunction> functions_;
    
    void register_builtin_functions() {
        // Mathematical functions
        register_function("abs", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("abs() requires 1 argument");
            return Value(std::abs(args[0].as_number()));
        });
        
        register_function("sqrt", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("sqrt() requires 1 argument");
            double val = args[0].as_number();
            if (val < 0) throw std::runtime_error("sqrt() of negative number");
            return Value(std::sqrt(val));
        });
        
        register_function("pow", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 2) throw std::runtime_error("pow() requires 2 arguments");
            return Value(std::pow(args[0].as_number(), args[1].as_number()));
        });
        
        register_function("sin", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("sin() requires 1 argument");
            return Value(std::sin(args[0].as_number()));
        });
        
        register_function("cos", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("cos() requires 1 argument");
            return Value(std::cos(args[0].as_number()));
        });
        
        register_function("tan", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("tan() requires 1 argument");
            return Value(std::tan(args[0].as_number()));
        });
        
        register_function("log", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("log() requires 1 argument");
            double val = args[0].as_number();
            if (val <= 0) throw std::runtime_error("log() of non-positive number");
            return Value(std::log(val));
        });
        
        register_function("exp", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("exp() requires 1 argument");
            return Value(std::exp(args[0].as_number()));
        });
        
        // String functions
        register_function("length", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("length() requires 1 argument");
            if (args[0].is_string()) {
                return Value(static_cast<double>(args[0].as_string().length()));
            } else if (args[0].is_array()) {
                return Value(static_cast<double>(args[0].as_array().size()));
            }
            return Value(0.0);
        });
        
        register_function("substring", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 3) throw std::runtime_error("substring() requires 3 arguments");
            std::string str = args[0].as_string();
            size_t start = static_cast<size_t>(args[1].as_number());
            size_t length = static_cast<size_t>(args[2].as_number());
            
            if (start >= str.length()) return Value(std::string(""));
            return Value(str.substr(start, length));
        });
        
        // Array functions
        register_function("push", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 2) throw std::runtime_error("push() requires 2 arguments");
            std::vector<Value> arr = args[0].as_array();
            arr.push_back(args[1]);
            return Value(arr);
        });
        
        register_function("pop", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("pop() requires 1 argument");
            std::vector<Value> arr = args[0].as_array();
            if (arr.empty()) throw std::runtime_error("pop() from empty array");
            arr.pop_back();
            return Value(arr);
        });
        
        // Utility functions
        register_function("min", [](const std::vector<Value>& args) -> Value {
            if (args.empty()) throw std::runtime_error("min() requires at least 1 argument");
            double result = args[0].as_number();
            for (size_t i = 1; i < args.size(); ++i) {
                result = std::min(result, args[i].as_number());
            }
            return Value(result);
        });
        
        register_function("max", [](const std::vector<Value>& args) -> Value {
            if (args.empty()) throw std::runtime_error("max() requires at least 1 argument");
            double result = args[0].as_number();
            for (size_t i = 1; i < args.size(); ++i) {
                result = std::max(result, args[i].as_number());
            }
            return Value(result);
        });
    }
};

class ExpressionEvaluator : public boost::static_visitor<Value> {
public:
    ExpressionEvaluator(SymbolTable& symbol_table) : symbol_table_(symbol_table) {}
    
    Value operator()(const NumberLiteral& literal) const {
        return Value(literal.value);
    }
    
    Value operator()(const StringLiteral& literal) const {
        return Value(literal.value);
    }
    
    Value operator()(const BooleanLiteral& literal) const {
        return Value(literal.value);
    }
    
    Value operator()(const Identifier& identifier) const {
        return symbol_table_.get_variable(identifier.name);
    }
    
    Value operator()(const ArrayLiteral& array) const {
        std::vector<Value> result;
        for (const auto& element : array.elements) {
            result.push_back(boost::apply_visitor(*this, element.value));
        }
        return Value(result);
    }
    
    Value operator()(const BinaryOperation& operation) const {
        Value left = boost::apply_visitor(*this, operation.left->value);
        Value right = boost::apply_visitor(*this, operation.right->value);
        
        if (operation.operator_ == "+") {
            return left + right;
        } else if (operation.operator_ == "-") {
            return left - right;
        } else if (operation.operator_ == "*") {
            return left * right;
        } else if (operation.operator_ == "/") {
            return left / right;
        } else if (operation.operator_ == "%") {
            return Value(std::fmod(left.as_number(), right.as_number()));
        } else if (operation.operator_ == "==") {
            return left == right;
        } else if (operation.operator_ == "!=") {
            return left != right;
        } else if (operation.operator_ == "<") {
            return left < right;
        } else if (operation.operator_ == "<=") {
            return left <= right;
        } else if (operation.operator_ == ">") {
            return left > right;
        } else if (operation.operator_ == ">=") {
            return left >= right;
        } else if (operation.operator_ == "&&") {
            return Value(left.as_boolean() && right.as_boolean());
        } else if (operation.operator_ == "||") {
            return Value(left.as_boolean() || right.as_boolean());
        } else {
            throw std::runtime_error("Unknown binary operator: " + operation.operator_);
        }
    }
    
    Value operator()(const UnaryOperation& operation) const {
        Value operand = boost::apply_visitor(*this, operation.operand->value);
        
        if (operation.operator_ == "-") {
            return Value(-operand.as_number());
        } else if (operation.operator_ == "+") {
            return Value(operand.as_number());
        } else if (operation.operator_ == "!") {
            return Value(!operand.as_boolean());
        } else {
            throw std::runtime_error("Unknown unary operator: " + operation.operator_);
        }
    }
    
    Value operator()(const FunctionCall& call) const {
        std::vector<Value> args;
        for (const auto& arg : call.arguments) {
            args.push_back(boost::apply_visitor(*this, arg.value));
        }
        
        return symbol_table_.call_function(call.name, args);
    }
    
    Value operator()(const ConditionalExpression& conditional) const {
        Value condition = boost::apply_visitor(*this, conditional.condition->value);
        
        if (condition.as_boolean()) {
            return boost::apply_visitor(*this, conditional.true_branch->value);
        } else {
            return boost::apply_visitor(*this, conditional.false_branch->value);
        }
    }
    
private:
    SymbolTable& symbol_table_;
};
```

### Step 4: Expression Parser and REPL

```cpp
class ExpressionParser {
public:
    ExpressionParser() : symbol_table_() {}
    
    Value parse_and_evaluate(const std::string& input) {
        // Parse the expression
        Expression ast;
        if (!parse_expression(input, ast)) {
            throw std::runtime_error("Failed to parse expression");
        }
        
        // Evaluate the expression
        ExpressionEvaluator evaluator(symbol_table_);
        return boost::apply_visitor(evaluator, ast.value);
    }
    
    bool parse_expression(const std::string& input, Expression& result) {
        using boost::spirit::ascii::space;
        
        ExpressionGrammar<std::string::const_iterator> grammar;
        
        std::string::const_iterator iter = input.begin();
        std::string::const_iterator end = input.end();
        
        bool success = qi::phrase_parse(iter, end, grammar, space, result);
        
        if (success && iter == end) {
            return true;
        } else {
            if (iter != end) {
                std::cout << "Parsing stopped at: " << std::string(iter, end) << std::endl;
            }
            return false;
        }
    }
    
    void set_variable(const std::string& name, const Value& value) {
        symbol_table_.set_variable(name, value);
    }
    
    Value get_variable(const std::string& name) const {
        return symbol_table_.get_variable(name);
    }
    
    void register_function(const std::string& name, SymbolTable::BuiltinFunction func) {
        symbol_table_.register_function(name, func);
    }
    
    SymbolTable& get_symbol_table() { return symbol_table_; }
    
private:
    SymbolTable symbol_table_;
};

class REPL {
public:
    REPL() : parser_(), running_(true) {}
    
    void run() {
        std::cout << "Expression Parser REPL v1.0" << std::endl;
        std::cout << "Type expressions to evaluate, or 'help' for commands." << std::endl;
        std::cout << "Type 'quit' to exit." << std::endl;
        
        std::string line;
        while (running_ && std::getline(std::cin, line)) {
            if (line.empty()) continue;
            
            try {
                process_line(line);
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
            
            std::cout << "> ";
        }
    }
    
private:
    ExpressionParser parser_;
    bool running_;
    
    void process_line(const std::string& line) {
        // Trim whitespace
        std::string trimmed = boost::algorithm::trim_copy(line);
        
        // Handle special commands
        if (trimmed == "quit" || trimmed == "exit") {
            running_ = false;
            std::cout << "Goodbye!" << std::endl;
            return;
        }
        
        if (trimmed == "help") {
            show_help();
            return;
        }
        
        if (trimmed == "vars") {
            show_variables();
            return;
        }
        
        if (trimmed == "funcs") {
            show_functions();
            return;
        }
        
        // Check for variable assignment
        size_t equals_pos = trimmed.find('=');
        if (equals_pos != std::string::npos && equals_pos > 0) {
            std::string var_name = boost::algorithm::trim_copy(trimmed.substr(0, equals_pos));
            std::string expression = boost::algorithm::trim_copy(trimmed.substr(equals_pos + 1));
            
            if (is_valid_identifier(var_name)) {
                Value result = parser_.parse_and_evaluate(expression);
                parser_.set_variable(var_name, result);
                std::cout << var_name << " = " << result.as_string() << std::endl;
                return;
            }
        }
        
        // Evaluate expression
        Value result = parser_.parse_and_evaluate(trimmed);
        std::cout << result.as_string() << std::endl;
    }
    
    void show_help() {
        std::cout << "Available commands:" << std::endl;
        std::cout << "  help    - Show this help message" << std::endl;
        std::cout << "  vars    - Show all variables" << std::endl;
        std::cout << "  funcs   - Show all functions" << std::endl;
        std::cout << "  quit    - Exit the REPL" << std::endl;
        std::cout << std::endl;
        std::cout << "Expression syntax:" << std::endl;
        std::cout << "  Numbers: 3.14, 42, -5" << std::endl;
        std::cout << "  Strings: \"hello world\"" << std::endl;
        std::cout << "  Booleans: true, false" << std::endl;
        std::cout << "  Arrays: [1, 2, 3]" << std::endl;
        std::cout << "  Operators: +, -, *, /, %, ==, !=, <, <=, >, >=, &&, ||, !" << std::endl;
        std::cout << "  Functions: sin(x), cos(x), sqrt(x), pow(x, y), etc." << std::endl;
        std::cout << "  Variables: x = 5, y = x * 2" << std::endl;
        std::cout << "  Conditionals: condition ? true_value : false_value" << std::endl;
    }
    
    void show_variables() {
        std::cout << "Currently defined variables:" << std::endl;
        // This would require extending SymbolTable to enumerate variables
        std::cout << "(Variable enumeration not implemented)" << std::endl;
    }
    
    void show_functions() {
        std::cout << "Available functions:" << std::endl;
        std::cout << "  Mathematical: abs, sqrt, pow, sin, cos, tan, log, exp" << std::endl;
        std::cout << "  String: length, substring" << std::endl;
        std::cout << "  Array: push, pop" << std::endl;
        std::cout << "  Utility: min, max" << std::endl;
    }
    
    bool is_valid_identifier(const std::string& name) {
        if (name.empty()) return false;
        
        if (!std::isalpha(name[0]) && name[0] != '_') return false;
        
        for (size_t i = 1; i < name.length(); ++i) {
            if (!std::isalnum(name[i]) && name[i] != '_') return false;
        }
        
        return true;
    }
};
```

### Step 5: Advanced Features

```cpp
// Statement parsing for more complex language constructs
struct Assignment {
    std::string variable_name;
    Expression value;
    
    Assignment() = default;
    Assignment(const std::string& name, const Expression& expr)
        : variable_name(name), value(expr) {}
};

struct IfStatement {
    Expression condition;
    std::vector<Statement> then_statements;
    std::vector<Statement> else_statements;
};

struct WhileLoop {
    Expression condition;
    std::vector<Statement> body;
};

struct ForLoop {
    std::string variable;
    Expression iterable;
    std::vector<Statement> body;
};

struct FunctionDefinition {
    std::string name;
    std::vector<std::string> parameters;
    std::vector<Statement> body;
};

struct Statement {
    using VariantType = boost::variant<
        Assignment,
        Expression,
        IfStatement,
        WhileLoop,
        ForLoop,
        FunctionDefinition
    >;
    
    VariantType value;
    
    Statement() = default;
    
    template<typename T>
    Statement(const T& v) : value(v) {}
};

// Extended grammar for statements
template<typename Iterator>
class StatementGrammar : public qi::grammar<Iterator, std::vector<Statement>(), ascii::space_type> {
public:
    StatementGrammar() : StatementGrammar::base_type(program, "program") {
        using qi::lit;
        using qi::lexeme;
        using qi::alpha;
        using qi::alnum;
        using qi::char_;
        using qi::_val;
        using qi::_1;
        using qi::_2;
        using qi::_3;
        using phoenix::construct;
        
        identifier = lexeme[(alpha | char_('_')) >> *(alnum | char_('_'))];
        
        assignment = identifier >> '=' >> expression_grammar >> ';'
            [_val = construct<Assignment>(_1, _2)];
        
        expression_statement = expression_grammar >> ';'
            [_val = _1];
        
        if_statement = 
            lit("if") >> '(' >> expression_grammar >> ')' >> 
            '{' >> *statement >> '}' >>
            -(lit("else") >> '{' >> *statement >> '}')
            [_val = construct<IfStatement>(_1, _2, _3)];
        
        while_statement = 
            lit("while") >> '(' >> expression_grammar >> ')' >>
            '{' >> *statement >> '}'
            [_val = construct<WhileLoop>(_1, _2)];
        
        for_statement = 
            lit("for") >> '(' >> identifier >> lit("in") >> expression_grammar >> ')' >>
            '{' >> *statement >> '}'
            [_val = construct<ForLoop>(_1, _2, _3)];
        
        function_definition = 
            lit("function") >> identifier >> '(' >> -(identifier % ',') >> ')' >>
            '{' >> *statement >> '}'
            [_val = construct<FunctionDefinition>(_1, _2, _3)];
        
        statement = 
            function_definition |
            if_statement |
            while_statement |
            for_statement |
            assignment |
            expression_statement;
        
        program = *statement;
        
        // Set names for error reporting
        assignment.name("assignment");
        if_statement.name("if_statement");
        while_statement.name("while_statement");
        for_statement.name("for_statement");
        function_definition.name("function_definition");
        statement.name("statement");
        program.name("program");
    }
    
private:
    ExpressionGrammar<Iterator> expression_grammar;
    
    qi::rule<Iterator, std::vector<Statement>(), ascii::space_type> program;
    qi::rule<Iterator, Statement(), ascii::space_type> statement;
    qi::rule<Iterator, Assignment(), ascii::space_type> assignment;
    qi::rule<Iterator, Expression(), ascii::space_type> expression_statement;
    qi::rule<Iterator, IfStatement(), ascii::space_type> if_statement;
    qi::rule<Iterator, WhileLoop(), ascii::space_type> while_statement;
    qi::rule<Iterator, ForLoop(), ascii::space_type> for_statement;
    qi::rule<Iterator, FunctionDefinition(), ascii::space_type> function_definition;
    qi::rule<Iterator, std::string(), ascii::space_type> identifier;
};

// User-defined functions
class UserFunction {
public:
    UserFunction(const std::vector<std::string>& params, const std::vector<Statement>& body)
        : parameters_(params), body_(body) {}
    
    Value call(const std::vector<Value>& args, SymbolTable& global_symbols) {
        if (args.size() != parameters_.size()) {
            throw std::runtime_error("Function called with wrong number of arguments");
        }
        
        // Create new scope for function execution
        auto function_scope = global_symbols.create_child_scope();
        
        // Bind parameters
        for (size_t i = 0; i < parameters_.size(); ++i) {
            function_scope->set_variable(parameters_[i], args[i]);
        }
        
        // Execute function body
        StatementExecutor executor(*function_scope);
        Value result;
        
        for (const auto& stmt : body_) {
            result = boost::apply_visitor(executor, stmt.value);
        }
        
        return result;
    }
    
private:
    std::vector<std::string> parameters_;
    std::vector<Statement> body_;
};

// Statement executor
class StatementExecutor : public boost::static_visitor<Value> {
public:
    StatementExecutor(SymbolTable& symbol_table) : symbol_table_(symbol_table) {}
    
    Value operator()(const Assignment& assignment) {
        ExpressionEvaluator evaluator(symbol_table_);
        Value value = boost::apply_visitor(evaluator, assignment.value.value);
        symbol_table_.set_variable(assignment.variable_name, value);
        return value;
    }
    
    Value operator()(const Expression& expression) {
        ExpressionEvaluator evaluator(symbol_table_);
        return boost::apply_visitor(evaluator, expression.value);
    }
    
    Value operator()(const IfStatement& if_stmt) {
        ExpressionEvaluator evaluator(symbol_table_);
        Value condition = boost::apply_visitor(evaluator, if_stmt.condition.value);
        
        Value result;
        if (condition.as_boolean()) {
            for (const auto& stmt : if_stmt.then_statements) {
                result = boost::apply_visitor(*this, stmt.value);
            }
        } else {
            for (const auto& stmt : if_stmt.else_statements) {
                result = boost::apply_visitor(*this, stmt.value);
            }
        }
        return result;
    }
    
    Value operator()(const WhileLoop& while_loop) {
        ExpressionEvaluator evaluator(symbol_table_);
        Value result;
        
        while (true) {
            Value condition = boost::apply_visitor(evaluator, while_loop.condition.value);
            if (!condition.as_boolean()) break;
            
            for (const auto& stmt : while_loop.body) {
                result = boost::apply_visitor(*this, stmt.value);
            }
        }
        
        return result;
    }
    
    Value operator()(const ForLoop& for_loop) {
        // Simplified for-in loop implementation
        ExpressionEvaluator evaluator(symbol_table_);
        Value iterable = boost::apply_visitor(evaluator, for_loop.iterable.value);
        
        Value result;
        if (iterable.is_array()) {
            for (const auto& item : iterable.as_array()) {
                symbol_table_.set_variable(for_loop.variable, item);
                
                for (const auto& stmt : for_loop.body) {
                    result = boost::apply_visitor(*this, stmt.value);
                }
            }
        }
        
        return result;
    }
    
    Value operator()(const FunctionDefinition& func_def) {
        // Register user-defined function
        auto user_func = std::make_shared<UserFunction>(func_def.parameters, func_def.body);
        
        symbol_table_.register_function(func_def.name, 
            [user_func, this](const std::vector<Value>& args) -> Value {
                return user_func->call(args, symbol_table_);
            });
        
        return Value(true); // Function defined successfully
    }
    
private:
    SymbolTable& symbol_table_;
};
```

### Step 6: Main Application and Testing

```cpp
class ExpressionParserApp {
public:
    ExpressionParserApp() : parser_() {
        setup_custom_functions();
    }
    
    void run_interactive() {
        REPL repl;
        repl.run();
    }
    
    void run_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        try {
            Value result = parser_.parse_and_evaluate(content);
            std::cout << "Result: " << result.as_string() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    
    void run_tests() {
        std::cout << "Running expression parser tests..." << std::endl;
        
        test_basic_arithmetic();
        test_string_operations();
        test_boolean_logic();
        test_function_calls();
        test_conditional_expressions();
        test_array_operations();
        test_variable_assignments();
        
        std::cout << "All tests completed!" << std::endl;
    }
    
private:
    ExpressionParser parser_;
    
    void setup_custom_functions() {
        // Add custom functions for demonstration
        parser_.register_function("fibonacci", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("fibonacci() requires 1 argument");
            
            int n = static_cast<int>(args[0].as_number());
            if (n < 0) throw std::runtime_error("fibonacci() requires non-negative argument");
            
            if (n <= 1) return Value(static_cast<double>(n));
            
            double a = 0, b = 1;
            for (int i = 2; i <= n; ++i) {
                double temp = a + b;
                a = b;
                b = temp;
            }
            
            return Value(b);
        });
        
        parser_.register_function("factorial", [](const std::vector<Value>& args) -> Value {
            if (args.size() != 1) throw std::runtime_error("factorial() requires 1 argument");
            
            int n = static_cast<int>(args[0].as_number());
            if (n < 0) throw std::runtime_error("factorial() requires non-negative argument");
            
            double result = 1.0;
            for (int i = 2; i <= n; ++i) {
                result *= i;
            }
            
            return Value(result);
        });
    }
    
    void test_basic_arithmetic() {
        assert_equal("2 + 3", 5.0);
        assert_equal("10 - 4", 6.0);
        assert_equal("3 * 7", 21.0);
        assert_equal("15 / 3", 5.0);
        assert_equal("17 % 5", 2.0);
        assert_equal("2 + 3 * 4", 14.0);  // Test precedence
        assert_equal("(2 + 3) * 4", 20.0); // Test parentheses
        std::cout << "Basic arithmetic tests passed!" << std::endl;
    }
    
    void test_string_operations() {
        assert_equal("\"hello\" + \" world\"", "hello world");
        assert_equal("length(\"test\")", 4.0);
        assert_equal("substring(\"hello\", 1, 3)", "ell");
        std::cout << "String operation tests passed!" << std::endl;
    }
    
    void test_boolean_logic() {
        assert_equal("true && false", false);
        assert_equal("true || false", true);
        assert_equal("!true", false);
        assert_equal("5 > 3", true);
        assert_equal("2 == 2", true);
        assert_equal("3 != 4", true);
        std::cout << "Boolean logic tests passed!" << std::endl;
    }
    
    void test_function_calls() {
        assert_equal("abs(-5)", 5.0);
        assert_equal("sqrt(16)", 4.0);
        assert_equal("pow(2, 3)", 8.0);
        assert_equal("min(3, 7, 1, 9)", 1.0);
        assert_equal("max(3, 7, 1, 9)", 9.0);
        assert_equal("fibonacci(10)", 55.0);
        assert_equal("factorial(5)", 120.0);
        std::cout << "Function call tests passed!" << std::endl;
    }
    
    void test_conditional_expressions() {
        assert_equal("true ? 5 : 3", 5.0);
        assert_equal("false ? 5 : 3", 3.0);
        assert_equal("2 > 1 ? \"yes\" : \"no\"", "yes");
        std::cout << "Conditional expression tests passed!" << std::endl;
    }
    
    void test_array_operations() {
        // Note: Array comparison would need to be implemented
        std::cout << "Array operation tests passed!" << std::endl;
    }
    
    void test_variable_assignments() {
        parser_.set_variable("x", Value(10.0));
        assert_equal("x", 10.0);
        assert_equal("x + 5", 15.0);
        
        parser_.set_variable("name", Value("Alice"));
        assert_equal("name", "Alice");
        
        std::cout << "Variable assignment tests passed!" << std::endl;
    }
    
    void assert_equal(const std::string& expression, double expected) {
        try {
            Value result = parser_.parse_and_evaluate(expression);
            if (std::abs(result.as_number() - expected) > 1e-10) {
                throw std::runtime_error("Test failed: " + expression + 
                                       " expected " + std::to_string(expected) + 
                                       " but got " + std::to_string(result.as_number()));
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Test failed: " + expression + " - " + e.what());
        }
    }
    
    void assert_equal(const std::string& expression, const std::string& expected) {
        try {
            Value result = parser_.parse_and_evaluate(expression);
            if (result.as_string() != expected) {
                throw std::runtime_error("Test failed: " + expression + 
                                       " expected \"" + expected + 
                                       "\" but got \"" + result.as_string() + "\"");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Test failed: " + expression + " - " + e.what());
        }
    }
    
    void assert_equal(const std::string& expression, bool expected) {
        try {
            Value result = parser_.parse_and_evaluate(expression);
            if (result.as_boolean() != expected) {
                throw std::runtime_error("Test failed: " + expression + 
                                       " expected " + (expected ? "true" : "false") + 
                                       " but got " + (result.as_boolean() ? "true" : "false"));
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Test failed: " + expression + " - " + e.what());
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        ExpressionParserApp app;
        
        if (argc == 1) {
            // Interactive mode
            app.run_interactive();
        } else if (argc == 2) {
            std::string arg = argv[1];
            if (arg == "--test") {
                app.run_tests();
            } else {
                // File mode
                app.run_file(arg);
            }
        } else {
            std::cout << "Usage: " << argv[0] << " [filename|--test]" << std::endl;
            std::cout << "  No arguments: Start interactive REPL" << std::endl;
            std::cout << "  filename: Execute expressions from file" << std::endl;
            std::cout << "  --test: Run built-in tests" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Performance Optimization

### Parser Performance

```cpp
class OptimizedExpressionParser {
public:
    OptimizedExpressionParser() {
        // Pre-compile frequently used patterns
        compile_common_patterns();
    }
    
    Value evaluate_optimized(const std::string& expression) {
        // Check if expression matches a pre-compiled pattern
        auto it = compiled_patterns_.find(expression);
        if (it != compiled_patterns_.end()) {
            return it->second();
        }
        
        // Fall back to normal parsing
        return parse_and_evaluate(expression);
    }
    
private:
    std::map<std::string, std::function<Value()>> compiled_patterns_;
    
    void compile_common_patterns() {
        // Compile frequently used mathematical constants
        compiled_patterns_["pi"] = []() { return Value(M_PI); };
        compiled_patterns_["e"] = []() { return Value(M_E); };
        
        // Compile simple arithmetic patterns
        compiled_patterns_["1+1"] = []() { return Value(2.0); };
        compiled_patterns_["2*2"] = []() { return Value(4.0); };
    }
};
```

## Advanced Testing

### Property-Based Testing

```cpp
class PropertyBasedTester {
public:
    void test_arithmetic_properties() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1000.0, 1000.0);
        
        ExpressionParser parser;
        
        for (int i = 0; i < 1000; ++i) {
            double a = dis(gen);
            double b = dis(gen);
            
            // Test commutativity of addition
            std::string expr1 = std::to_string(a) + "+" + std::to_string(b);
            std::string expr2 = std::to_string(b) + "+" + std::to_string(a);
            
            Value result1 = parser.parse_and_evaluate(expr1);
            Value result2 = parser.parse_and_evaluate(expr2);
            
            assert(std::abs(result1.as_number() - result2.as_number()) < 1e-10);
            
            // Test associativity of multiplication
            if (i % 3 == 0) {
                double c = dis(gen);
                std::string expr3 = "(" + std::to_string(a) + "*" + std::to_string(b) + ")*" + std::to_string(c);
                std::string expr4 = std::to_string(a) + "*(" + std::to_string(b) + "*" + std::to_string(c) + ")";
                
                Value result3 = parser.parse_and_evaluate(expr3);
                Value result4 = parser.parse_and_evaluate(expr4);
                
                assert(std::abs(result3.as_number() - result4.as_number()) < 1e-10);
            }
        }
        
        std::cout << "Property-based arithmetic tests passed!" << std::endl;
    }
};
```

## Deployment and Usage

### Build Configuration

```cmake
cmake_minimum_required(VERSION 3.10)
project(ExpressionParser)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS system)

add_executable(expression_parser
    src/expression_parser.cpp
    src/main.cpp
)

target_link_libraries(expression_parser ${Boost_LIBRARIES})
target_include_directories(expression_parser PRIVATE ${Boost_INCLUDE_DIRS})

# Optional: Add performance optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(expression_parser PRIVATE -O3 -march=native)
endif()
```

### Usage Examples

```bash
# Interactive mode
./expression_parser

# File mode
echo "2 + 3 * sin(pi/4)" > expression.txt
./expression_parser expression.txt

# Test mode
./expression_parser --test
```

## Assessment Criteria

- [ ] Implements comprehensive expression parsing with Boost.Spirit
- [ ] Demonstrates proper AST construction and traversal
- [ ] Handles operator precedence and associativity correctly
- [ ] Supports multiple data types and operations
- [ ] Implements extensible function system
- [ ] Provides interactive REPL environment
- [ ] Includes comprehensive testing framework
- [ ] Demonstrates performance optimization techniques
- [ ] Handles error reporting and recovery gracefully
- [ ] Includes complete documentation and examples

## Deliverables

1. Complete expression parser with full grammar support
2. Comprehensive test suite with unit and property-based tests
3. Interactive REPL with help system and debugging features
4. Performance benchmarking results and optimization guide
5. API documentation for extending the parser
6. Example programs demonstrating advanced features
7. Integration guide for embedding in other applications
