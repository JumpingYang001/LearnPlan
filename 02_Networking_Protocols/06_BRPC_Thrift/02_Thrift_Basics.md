# Apache Thrift Basics

## Overview
Apache Thrift is an RPC framework that uses an Interface Definition Language (IDL) to define data types and service interfaces, supporting code generation for multiple languages.

## Key Concepts
- Thrift IDL: Used to define services and data structures.
- Type System: Supports various primitive and complex types.
- Code Generation: Generates client/server code for C++, Java, Python, etc.

## C++ Example: Thrift IDL and Service
```thrift
// calculator.thrift
service Calculator {
    i32 add(1:i32 num1, 2:i32 num2),
}
```

```cpp
// C++ server stub (generated)
#include "Calculator.h"
// ...existing code...
```
