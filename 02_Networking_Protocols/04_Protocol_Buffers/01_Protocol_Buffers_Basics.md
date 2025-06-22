# Protocol Buffers Basics

## Overview
Protocol Buffers (protobuf) is a language-neutral, platform-neutral, extensible mechanism for serializing structured data. It is more efficient than XML or JSON, offering smaller size, faster parsing, and type safety.

## .proto File Example
```proto
syntax = "proto3";

message Person {
  string name = 1;
  int32 id = 2;
  string email = 3;
}
```

## C++ Example: Serialize and Parse
```cpp
#include "person.pb.h"

Person person;
person.set_name("Alice");
person.set_id(123);
person.set_email("alice@example.com");

std::string output;
person.SerializeToString(&output);

Person parsed;
parsed.ParseFromString(output);
```
