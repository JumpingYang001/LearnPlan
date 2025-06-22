# Data Types and Fields

## Scalar Types, Repeated, Maps, Oneof
Protocol Buffers support various scalar types, repeated fields, maps, and oneof for mutually exclusive fields.

## Example .proto
```proto
syntax = "proto3";

message Example {
  int32 id = 1;
  repeated string tags = 2;
  map<string, int32> scores = 3;
  oneof value {
    string name = 4;
    int32 number = 5;
  }
}
```

## C++ Example
```cpp
#include "example.pb.h"

Example ex;
ex.set_id(1);
ex.add_tags("tag1");
(*ex.mutable_scores())["math"] = 95;
ex.set_name("Alice");
```
