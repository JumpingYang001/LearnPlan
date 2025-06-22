# Message Definition

## Message Structure and Fields
A Protocol Buffer message is defined in a .proto file. Each field has a unique number and type.

## Example .proto
```proto
syntax = "proto3";

message Address {
  string street = 1;
  string city = 2;
  int32 zip = 3;
}
```

## C++ Example: Using Message
```cpp
#include "address.pb.h"

Address addr;
addr.set_street("123 Main St");
addr.set_city("Metropolis");
addr.set_zip(12345);
```
