# Protocol Buffers in C++

## Compiling and Using in C++
Use protoc to generate C++ code from .proto files. Manipulate messages using generated classes.

## Compile Command
```sh
protoc --cpp_out=. person.proto
```

## C++ Example
```cpp
#include "person.pb.h"

Person p;
p.set_name("Bob");
p.set_id(456);

std::string data;
p.SerializeToString(&data);

Person p2;
p2.ParseFromString(data);
```
