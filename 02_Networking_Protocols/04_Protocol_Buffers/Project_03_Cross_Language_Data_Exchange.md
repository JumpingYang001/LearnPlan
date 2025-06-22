# Project: Cross-Language Data Exchange System

## Description
Build a system where multiple components in different languages exchange data. Implement Protocol Buffers as the common data format. Create adapters for each language.

## Example .proto
```proto
syntax = "proto3";

message DataPacket {
  int32 id = 1;
  string payload = 2;
}
```

## Example C++ Adapter
```cpp
#include "data_packet.pb.h"

DataPacket packet;
packet.set_id(1);
packet.set_payload("Hello");

std::string out;
packet.SerializeToString(&out);
// Send 'out' to another language component
```
