# Protocol Buffers with gRPC

## Defining gRPC Services
Use Protocol Buffers to define gRPC services and RPC methods.

## Example .proto
```proto
syntax = "proto3";

service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply);
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}
```

## C++ Example: gRPC Service Stub
```cpp
// Generated code and gRPC C++ API usage
// ... see gRPC C++ documentation for full example ...
```
