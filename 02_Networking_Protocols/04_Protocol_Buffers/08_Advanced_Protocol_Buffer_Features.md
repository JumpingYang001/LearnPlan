# Advanced Protocol Buffer Features

## Extensions, Reserved Fields, Compatibility
Use extensions, reserved fields, and unknown field handling for schema evolution.

## Example .proto
```proto
syntax = "proto3";

message Versioned {
  int32 id = 1;
  reserved 2, 3;
  string name = 4;
}
```

## C++ Example: Unknown Fields
```cpp
// C++ API provides UnknownFieldSet for advanced use
// ... see protobuf C++ docs for details ...
```
