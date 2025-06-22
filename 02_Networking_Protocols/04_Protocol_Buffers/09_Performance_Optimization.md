# Performance Optimization

## Serialization, Memory, Message Size
Optimize serialization/deserialization, memory usage, and message size.

## C++ Example: Reuse Messages
```cpp
// Reuse message objects to reduce allocations
for (int i = 0; i < 1000; ++i) {
  msg.Clear();
  msg.set_id(i);
  // ...
}
```
