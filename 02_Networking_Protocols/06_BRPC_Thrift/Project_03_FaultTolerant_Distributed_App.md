# Project: Fault-tolerant Distributed Application

## Description
Build a distributed system with automatic failover, timeout/retry, and circuit breaker patterns. Test resilience under failure conditions.

## C++ Example: BRPC with Retry and Circuit Breaker
```cpp
brpc::Channel channel;
brpc::ChannelOptions options;
options.max_retry = 3;
options.timeout_ms = 1000;
channel.Init("server_list", &options);
// ...existing code for making RPC calls...
```

## Testing Resilience
- Simulate server failures and observe client behavior.
