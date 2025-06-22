# Project: Data Serialization Library

## Description
Build a library that provides a common interface for multiple serialization formats. Implement Protocol Buffers as the primary format. Create benchmarks comparing with JSON and XML.

## Example C++ Code: Serialization Interface
```cpp
class Serializer {
public:
    virtual std::string serialize(const MyMessage& msg) = 0;
    virtual MyMessage deserialize(const std::string& data) = 0;
};

class ProtobufSerializer : public Serializer {
public:
    std::string serialize(const MyMessage& msg) override {
        std::string out;
        msg.SerializeToString(&out);
        return out;
    }
    MyMessage deserialize(const std::string& data) override {
        MyMessage msg;
        msg.ParseFromString(data);
        return msg;
    }
};
```
