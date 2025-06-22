# ROS Programming in C++

## Overview
Develop ROS/ROS2 nodes using C++. Learn about node handles, publishers, subscribers, services, actions, and advanced features like executors and time handling.

### Example: Simple ROS2 C++ Node
```cpp
#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node {
public:
    MyNode() : Node("my_node") {
        RCLCPP_INFO(this->get_logger(), "Hello from C++ node!");
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
    rclcpp::shutdown();
    return 0;
}
```
