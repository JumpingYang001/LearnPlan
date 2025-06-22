# ROS/ROS2 Communication

## Overview
Explore publisher-subscriber, service, and action communication patterns in ROS/ROS2. Learn about message types, QoS (ROS2), and best practices for robust communication.

### Example: Publisher and Subscriber (C++)
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node {
public:
    MinimalPublisher() : Node("minimal_publisher") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            [this]() {
                auto msg = std_msgs::msg::String();
                msg.data = "Hello, ROS2!";
                publisher_->publish(msg);
            });
    }
private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalPublisher>());
    rclcpp::shutdown();
    return 0;
}
```
