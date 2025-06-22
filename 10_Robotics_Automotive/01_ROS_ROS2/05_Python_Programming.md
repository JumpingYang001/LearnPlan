# ROS Programming in Python

## Overview
Develop ROS/ROS2 nodes using Python. Learn about node initialization, publishers, subscribers, services, actions, and Python-specific features.

### Example: Simple ROS2 Python Node
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Hello from Python node!')

rclpy.init()
node = MyNode()
rclpy.spin(node)
rclpy.shutdown()
```
