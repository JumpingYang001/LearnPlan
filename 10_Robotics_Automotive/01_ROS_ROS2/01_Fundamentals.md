# ROS/ROS2 Fundamentals

## Overview
Learn the core architecture and concepts of ROS and ROS2, including nodes, topics, services, actions, parameters, and workspaces. Understand the differences between ROS 1 and ROS 2, and how to set up your development environment.

### Example: Minimal ROS2 Node (Python)
```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Node has started!')

rclpy.init()
node = MinimalNode()
rclpy.spin(node)
rclpy.shutdown()
```
