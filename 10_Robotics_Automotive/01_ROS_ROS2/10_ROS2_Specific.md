# ROS 2 Specific Features

## Overview
Explore DDS middleware, security, real-time capabilities, and lifecycle nodes in ROS2.

### Example: Lifecycle Node (Python)
```python
import rclpy
from rclpy.lifecycle import LifecycleNode

class MyLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('my_lifecycle_node')

rclpy.init()
node = MyLifecycleNode()
rclpy.spin(node)
rclpy.shutdown()
```
