# Perception in ROS

## Overview
Integrate sensors, process point clouds, and use computer vision in ROS/ROS2. Learn about sensor fusion and state estimation.

### Example: Subscribing to a Camera Topic (Python)
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)

    def listener_callback(self, msg):
        self.get_logger().info('Received image frame')

rclpy.init()
node = ImageListener()
rclpy.spin(node)
rclpy.shutdown()
```
