# Transforms and Geometry

## Overview
Work with tf/tf2 libraries for coordinate transforms, frame hierarchies, and geometric computations in ROS/ROS2.

### Example: Broadcasting a Transform (Python)
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class FrameBroadcaster(Node):
    def __init__(self):
        super().__init__('frame_broadcaster')
        self.broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(1.0, self.broadcast)

    def broadcast(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'robot'
        t.transform.translation.x = 1.0
        t.transform.translation.y = 2.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)

rclpy.init()
node = FrameBroadcaster()
rclpy.spin(node)
rclpy.shutdown()
```
