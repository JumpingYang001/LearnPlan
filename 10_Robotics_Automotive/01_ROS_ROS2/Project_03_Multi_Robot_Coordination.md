# Project: Multi-robot Coordination

## Description
Design a multi-robot system, implement inter-robot communication, create coordination algorithms, and demonstrate collaborative tasks.

### Example: Multi-robot Launch (Python)
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_bringup',
            executable='robot1_node',
            namespace='robot1',
            output='screen'),
        Node(
            package='my_robot_bringup',
            executable='robot2_node',
            namespace='robot2',
            output='screen'),
    ])
```

### Example: Inter-robot Communication (Python)
```python
# Use ROS2 topics/services/actions with namespaces for communication
# Example: robot1 publishes to /robot1/status, robot2 subscribes
```
