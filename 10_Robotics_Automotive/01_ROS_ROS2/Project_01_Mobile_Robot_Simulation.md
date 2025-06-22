# Project: Mobile Robot Simulation

## Description
Create a simulated differential drive robot, implement navigation, add sensor processing, and develop autonomous behaviors using ROS2 and Gazebo.

### Example: Differential Drive Robot URDF
```xml
<robot name="diff_drive_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </visual>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
  </joint>
  <link name="left_wheel"/>
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
  </joint>
  <link name="right_wheel"/>
</robot>
```

### Example: Navigation Launch (Python)
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_bringup',
            executable='navigation_launch.py',
            output='screen'),
    ])
```
