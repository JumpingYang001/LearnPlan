# Project: ROS Web Interface

## Description
Create a web dashboard for a robot, implement teleoperation via web, add sensor data visualization, and develop a user-friendly control interface.

### Example: rosbridge WebSocket Launch
```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

### Example: Simple Web Teleop (JavaScript)
```javascript
// Using roslibjs
var ros = new ROSLIB.Ros({url : 'ws://localhost:9090'});
var cmdVel = new ROSLIB.Topic({
  ros : ros,
  name : '/cmd_vel',
  messageType : 'geometry_msgs/Twist'
});
function sendCommand(linear, angular) {
  var twist = new ROSLIB.Message({
    linear : { x: linear, y: 0, z: 0 },
    angular : { x: 0, y: 0, z: angular }
  });
  cmdVel.publish(twist);
}
```
