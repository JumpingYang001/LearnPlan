# Media Handling

## Overview
Learn about audio/video capture, codecs, compression, and adaptive streaming in WebRTC.

### C/C++ Example: Capturing Video Frame (Pseudocode)
```cpp
// Pseudocode for capturing a video frame using OpenCV (commonly used with C++)
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0); // Open default camera
    if (!cap.isOpened()) return -1;
    cv::Mat frame;
    cap >> frame; // Capture a frame
    cv::imwrite("frame.jpg", frame); // Save frame
    return 0;
}
```
