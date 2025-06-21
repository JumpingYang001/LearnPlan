# GDI+ Drawing Basics

## Topics
- Graphics class and its methods
- Pens, brushes, and paths
- Transformations and coordinate systems
- Basic GDI+ drawing applications

### Example: Drawing with GDI+ (C++)
```cpp
Graphics graphics(hdc);
Pen pen(Color(255, 0, 0, 255));
graphics.DrawLine(&pen, 10, 10, 200, 100);
```
