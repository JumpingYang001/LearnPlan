# GDI/GDI+ Performance Optimization

## Topics
- Double buffering techniques
- Resource management
- Rendering optimization strategies
- Optimized GDI/GDI+ applications

### Example: Double Buffering (C++)
```cpp
HDC hdcMem = CreateCompatibleDC(hdc);
HBITMAP hbmMem = CreateCompatibleBitmap(hdc, width, height);
SelectObject(hdcMem, hbmMem);
// Draw to hdcMem
BitBlt(hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY);
DeleteObject(hbmMem);
DeleteDC(hdcMem);
```
