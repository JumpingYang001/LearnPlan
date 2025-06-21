# GDI Fundamentals

## Key Concepts
- Device Contexts (DC)
- Coordinate systems and mapping modes
- GDI objects (pens, brushes, fonts, bitmaps)
- Basic GDI drawing operations

### Example: Creating a Device Context and Drawing a Line (C)
```c
HDC hdc = GetDC(hwnd);
MoveToEx(hdc, 10, 10, NULL);
LineTo(hdc, 100, 100);
ReleaseDC(hwnd, hdc);
```
