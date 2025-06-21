# Basic Drawing with GDI

## Topics
- Line and shape drawing
- Text rendering and formatting
- Color management and palettes
- Applications with basic GDI graphics

### Example: Drawing a Rectangle and Text (C)
```c
HDC hdc = GetDC(hwnd);
Rectangle(hdc, 20, 20, 120, 80);
TextOut(hdc, 30, 40, L"Hello, GDI!", 11);
ReleaseDC(hwnd, hdc);
```
