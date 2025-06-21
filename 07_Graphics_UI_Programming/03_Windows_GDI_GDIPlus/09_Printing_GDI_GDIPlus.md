# Printing with GDI/GDI+

## Topics
- Printer device contexts
- Print job management
- Print preview implementation
- Print functionality in applications

### Example: Printing a Page (C++)
```cpp
// In WM_PRINT or print handler
DOCINFO di = { sizeof(DOCINFO), L"My Document" };
StartDoc(hdc, &di);
StartPage(hdc);
TextOut(hdc, 100, 100, L"Printing with GDI!", 19);
EndPage(hdc);
EndDoc(hdc);
```
