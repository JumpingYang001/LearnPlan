# GDI+ Introduction

## Topics
- GDI+ architecture and classes
- Differences between GDI and GDI+
- GDI+ namespaces and object model
- Setting up GDI+ in applications

### Example: Initializing GDI+ (C++)
```cpp
#include <gdiplus.h>
using namespace Gdiplus;

ULONG_PTR gdiplusToken;
GdiplusStartupInput gdiplusStartupInput;
GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
// ...
GdiplusShutdown(gdiplusToken);
```
