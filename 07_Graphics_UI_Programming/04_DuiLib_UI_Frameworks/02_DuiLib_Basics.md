# DuiLib Basics

## Topics
- DuiLib architecture and components
- Window creation and management
- Control hierarchy and layout system
- DuiLib development environment setup

### Example: Creating a Window (C++)
```cpp
CPaintManagerUI::SetInstance(hInstance);
CMainFrame* pFrame = new CMainFrame();
pFrame->Create(NULL, _T("DuiLib Window"), UI_WNDSTYLE_FRAME, 0L, 0, 0, 800, 600);
```
