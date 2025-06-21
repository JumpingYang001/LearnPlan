# DuiLib Controls

## Topics
- Built-in controls and their properties
- Control events and handling
- Control customization and styling
- Applications with various controls

### Example: Button Event Handler (C++)
```cpp
void CMainFrame::Notify(TNotifyUI& msg) {
    if (msg.sType == _T("click") && msg.pSender->GetName() == _T("btn_ok")) {
        // Handle OK button click
    }
}
```
