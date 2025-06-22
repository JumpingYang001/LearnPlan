# Windows Registry Programming

## Description
This section covers registry structure, keys, values, and virtualization. Below is a C example for creating and reading a registry key.

## Example: Create and Read Registry Key (C)

```c
#include <windows.h>
#include <stdio.h>

int main() {
    HKEY hKey;
    if (RegCreateKeyEx(HKEY_CURRENT_USER, "Software\\MyApp", 0, NULL, 0, KEY_WRITE, NULL, &hKey, NULL) == ERROR_SUCCESS) {
        const char* data = "TestValue";
        RegSetValueEx(hKey, "MyValue", 0, REG_SZ, (const BYTE*)data, (DWORD)(strlen(data) + 1));
        RegCloseKey(hKey);
    }
    if (RegOpenKeyEx(HKEY_CURRENT_USER, "Software\\MyApp", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD bufSize = sizeof(buffer);
        if (RegQueryValueEx(hKey, "MyValue", 0, NULL, (LPBYTE)buffer, &bufSize) == ERROR_SUCCESS) {
            printf("Registry Value: %s\n", buffer);
        }
        RegCloseKey(hKey);
    }
    return 0;
}
```
