# Project: Custom File System Implementation

## Description
Build a simple file system for both Linux and Windows. Implement core file system operations and management tools.

## Linux Example: FUSE Skeleton
```c
#define FUSE_USE_VERSION 31
#include <fuse3/fuse.h>
static int do_getattr(const char *path, struct stat *stbuf) {
    // Implement getattr
    return 0;
}
static struct fuse_operations ops = {
    .getattr = do_getattr,
};
int main(int argc, char *argv[]) {
    return fuse_main(argc, argv, &ops, NULL);
}
```

## Windows Example: File Operations
```cpp
#include <windows.h>
#include <iostream>
int main() {
    HANDLE hFile = CreateFile("testfs.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hFile, "FS Data", 7, &written, NULL);
        CloseHandle(hFile);
    }
    return 0;
}
```
