# File Systems in Linux

## Description
VFS layer, inode structures, dentry cache, ext4/XFS/Btrfs, I/O scheduling, block layer.

## C Example: stat Usage
```c
#include <stdio.h>
#include <sys/stat.h>

int main() {
    struct stat st;
    if (stat("test.txt", &st) == 0) {
        printf("File size: %ld\n", st.st_size);
    }
    return 0;
}
```
