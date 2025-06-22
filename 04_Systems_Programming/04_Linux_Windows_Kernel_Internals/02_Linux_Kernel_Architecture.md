# Linux Kernel Architecture

## Description
Linux kernel subsystems, monolithic vs. modular design, kernel/user space, and development model.

## C Example: Print from Kernel Module
```c
#include <linux/module.h>
#include <linux/kernel.h>

int init_module(void) {
    printk(KERN_INFO "Hello, Linux kernel!\n");
    return 0;
}

void cleanup_module(void) {
    printk(KERN_INFO "Goodbye, Linux kernel!\n");
}
```
