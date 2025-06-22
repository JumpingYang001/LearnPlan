# Kernel Module Programming

## Overview
Covers module basics, development, debugging, and best practices.

### Module Basics
- Module vs. built-in
- Loading/unloading

#### Example: Hello World Kernel Module (C)
```c
#include <linux/module.h>
#include <linux/kernel.h>

static int __init hello_init(void) {
    printk(KERN_INFO "Hello, Kernel!\n");
    return 0;
}
static void __exit hello_exit(void) {
    printk(KERN_INFO "Goodbye, Kernel!\n");
}
module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```

### Module Debugging
- printk, ftrace, kgdb

---
