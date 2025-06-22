# System Boot and Initialization

## Description
Linux boot process, init systems, Windows boot sequence, bootloaders, firmware interfaces, comparison.

## C Example: Print Boot Arguments (Linux)
```c
#include <linux/module.h>
#include <linux/init.h>

static int __init boot_init(void) {
    printk(KERN_INFO "Booting Linux kernel module\n");
    return 0;
}

static void __exit boot_exit(void) {
    printk(KERN_INFO "Exiting boot module\n");
}

module_init(boot_init);
module_exit(boot_exit);
MODULE_LICENSE("GPL");
```
