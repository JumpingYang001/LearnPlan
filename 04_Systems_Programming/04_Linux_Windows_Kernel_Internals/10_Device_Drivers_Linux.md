# Device Drivers in Linux

## Description
Linux driver model, character/block/network drivers, kernel modules, loading mechanisms.

## C Example: Simple Character Device
```c
#include <linux/module.h>
#include <linux/fs.h>

static int major;

static int my_open(struct inode *inode, struct file *file) { return 0; }
static int my_release(struct inode *inode, struct file *file) { return 0; }

static struct file_operations fops = {
    .open = my_open,
    .release = my_release,
};

static int __init my_init(void) {
    major = register_chrdev(0, "mychardev", &fops);
    return 0;
}

static void __exit my_exit(void) {
    unregister_chrdev(major, "mychardev");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
```
