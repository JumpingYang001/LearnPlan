# Project 2: Character Device Driver

## Description
Create a character device with read/write operations and implement ioctl commands.

### Example: Simple Char Device Driver (C)
```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#define DEVICE_NAME "mychardev"
static int major;
static char msg[100] = "";
static ssize_t dev_read(struct file *f, char __user *buf, size_t len, loff_t *off) {
    return simple_read_from_buffer(buf, len, off, msg, strlen(msg));
}
static ssize_t dev_write(struct file *f, const char __user *buf, size_t len, loff_t *off) {
    return simple_write_to_buffer(msg, sizeof(msg), off, buf, len);
}
static long dev_ioctl(struct file *f, unsigned int cmd, unsigned long arg) {
    // Example: handle ioctl commands
    return 0;
}
static struct file_operations fops = {
    .read = dev_read,
    .write = dev_write,
    .unlocked_ioctl = dev_ioctl,
};
static int __init chardev_init(void) {
    major = register_chrdev(0, DEVICE_NAME, &fops);
    return 0;
}
static void __exit chardev_exit(void) {
    unregister_chrdev(major, DEVICE_NAME);
}
module_init(chardev_init);
module_exit(chardev_exit);
MODULE_LICENSE("GPL");
```
