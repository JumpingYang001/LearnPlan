# Project 1: Simple Kernel Module

## Description
Develop a "Hello World" kernel module. Implement module parameters and a procfs interface.

### Example: Hello World Kernel Module (C)
```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>

#define PROC_NAME "hello_proc"
static char proc_data[100] = "Hello from kernel!\n";

static ssize_t proc_read(struct file *file, char __user *buf, size_t count, loff_t *ppos) {
    return simple_read_from_buffer(buf, count, ppos, proc_data, strlen(proc_data));
}
static const struct proc_ops proc_fops = {
    .proc_read = proc_read,
};
static int __init hello_init(void) {
    proc_create(PROC_NAME, 0, NULL, &proc_fops);
    printk(KERN_INFO "Hello, Kernel Module Loaded!\n");
    return 0;
}
static void __exit hello_exit(void) {
    remove_proc_entry(PROC_NAME, NULL);
    printk(KERN_INFO "Goodbye, Kernel Module Unloaded!\n");
}
module_init(hello_init);
module_exit(hello_exit);
MODULE_LICENSE("GPL");
```
