# Building and Booting the Kernel

## Overview
This section explains kernel configuration, compilation, and the boot process.

### Kernel Configuration
- menuconfig, xconfig, gconfig tools
- Kconfig system

#### Example: Using menuconfig (Shell)
```sh
make menuconfig
```

### Kernel Compilation
- Building from source
- Cross-compilation

#### Example: Build Kernel (Shell)
```sh
make -j$(nproc)
```

### Boot Process
- Bootloaders (GRUB, LILO)
- initramfs/initrd

---
