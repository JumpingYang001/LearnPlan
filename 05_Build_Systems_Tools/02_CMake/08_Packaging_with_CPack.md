# Packaging with CPack

## CPack Basics
- include(CPack)
- Package generators

### Example: Minimal CPack
```cmake
include(CPack)
set(CPACK_GENERATOR "ZIP")
```
