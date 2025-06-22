# Memory Leak Diagnosis

## Overview
How to interpret leak reports, visualize memory, and analyze root causes.

### Leak Report Example
```
==1234== LEAK SUMMARY:
==1234==    definitely lost: 40 bytes in 1 blocks
==1234==    indirectly lost: 0 bytes in 0 blocks
```

### Visualization
Use tools like massif-visualizer or heaptrack for heap graphs.
