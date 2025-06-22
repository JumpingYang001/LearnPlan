# Profiling Tools on Linux

Overview of popular Linux profiling tools: perf, Valgrind, Perfetto, and BPF/eBPF tools.

## perf Example
```sh
# Compile with debug info
gcc -g -o myprog myprog.c
# Record performance data
perf record ./myprog
# Show report
perf report
```

## Valgrind Callgrind Example
```sh
valgrind --tool=callgrind ./myprog
callgrind_annotate callgrind.out.*
```

## Perfetto Trace Example
```sh
perfetto -o trace_file.perfetto-trace -t 10s
```

## BPF/eBPF Example (bpftrace)
```sh
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'
```
