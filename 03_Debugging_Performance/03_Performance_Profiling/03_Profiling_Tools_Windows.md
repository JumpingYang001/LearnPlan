# Profiling Tools on Windows

Covers WPA, Visual Studio Profiler, WinDbg, and Windows Performance Recorder.

## Windows Performance Analyzer (WPA) Example
```bat
REM Record a trace
wpr -start generalprofile -filemode
REM Stop and save trace
wpr -stop trace.etl
REM Open in WPA for analysis
```

## Visual Studio Profiler Example
- Open your project in Visual Studio
- Go to Debug > Performance Profiler
- Select CPU Usage, Memory Usage, or GPU Usage
- Start profiling and analyze the results

## WinDbg Example
```bat
REM Launch WinDbg and attach to process
REM Use !analyze -v for analysis
```
