# Gradle Build System

## Overview
This section covers Gradle's concepts, plugins, tasks, and incremental builds. Compare Gradle with Bazel and see sample projects.

## Example: Gradle C++ Application
```groovy
// build.gradle
apply plugin: 'cpp-application'
application {
    targetMachines = [machines.windows.x86_64]
}
```
