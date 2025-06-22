# Bazel for Java and Kotlin

## Overview
This section explains Bazel's support for Java and Kotlin, including JVM toolchain configuration, JAR creation, and dependency management.

## Example: Java Binary with Bazel
```python
# BUILD file
java_binary(
    name = "HelloJava",
    srcs = ["HelloJava.java"],
    main_class = "HelloJava",
)
```
```java
// HelloJava.java
public class HelloJava {
    public static void main(String[] args) {
        System.out.println("Hello from Bazel Java!");
    }
}
```
