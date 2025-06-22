# CMake Build System

*Last Updated: May 25, 2025*

## Overview

CMake is a cross-platform build system generator that simplifies the process of building, testing, and packaging software across different platforms and compilers. This learning track covers CMake from basic usage to advanced techniques for managing complex C/C++ projects.

## Learning Path

### 1. CMake Fundamentals (1 week)
[See details in 01_CMake_Fundamentals.md](02_CMake/01_CMake_Fundamentals.md)
- **Build System Concepts**
  - Build system vs. build system generator
  - Build configurations (Debug, Release)
  - Cross-platform considerations
  - Dependency management
- **CMake Architecture**
  - CMake language
  - Generator concept
  - Build process stages
  - CMakeCache and CMakeFiles
- **Basic CMake Usage**
  - Creating a CMakeLists.txt file
  - Configuring and generating build files
  - Building with generated build system
  - Command-line interface
- **CMake GUI and ccmake**
  - Visual configuration
  - Cache variable manipulation
  - Project settings
  - Generator selection

### 2. CMake Language Basics (1 week)
[See details in 02_CMake_Language_Basics.md](02_CMake/02_CMake_Language_Basics.md)
- **Variables and Properties**
  - Setting and accessing variables
  - Cache variables
  - Environment variables
  - Target properties
  - Global, directory, and source file properties
- **Control Structures**
  - if/else/endif
  - foreach/endforeach
  - while/endwhile
  - Function and macro behavior
- **Lists and Strings**
  - List operations
  - String manipulation
  - Regular expressions
  - File path handling
- **Comments and Documentation**
  - Comment syntax
  - Documentation best practices
  - Documenting CMakeLists.txt
  - Documenting custom modules

### 3. Building C/C++ Projects (2 weeks)
[See details in 03_Building_CC_Projects.md](02_CMake/03_Building_CC_Projects.md)
- **Project Structure**
  - Single vs. multi-directory projects
  - Subprojects and subdirectories
  - External projects
  - Component organization
- **Targets and Libraries**
  - add_executable
  - add_library (STATIC, SHARED, MODULE)
  - Object libraries
  - Interface libraries
- **Dependencies and Linking**
  - target_link_libraries
  - Public, private, and interface dependencies
  - Link order
  - Circular dependencies
- **Include Directories**
  - target_include_directories
  - SYSTEM includes
  - PUBLIC vs. PRIVATE includes
  - Interface includes
- **Compile Definitions and Options**
  - target_compile_definitions
  - target_compile_options
  - target_compile_features
  - Compiler-specific options

### 4. Modern CMake Best Practices (1 week)
[See details in 04_Modern_CMake_Best_Practices.md](02_CMake/04_Modern_CMake_Best_Practices.md)
- **Target-Based Design**
  - Targets as self-contained units
  - Target properties vs. global settings
  - Transitive usage requirements
  - Target-based organization
- **Generator Expressions**
  - Conditional expressions
  - String transformations
  - Target property queries
  - Platform and configuration selection
- **Policies**
  - Policy handling
  - Version-specific behaviors
  - Policy scope
  - Minimum required version
- **CMake Anti-Patterns**
  - Global property pollution
  - Scope issues
  - Brittle custom commands
  - Inefficient dependency handling

### 5. Finding and Using Packages (2 weeks)
[See details in 05_Finding_and_Using_Packages.md](02_CMake/05_Finding_and_Using_Packages.md)
- **find_package Command**
  - Module mode vs. Config mode
  - VERSION and COMPONENTS
  - REQUIRED and QUIET options
  - Package locations
- **Standard Find Modules**
  - FindBoost, FindQt5, etc.
  - Module structure
  - Result variables
  - Imported targets
- **Creating Config Packages**
  - PackageConfig.cmake files
  - Version files
  - Exporting targets
  - Dependencies in packages
- **Using External Packages**
  - Imported targets
  - INTERFACE libraries
  - Packaging considerations
  - Version constraints

### 6. File System and I/O Operations (1 week)
[See details in 06_File_System_and_IO_Operations.md](02_CMake/06_File_System_and_IO_Operations.md)
- **File System Commands**
  - file(GLOB)
  - file(READ/WRITE)
  - file(DOWNLOAD)
  - file(MAKE_DIRECTORY)
- **File Generation**
  - configure_file
  - file(GENERATE)
  - Custom file generation
  - Template processing
- **Path Manipulation**
  - CMAKE_CURRENT_SOURCE_DIR
  - CMAKE_CURRENT_BINARY_DIR
  - Relative and absolute paths
  - Path normalization
- **Working with Project Files**
  - Source file properties
  - Automatic source grouping
  - IDE integration
  - Custom commands for files

### 7. Testing with CTest (1 week)
[See details in 07_Testing_with_CTest.md](02_CMake/07_Testing_with_CTest.md)
- **CTest Basics**
  - enable_testing
  - add_test
  - Test properties
  - Running tests
- **Test Organization**
  - Test labels
  - Test fixtures
  - Test dependencies
  - Resource allocation
- **Test Output and Results**
  - PASS/FAIL criteria
  - Output collection
  - Result analysis
  - Test timeouts
- **Advanced Testing Features**
  - Parallel testing
  - Distributed testing
  - Memory and leak checking
  - Coverage analysis with CTest

### 8. Packaging with CPack (1 week)
[See details in 08_Packaging_with_CPack.md](02_CMake/08_Packaging_with_CPack.md)
- **CPack Basics**
  - include(CPack)
  - Package generators
  - Package contents
  - Package metadata
- **Platform-Specific Packaging**
  - Windows (NSIS, WIX)
  - Linux (DEB, RPM)
  - macOS (DMG, PKG)
  - Cross-platform (ZIP, TGZ)
- **Component-Based Packaging**
  - COMPONENT specification
  - Component dependencies
  - Component grouping
  - Component installation
- **Deployment Considerations**
  - Runtime dependencies
  - Library paths
  - Installation directories
  - Post-install scripts

### 9. Cross-Compilation and Toolchains (1 week)
[See details in 09_Cross-Compilation_and_Toolchains.md](02_CMake/09_Cross-Compilation_and_Toolchains.md)
- **Toolchain Files**
  - Structure and components
  - Compiler selection
  - System root configuration
  - Platform specifications
- **Cross-Compilation Setup**
  - CMAKE_TOOLCHAIN_FILE
  - CMAKE_CROSSCOMPILING
  - CMAKE_SYSTEM_NAME
  - CMAKE_<LANG>_COMPILER
- **Platform-Specific Considerations**
  - Android
  - iOS
  - Windows on ARM
  - Embedded systems
- **Testing in Cross-Compilation**
  - Emulator support
  - Host tool execution
  - Cross-compiled tests
  - CTest configuration

### 10. Advanced CMake Features (2 weeks)
[See details in 10_Advanced_CMake_Features.md](02_CMake/10_Advanced_CMake_Features.md)
- **Custom Commands and Targets**
  - add_custom_command
  - add_custom_target
  - Command dependencies
  - Output generation
- **Code Generation**
  - Generating source code
  - Handling generated sources
  - Integration with build process
  - Language-specific considerations
- **IDE Integration**
  - Visual Studio integration
  - Xcode integration
  - CLion integration
  - Source groups and organization
- **Parallel Builds**
  - Dependency graph
  - Job pools
  - Resource limits
  - Optimization strategies

### 11. CMake Extensions and Modules (1 week)
[See details in 11_CMake_Extensions_and_Modules.md](02_CMake/11_CMake_Extensions_and_Modules.md)
- **Writing CMake Modules**
  - Module structure
  - Namespacing
  - Reusable functionality
  - Module documentation
- **Custom Find Modules**
  - Structure and naming
  - Finding components
  - Version checking
  - Handling dependencies
- **Functions and Macros**
  - Function vs. macro
  - Arguments and scope
  - Return values
  - Error handling
- **Script Mode and Utility Scripts**
  - cmake -P usage
  - Standalone scripts
  - Build system integration
  - Automation with CMake scripts

### 12. Modern C++ and CMake (1 week)
[See details in 12_Modern_C_and_CMake.md](02_CMake/12_Modern_C_and_CMake.md)
- **C++ Standard Selection**
  - target_compile_features
  - CMAKE_CXX_STANDARD
  - Feature detection
  - Compiler compatibility
- **Language Features and Requirements**
  - C++11/14/17/20 features
  - Feature-based requirements
  - Compiler-specific workarounds
  - Platform considerations
- **Library Integration**
  - Boost with CMake
  - Qt with CMake
  - Modern C++ libraries
  - Package management integration
- **Modern Patterns**
  - Header-only libraries
  - Template libraries
  - Compile-time polymorphism
  - Build-time configuration

## Projects

1. **Cross-Platform Library**
   [See project details in project_01_Cross-Platform_Library.md](02_CMake/project_01_Cross-Platform_Library.md)
   - Create a library with CMake build system
   - Support multiple platforms and compilers
   - Implement proper package export

2. **Multi-Component Application**
   [See project details in project_02_Multi-Component_Application.md](02_CMake/project_02_Multi-Component_Application.md)
   - Build a complex application with multiple components
   - Manage internal and external dependencies
   - Create installation and packaging

3. **Custom Build Tool Integration**
   [See project details in project_03_Custom_Build_Tool_Integration.md](02_CMake/project_03_Custom_Build_Tool_Integration.md)
   - Integrate code generation tools
   - Create custom build steps
   - Ensure proper dependency tracking

4. **Cross-Compilation Toolchain**
   [See project details in project_04_Cross-Compilation_Toolchain.md](02_CMake/project_04_Cross-Compilation_Toolchain.md)
   - Create toolchain files for different platforms
   - Test cross-compilation workflow
   - Implement platform-specific optimizations

5. **CI/CD Integration**
   [See project details in project_05_CICD_Integration.md](02_CMake/project_05_CICD_Integration.md)
   - Set up CMake project for continuous integration
   - Implement multi-platform testing
   - Create automated deployment with CPack

## Resources

### Books
- "Professional CMake: A Practical Guide" by Craig Scott
- "CMake Cookbook" by Radovan Bast and Roberto Di Remigio
- "Mastering CMake" by Ken Martin and Bill Hoffman
- "Modern CMake for C++" by Rafal Swidzinski

### Online Resources
- [CMake Official Documentation](https://cmake.org/documentation/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [The Modern CMake Book](https://cliutils.gitlab.io/modern-cmake/)
- [Effective Modern CMake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
- [CMake Examples](https://github.com/ttroy50/cmake-examples)

### Video Courses
- "CMake for Beginners" on Udemy
- "Professional CMake" on Pluralsight
- "C++ Development with CMake" courses

## Assessment Criteria

You should be able to:
- Create well-structured, maintainable CMake build systems
- Handle complex dependencies and package requirements
- Configure cross-platform builds with appropriate conditionals
- Implement testing and packaging for C/C++ projects
- Debug and troubleshoot CMake-related issues
- Follow modern CMake best practices
- Extend CMake with custom modules and functions

## Next Steps

After mastering CMake, consider exploring:
- Integrating with other build systems (Meson, Bazel)
- Advanced C++ build techniques
- Package managers (Conan, vcpkg)
- Build system generators (CMake vs. Premake vs. GN)
- DevOps integration for C++ projects
- Performance optimization of build systems
