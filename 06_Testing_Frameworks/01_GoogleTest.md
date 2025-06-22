# GoogleTest (GTest)

## Overview
GoogleTest is a C++ testing framework developed by Google. It provides a rich set of assertions, test fixtures, and test runners that make it easy to write and run tests. GoogleTest helps in creating repeatable tests for C++ code that can be integrated into continuous integration systems.

## Learning Path

### 1. GoogleTest Basics (1 week)
[See details in 01_Basics.md](01_GoogleTest/01_Basics.md)
- Understand the basic structure of GoogleTest
- Learn to write simple assertions (ASSERT_* and EXPECT_*)
- Understand the difference between fatal and non-fatal assertions
- Create and run basic test cases

### 2. Test Fixtures (1 week)
[See details in 02_Test_Fixtures.md](01_GoogleTest/02_Test_Fixtures.md)
- Understand test fixtures and how they help in code reuse
- Learn to use SetUp() and TearDown() methods
- Create test suites with shared resources
- Implement parameterized test fixtures

### 3. Advanced Assertions and Matchers (1 week)
[See details in 03_Advanced_Assertions_and_Matchers.md](01_GoogleTest/03_Advanced_Assertions_and_Matchers.md)
- Master advanced assertions
- Learn to use GoogleMock matchers with GoogleTest
- Create custom matchers for complex validations
- Use string and container-specific assertions

### 4. Parameterized Testing (1 week)
[See details in 04_Parameterized_Testing.md](01_GoogleTest/04_Parameterized_Testing.md)
- Implement tests with different parameters
- Use value-parameterized tests
- Implement type-parameterized tests
- Create test generators for complex test cases

### 5. Death Tests and Exception Testing (1 week)
[See details in 05_Death_Tests_and_Exception_Testing.md](01_GoogleTest/05_Death_Tests_and_Exception_Testing.md)
- Learn to test functions that are expected to crash
- Implement death tests with different styles
- Test for specific exception types
- Verify exception messages

### 6. Integration with Build Systems (1 week)
[See details in 06_Integration_with_Build_Systems.md](01_GoogleTest/06_Integration_with_Build_Systems.md)
- Integrate GoogleTest with CMake
- Set up test discovery and execution
- Configure test output formats
- Implement test filtering and sharding

## Projects

1. **Unit Testing Library**
   [See project details in project_01_Unit_Testing_Library.md](01_GoogleTest/project_01_Unit_Testing_Library.md)
   - Create a small utility library with comprehensive GoogleTest coverage
   - Implement fixtures, parameterized tests, and custom matchers
   - Generate test reports and analyze coverage

2. **Legacy Code Testing**
   [See project details in project_02_Legacy_Code_Testing.md](01_GoogleTest/project_02_Legacy_Code_Testing.md)
   - Take an existing codebase with minimal testing
   - Implement a test suite with GoogleTest
   - Refactor as needed to improve testability

3. **Test-Driven Development Project**
   [See details in Project_03_TDD_Project.md](01_GoogleTest/Project_03_TDD_Project.md)
   - Build a data structure or algorithm implementation using TDD
   - Write tests first, then implement the functionality
   - Achieve high test coverage and document the process

4. **Mocking Framework Integration**
   [See project details in project_04_Mocking_Framework_Integration.md](01_GoogleTest/project_04_Mocking_Framework_Integration.md)
   - Build a system with external dependencies
   - Use GoogleMock to isolate components for testing
   - Create a comprehensive test suite with mocks and stubs

5. **Continuous Integration Pipeline**
   [See project details in project_05_Continuous_Integration_Pipeline.md](01_GoogleTest/project_05_Continuous_Integration_Pipeline.md)
   - Set up a CI pipeline that runs GoogleTest tests
   - Configure test result visualization
   - Implement automatic test execution on code changes

## Resources

### Books
- "The Way of the Web Tester" by Jonathan Rasmusson
- "Professional C++" by Marc Gregoire (Sections on Testing)
- "C++ Unit Testing with GoogleTest" by Various Authors

### Online Resources
- [GoogleTest GitHub Repository](https://github.com/google/googletest)
- [GoogleTest User's Guide](https://google.github.io/googletest/)
- [GoogleTest Primer](https://google.github.io/googletest/primer.html)
- [GoogleTest Advanced Guide](https://google.github.io/googletest/advanced.html)

### Video Courses
- "Modern C++ Testing with GoogleTest" on Pluralsight
- "Test-Driven Development in C++" on Udemy
- "Mastering GoogleTest" on LinkedIn Learning

## Assessment Criteria

### Beginner Level
- Can write basic test cases with simple assertions
- Understands how to compile and run tests
- Can interpret test results and fix failing tests

### Intermediate Level
- Implements test fixtures effectively
- Uses parameterized tests for comprehensive coverage
- Integrates GoogleTest with build systems
- Can measure and report on test coverage

### Advanced Level
- Creates custom matchers and complex assertions
- Implements effective test strategies for complex systems
- Builds comprehensive test suites with high coverage
- Integrates testing with continuous integration pipelines

## Next Steps
- Explore GoogleMock for more advanced testing capabilities
- Learn about Behavior-Driven Development with GoogleTest
- Study test coverage tools like gcov and lcov
- Investigate property-based testing approaches
