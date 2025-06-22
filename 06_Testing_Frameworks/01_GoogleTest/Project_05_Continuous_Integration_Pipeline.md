# Project 5: Continuous Integration Pipeline

## Description
Set up a CI pipeline that runs GoogleTest tests. Configure test result visualization and implement automatic test execution on code changes.

## Example Code
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: sudo apt-get install -y cmake g++ libgtest-dev
      - name: Build
        run: |
          cmake .
          make
      - name: Run tests
        run: ctest --output-on-failure
```
