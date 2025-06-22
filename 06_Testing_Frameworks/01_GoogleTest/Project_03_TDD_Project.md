# Project 3: Test-Driven Development Project

## Description
Build a data structure or algorithm implementation using TDD. Write tests first, then implement the functionality. Achieve high test coverage and document the process.

## Example Code
```cpp
// stack.h
#pragma once
#include <vector>
template<typename T>
class Stack {
    std::vector<T> data;
public:
    void push(const T& value) { data.push_back(value); }
    void pop() { if (!data.empty()) data.pop_back(); }
    T top() const { return data.back(); }
    bool empty() const { return data.empty(); }
};

// test_stack.cpp
#include <gtest/gtest.h>
#include "stack.h"

TEST(StackTest, PushPop) {
    Stack<int> s;
    EXPECT_TRUE(s.empty());
    s.push(1);
    EXPECT_EQ(s.top(), 1);
    s.pop();
    EXPECT_TRUE(s.empty());
}
```
