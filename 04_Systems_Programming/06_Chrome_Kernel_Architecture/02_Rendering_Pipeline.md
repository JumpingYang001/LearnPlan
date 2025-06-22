# Rendering Pipeline

## Overview
The rendering pipeline transforms HTML, CSS, and JavaScript into pixels on the screen. Chrome's pipeline includes DOM construction, style calculation, layout, painting, and compositing.

## Key Concepts
- DOM, CSSOM, render tree
- Layout, painting, compositing
- Performance optimizations

## Example: DOM Tree Representation in C++
```cpp
#include <iostream>
#include <vector>
#include <string>

struct Node {
    std::string tag;
    std::vector<Node*> children;
    Node(const std::string& t) : tag(t) {}
};

void printDOM(Node* node, int depth = 0) {
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << node->tag << std::endl;
    for (auto child : node->children) printDOM(child, depth + 1);
}

int main() {
    Node html("html");
    Node body("body");
    Node div("div");
    html.children.push_back(&body);
    body.children.push_back(&div);
    printDOM(&html);
    return 0;
}
```

This code models a simple DOM tree, similar to Chrome's internal representation.
