# STL Algorithms

*Part of STL Learning Track - 2 weeks*

## Overview

STL algorithms are function templates that perform operations on sequences of elements. They work with iterators and are designed to be generic, efficient, and composable. The algorithms are divided into several categories based on their functionality.

## Non-modifying Sequence Operations

### std::find, std::find_if, std::find_if_not
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void find_examples() {
    std::vector<int> vec = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    
    // Find specific value
    auto it = std::find(vec.begin(), vec.end(), 7);
    if (it != vec.end()) {
        std::cout << "Found 7 at position: " << std::distance(vec.begin(), it) << std::endl;
    }
    
    // Find with predicate
    auto even_it = std::find_if(vec.begin(), vec.end(), [](int n) { return n % 2 == 0; });
    if (even_it != vec.end()) {
        std::cout << "First even number: " << *even_it << std::endl;
    }
    
    // Find if not matching predicate
    auto odd_it = std::find_if_not(vec.begin(), vec.end(), [](int n) { return n % 2 == 0; });
    if (odd_it != vec.end()) {
        std::cout << "First odd number: " << *odd_it << std::endl;
    }
}
```

### std::count, std::count_if
```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

void count_examples() {
    std::vector<int> numbers = {1, 2, 3, 2, 4, 2, 5, 6, 2};
    std::string text = "hello world";
    
    // Count occurrences of specific value
    int count_2 = std::count(numbers.begin(), numbers.end(), 2);
    std::cout << "Number of 2s: " << count_2 << std::endl;
    
    // Count characters in string
    int count_l = std::count(text.begin(), text.end(), 'l');
    std::cout << "Number of 'l' characters: " << count_l << std::endl;
    
    // Count with predicate
    int even_count = std::count_if(numbers.begin(), numbers.end(), 
                                   [](int n) { return n % 2 == 0; });
    std::cout << "Number of even numbers: " << even_count << std::endl;
    
    // Count vowels in string
    int vowel_count = std::count_if(text.begin(), text.end(), 
                                    [](char c) { return c == 'a' || c == 'e' || c == 'i' || 
                                                        c == 'o' || c == 'u'; });
    std::cout << "Number of vowels: " << vowel_count << std::endl;
}
```

### std::for_each
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void for_each_examples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Simple lambda
    std::cout << "Elements: ";
    std::for_each(vec.begin(), vec.end(), [](int n) { std::cout << n << " "; });
    std::cout << std::endl;
    
    // Modify elements (non-const lambda)
    std::for_each(vec.begin(), vec.end(), [](int& n) { n *= 2; });
    
    std::cout << "After doubling: ";
    std::for_each(vec.begin(), vec.end(), [](int n) { std::cout << n << " "; });
    std::cout << std::endl;
    
    // Using function object with state
    class Sum {
    public:
        int total = 0;
        void operator()(int n) { total += n; }
    };
    
    Sum sum_func = std::for_each(vec.begin(), vec.end(), Sum{});
    std::cout << "Sum of elements: " << sum_func.total << std::endl;
}
```

### std::all_of, std::any_of, std::none_of
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void condition_check_examples() {
    std::vector<int> all_even = {2, 4, 6, 8, 10};
    std::vector<int> mixed = {1, 2, 3, 4, 5};
    std::vector<int> all_odd = {1, 3, 5, 7, 9};
    
    auto is_even = [](int n) { return n % 2 == 0; };
    
    // Check if all elements satisfy condition
    std::cout << "all_even - all even: " << std::all_of(all_even.begin(), all_even.end(), is_even) << std::endl;
    std::cout << "mixed - all even: " << std::all_of(mixed.begin(), mixed.end(), is_even) << std::endl;
    
    // Check if any element satisfies condition
    std::cout << "mixed - any even: " << std::any_of(mixed.begin(), mixed.end(), is_even) << std::endl;
    std::cout << "all_odd - any even: " << std::any_of(all_odd.begin(), all_odd.end(), is_even) << std::endl;
    
    // Check if no element satisfies condition
    std::cout << "all_odd - none even: " << std::none_of(all_odd.begin(), all_odd.end(), is_even) << std::endl;
    std::cout << "mixed - none even: " << std::none_of(mixed.begin(), mixed.end(), is_even) << std::endl;
}
```

## Modifying Sequence Operations

### std::copy, std::copy_if, std::copy_n
```cpp
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>

void copy_examples() {
    std::vector<int> source = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> dest(source.size());
    
    // Basic copy
    std::copy(source.begin(), source.end(), dest.begin());
    
    std::cout << "Copied elements: ";
    for (const auto& item : dest) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Copy with condition
    std::vector<int> even_dest;
    std::copy_if(source.begin(), source.end(), std::back_inserter(even_dest),
                 [](int n) { return n % 2 == 0; });
    
    std::cout << "Even elements: ";
    for (const auto& item : even_dest) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Copy first n elements
    std::vector<int> first_five(5);
    std::copy_n(source.begin(), 5, first_five.begin());
    
    std::cout << "First 5 elements: ";
    for (const auto& item : first_five) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::move
```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

void move_examples() {
    std::vector<std::string> source = {"hello", "world", "how", "are", "you"};
    std::vector<std::string> dest(source.size());
    
    std::cout << "Before move - source size: " << source.size() << std::endl;
    
    // Move elements (source strings will be in moved-from state)
    std::move(source.begin(), source.end(), dest.begin());
    
    std::cout << "After move - dest contents: ";
    for (const auto& str : dest) {
        std::cout << str << " ";
    }
    std::cout << std::endl;
    
    std::cout << "After move - source contents: ";
    for (const auto& str : source) {
        std::cout << "'" << str << "' ";
    }
    std::cout << std::endl;
}
```

### std::transform
```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <cctype>
#include <iostream>

void transform_examples() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> squares(numbers.size());
    
    // Transform with unary operation
    std::transform(numbers.begin(), numbers.end(), squares.begin(),
                   [](int n) { return n * n; });
    
    std::cout << "Squares: ";
    for (const auto& item : squares) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Transform with binary operation
    std::vector<int> other = {10, 20, 30, 40, 50};
    std::vector<int> sums(numbers.size());
    
    std::transform(numbers.begin(), numbers.end(), other.begin(), sums.begin(),
                   [](int a, int b) { return a + b; });
    
    std::cout << "Sums: ";
    for (const auto& item : sums) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Transform string to uppercase
    std::string text = "hello world";
    std::transform(text.begin(), text.end(), text.begin(),
                   [](char c) { return std::toupper(c); });
    
    std::cout << "Uppercase: " << text << std::endl;
}
```

### std::replace, std::replace_if
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void replace_examples() {
    std::vector<int> vec = {1, 2, 3, 2, 4, 2, 5};
    
    std::cout << "Original: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Replace specific value
    std::replace(vec.begin(), vec.end(), 2, 99);
    
    std::cout << "After replacing 2 with 99: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Replace with condition
    std::vector<int> vec2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::replace_if(vec2.begin(), vec2.end(), 
                    [](int n) { return n % 2 == 0; }, 0);
    
    std::cout << "After replacing even numbers with 0: ";
    for (const auto& item : vec2) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

### std::fill, std::fill_n, std::generate
```cpp
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

void fill_generate_examples() {
    std::vector<int> vec(10);
    
    // Fill with specific value
    std::fill(vec.begin(), vec.end(), 42);
    
    std::cout << "Filled with 42: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Fill first n elements
    std::fill_n(vec.begin(), 5, 99);
    
    std::cout << "First 5 filled with 99: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Generate with function
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    
    std::cout << "Generated random numbers: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Sorting and Related Operations

### std::sort, std::partial_sort, std::nth_element
```cpp
#include <algorithm>
#include <vector>
#include <functional>
#include <iostream>

void sort_examples() {
    std::vector<int> vec = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    
    // Basic sort
    std::vector<int> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    
    std::cout << "Sorted: ";
    for (const auto& item : sorted_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Sort in descending order
    std::vector<int> desc_vec = vec;
    std::sort(desc_vec.begin(), desc_vec.end(), std::greater<int>());
    
    std::cout << "Sorted descending: ";
    for (const auto& item : desc_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Partial sort - only first 4 elements are sorted
    std::vector<int> partial_vec = vec;
    std::partial_sort(partial_vec.begin(), partial_vec.begin() + 4, partial_vec.end());
    
    std::cout << "Partial sort (first 4): ";
    for (const auto& item : partial_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // nth_element - element at position n is in its correct sorted position
    std::vector<int> nth_vec = vec;
    std::nth_element(nth_vec.begin(), nth_vec.begin() + 4, nth_vec.end());
    
    std::cout << "After nth_element (n=4): ";
    for (const auto& item : nth_vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "5th element: " << nth_vec[4] << std::endl;
}
```

### std::stable_sort
```cpp
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

void stable_sort_examples() {
    struct Person {
        std::string name;
        int age;
    };
    
    std::vector<Person> people = {
        {"Alice", 30},
        {"Bob", 25},
        {"Charlie", 30},
        {"David", 25},
        {"Eve", 35}
    };
    
    // Sort by age, preserving relative order for equal ages
    std::stable_sort(people.begin(), people.end(), 
                     [](const Person& a, const Person& b) {
                         return a.age < b.age;
                     });
    
    std::cout << "Stable sort by age:" << std::endl;
    for (const auto& person : people) {
        std::cout << person.name << " (" << person.age << ")" << std::endl;
    }
}
```

## Binary Search Operations

### std::lower_bound, std::upper_bound, std::equal_range
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void binary_search_examples() {
    std::vector<int> vec = {1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // Find first position where 2 could be inserted
    auto lower = std::lower_bound(vec.begin(), vec.end(), 2);
    std::cout << "Lower bound of 2: position " << std::distance(vec.begin(), lower) << std::endl;
    
    // Find first position where element > 2 could be inserted
    auto upper = std::upper_bound(vec.begin(), vec.end(), 2);
    std::cout << "Upper bound of 2: position " << std::distance(vec.begin(), upper) << std::endl;
    
    // Get range of all elements equal to 2
    auto range = std::equal_range(vec.begin(), vec.end(), 2);
    std::cout << "Equal range of 2: [" << std::distance(vec.begin(), range.first) 
              << ", " << std::distance(vec.begin(), range.second) << ")" << std::endl;
    
    // Count occurrences using equal_range
    int count = std::distance(range.first, range.second);
    std::cout << "Count of 2: " << count << std::endl;
}
```

### std::binary_search
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void binary_search_check_examples() {
    std::vector<int> vec = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    
    // Check if element exists (vector must be sorted)
    std::cout << "Is 7 in vector? " << std::binary_search(vec.begin(), vec.end(), 7) << std::endl;
    std::cout << "Is 8 in vector? " << std::binary_search(vec.begin(), vec.end(), 8) << std::endl;
    
    // Binary search with custom comparator
    std::vector<std::string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    
    std::cout << "Is 'cherry' in vector? " 
              << std::binary_search(words.begin(), words.end(), "cherry") << std::endl;
    std::cout << "Is 'grape' in vector? " 
              << std::binary_search(words.begin(), words.end(), "grape") << std::endl;
}
```

## Set Operations

### std::set_union, std::set_intersection
```cpp
#include <algorithm>
#include <vector>
#include <iterator>
#include <iostream>

void set_operations_examples() {
    std::vector<int> set1 = {1, 2, 3, 4, 5};
    std::vector<int> set2 = {3, 4, 5, 6, 7, 8};
    
    std::vector<int> result;
    
    // Union of two sorted sets
    std::set_union(set1.begin(), set1.end(),
                   set2.begin(), set2.end(),
                   std::back_inserter(result));
    
    std::cout << "Union: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Intersection of two sorted sets
    result.clear();
    std::set_intersection(set1.begin(), set1.end(),
                          set2.begin(), set2.end(),
                          std::back_inserter(result));
    
    std::cout << "Intersection: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Difference (elements in set1 but not in set2)
    result.clear();
    std::set_difference(set1.begin(), set1.end(),
                        set2.begin(), set2.end(),
                        std::back_inserter(result));
    
    std::cout << "Difference (set1 - set2): ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Symmetric difference (elements in either set but not in both)
    result.clear();
    std::set_symmetric_difference(set1.begin(), set1.end(),
                                  set2.begin(), set2.end(),
                                  std::back_inserter(result));
    
    std::cout << "Symmetric difference: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Heap Operations

### std::make_heap, std::push_heap, std::pop_heap
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void heap_examples() {
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};
    
    std::cout << "Original: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Create max heap
    std::make_heap(vec.begin(), vec.end());
    
    std::cout << "After make_heap: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "Max element: " << vec.front() << std::endl;
    
    // Add element to heap
    vec.push_back(8);
    std::push_heap(vec.begin(), vec.end());
    
    std::cout << "After adding 8: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "Max element: " << vec.front() << std::endl;
    
    // Remove max element
    std::pop_heap(vec.begin(), vec.end());
    int max_elem = vec.back();
    vec.pop_back();
    
    std::cout << "Removed max element: " << max_elem << std::endl;
    std::cout << "After pop_heap: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Sort using heap
    std::sort_heap(vec.begin(), vec.end());
    
    std::cout << "After sort_heap: ";
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Min/Max Operations

### std::min, std::max, std::minmax
```cpp
#include <algorithm>
#include <vector>
#include <iostream>

void min_max_examples() {
    // Basic min/max
    int a = 5, b = 3;
    std::cout << "min(5, 3) = " << std::min(a, b) << std::endl;
    std::cout << "max(5, 3) = " << std::max(a, b) << std::endl;
    
    // minmax returns pair
    auto result = std::minmax(a, b);
    std::cout << "minmax(5, 3) = {" << result.first << ", " << result.second << "}" << std::endl;
    
    // With initializer list
    auto list_result = std::minmax({1, 5, 2, 8, 3});
    std::cout << "minmax of {1, 5, 2, 8, 3} = {" << list_result.first 
              << ", " << list_result.second << "}" << std::endl;
    
    // Find min/max elements in container
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};
    
    auto min_it = std::min_element(vec.begin(), vec.end());
    auto max_it = std::max_element(vec.begin(), vec.end());
    
    std::cout << "Min element: " << *min_it << " at position " 
              << std::distance(vec.begin(), min_it) << std::endl;
    std::cout << "Max element: " << *max_it << " at position " 
              << std::distance(vec.begin(), max_it) << std::endl;
    
    // minmax_element returns pair of iterators
    auto minmax_it = std::minmax_element(vec.begin(), vec.end());
    std::cout << "Min: " << *minmax_it.first << ", Max: " << *minmax_it.second << std::endl;
}
```

## Numeric Operations

### std::accumulate
```cpp
#include <numeric>
#include <vector>
#include <string>
#include <iostream>

void accumulate_examples() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // Sum all elements
    int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    std::cout << "Sum: " << sum << std::endl;
    
    // Product of all elements
    int product = std::accumulate(numbers.begin(), numbers.end(), 1, 
                                  [](int a, int b) { return a * b; });
    std::cout << "Product: " << product << std::endl;
    
    // Concatenate strings
    std::vector<std::string> words = {"Hello", " ", "World", "!"};
    std::string sentence = std::accumulate(words.begin(), words.end(), std::string(""));
    std::cout << "Concatenated: " << sentence << std::endl;
    
    // Count even numbers
    int even_count = std::accumulate(numbers.begin(), numbers.end(), 0,
                                     [](int count, int n) {
                                         return count + (n % 2 == 0 ? 1 : 0);
                                     });
    std::cout << "Even numbers count: " << even_count << std::endl;
}
```

### std::inner_product
```cpp
#include <numeric>
#include <vector>
#include <iostream>

void inner_product_examples() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {5, 4, 3, 2, 1};
    
    // Dot product
    int dot_product = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);
    std::cout << "Dot product: " << dot_product << std::endl;
    
    // Custom operations
    int custom_result = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0,
                                           [](int a, int b) { return a + b; },    // accumulation op
                                           [](int a, int b) { return a * b * 2; }); // transform op
    std::cout << "Custom inner product: " << custom_result << std::endl;
}
```

### std::partial_sum, std::adjacent_difference
```cpp
#include <numeric>
#include <vector>
#include <iostream>

void partial_sum_adjacent_diff_examples() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> result(numbers.size());
    
    // Partial sums (running totals)
    std::partial_sum(numbers.begin(), numbers.end(), result.begin());
    
    std::cout << "Original: ";
    for (const auto& item : numbers) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Partial sums: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Adjacent differences
    std::adjacent_difference(numbers.begin(), numbers.end(), result.begin());
    
    std::cout << "Adjacent differences: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    
    // Custom operations
    std::vector<int> fibonacci = {1, 1, 2, 3, 5, 8, 13};
    std::partial_sum(fibonacci.begin(), fibonacci.end(), result.begin(),
                     [](int a, int b) { return std::max(a, b); });
    
    std::cout << "Running maximum: ";
    for (const auto& item : result) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}
```

## Algorithm Composition Examples

### Complex Data Processing Pipeline
```cpp
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <random>

struct Product {
    std::string name;
    double price;
    int quantity;
    std::string category;
};

void data_processing_pipeline() {
    std::vector<Product> products = {
        {"Laptop", 999.99, 5, "Electronics"},
        {"Book", 19.99, 100, "Books"},
        {"Phone", 699.99, 15, "Electronics"},
        {"Pen", 1.99, 200, "Office"},
        {"Monitor", 299.99, 8, "Electronics"},
        {"Notebook", 4.99, 50, "Office"},
        {"Tablet", 399.99, 12, "Electronics"}
    };
    
    // Pipeline: Filter electronics -> Sort by price -> Calculate total value
    
    // Step 1: Filter electronics
    std::vector<Product> electronics;
    std::copy_if(products.begin(), products.end(), std::back_inserter(electronics),
                 [](const Product& p) { return p.category == "Electronics"; });
    
    std::cout << "Electronics products:" << std::endl;
    std::for_each(electronics.begin(), electronics.end(),
                  [](const Product& p) {
                      std::cout << p.name << " - $" << p.price << std::endl;
                  });
    
    // Step 2: Sort by price (descending)
    std::sort(electronics.begin(), electronics.end(),
              [](const Product& a, const Product& b) {
                  return a.price > b.price;
              });
    
    std::cout << "\nSorted by price (descending):" << std::endl;
    std::for_each(electronics.begin(), electronics.end(),
                  [](const Product& p) {
                      std::cout << p.name << " - $" << p.price << std::endl;
                  });
    
    // Step 3: Calculate total value
    double total_value = std::accumulate(electronics.begin(), electronics.end(), 0.0,
                                         [](double sum, const Product& p) {
                                             return sum + (p.price * p.quantity);
                                         });
    
    std::cout << "\nTotal value of electronics: $" << total_value << std::endl;
    
    // Step 4: Find most expensive product
    auto max_price_it = std::max_element(electronics.begin(), electronics.end(),
                                         [](const Product& a, const Product& b) {
                                             return a.price < b.price;
                                         });
    
    if (max_price_it != electronics.end()) {
        std::cout << "Most expensive: " << max_price_it->name 
                  << " at $" << max_price_it->price << std::endl;
    }
}
```

### Text Processing Example
```cpp
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>
#include <iostream>

void text_processing_example() {
    std::string text = "The quick brown fox jumps over the lazy dog. "
                       "The dog was sleeping under the tree.";
    
    // Convert to lowercase
    std::transform(text.begin(), text.end(), text.begin(),
                   [](char c) { return std::tolower(c); });
    
    // Split into words
    std::istringstream iss(text);
    std::vector<std::string> words;
    std::string word;
    
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(),
                                  [](char c) { return std::ispunct(c); }),
                   word.end());
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    // Sort words
    std::sort(words.begin(), words.end());
    
    // Remove duplicates
    auto last = std::unique(words.begin(), words.end());
    words.erase(last, words.end());
    
    std::cout << "Unique words (sorted):" << std::endl;
    std::for_each(words.begin(), words.end(),
                  [](const std::string& w) { std::cout << w << " "; });
    std::cout << std::endl;
    
    // Find longest word
    auto longest = std::max_element(words.begin(), words.end(),
                                    [](const std::string& a, const std::string& b) {
                                        return a.length() < b.length();
                                    });
    
    if (longest != words.end()) {
        std::cout << "Longest word: " << *longest 
                  << " (" << longest->length() << " characters)" << std::endl;
    }
    
    // Count words starting with 't'
    int t_words = std::count_if(words.begin(), words.end(),
                                [](const std::string& w) {
                                    return !w.empty() && w[0] == 't';
                                });
    
    std::cout << "Words starting with 't': " << t_words << std::endl;
}
```

## Performance Considerations

### Algorithm Complexity Examples
```cpp
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>

void performance_comparison() {
    const size_t SIZE = 1000000;
    std::vector<int> vec(SIZE);
    
    // Fill with random numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000000);
    
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    
    // Time std::find (O(n))
    auto start = std::chrono::high_resolution_clock::now();
    auto it = std::find(vec.begin(), vec.end(), 500000);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "std::find took: " << duration.count() << " microseconds" << std::endl;
    
    // Sort the vector for binary search
    std::sort(vec.begin(), vec.end());
    
    // Time std::binary_search (O(log n))
    start = std::chrono::high_resolution_clock::now();
    bool found = std::binary_search(vec.begin(), vec.end(), 500000);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "std::binary_search took: " << duration.count() << " microseconds" << std::endl;
    
    std::cout << "Element found: " << (found ? "yes" : "no") << std::endl;
}
```

## Complete Example Program

```cpp
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

int main() {
    std::cout << "=== STL Algorithms Examples ===" << std::endl;
    
    std::cout << "\n--- Find Operations ---" << std::endl;
    find_examples();
    
    std::cout << "\n--- Count Operations ---" << std::endl;
    count_examples();
    
    std::cout << "\n--- For Each ---" << std::endl;
    for_each_examples();
    
    std::cout << "\n--- Condition Checks ---" << std::endl;
    condition_check_examples();
    
    std::cout << "\n--- Copy Operations ---" << std::endl;
    copy_examples();
    
    std::cout << "\n--- Move Operations ---" << std::endl;
    move_examples();
    
    std::cout << "\n--- Transform Operations ---" << std::endl;
    transform_examples();
    
    std::cout << "\n--- Replace Operations ---" << std::endl;
    replace_examples();
    
    std::cout << "\n--- Fill and Generate ---" << std::endl;
    fill_generate_examples();
    
    std::cout << "\n--- Sorting Operations ---" << std::endl;
    sort_examples();
    stable_sort_examples();
    
    std::cout << "\n--- Binary Search Operations ---" << std::endl;
    binary_search_examples();
    binary_search_check_examples();
    
    std::cout << "\n--- Set Operations ---" << std::endl;
    set_operations_examples();
    
    std::cout << "\n--- Heap Operations ---" << std::endl;
    heap_examples();
    
    std::cout << "\n--- Min/Max Operations ---" << std::endl;
    min_max_examples();
    
    std::cout << "\n--- Numeric Operations ---" << std::endl;
    accumulate_examples();
    inner_product_examples();
    partial_sum_adjacent_diff_examples();
    
    std::cout << "\n--- Complex Examples ---" << std::endl;
    data_processing_pipeline();
    text_processing_example();
    
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    performance_comparison();
    
    return 0;
}
```

## Algorithm Selection Guidelines

### When to use which algorithm:

1. **Finding Elements**: Use `find` for unsorted data, `binary_search` for sorted data
2. **Sorting**: Use `sort` for general purpose, `stable_sort` when order matters, `partial_sort` when you only need top-k elements
3. **Set Operations**: Require sorted input, very efficient for mathematical set operations
4. **Heap Operations**: Useful for priority queues and finding top-k elements
5. **Numeric Operations**: Specialized algorithms for mathematical computations

## Best Practices

1. **Use appropriate algorithms**: Choose the right algorithm for your data and requirements
2. **Consider complexity**: Understand time and space complexity of algorithms
3. **Use iterators properly**: Ensure iterators are valid and meet algorithm requirements
4. **Leverage function objects**: Use lambdas and function objects for custom behavior
5. **Compose algorithms**: Combine simple algorithms to solve complex problems
6. **Prefer algorithms over raw loops**: STL algorithms are optimized and less error-prone

## Exercises

1. Implement a function that finds the k-th largest element in unsorted data
2. Create a word frequency counter using STL algorithms
3. Implement merge sort using STL algorithms
4. Write a function that removes consecutive duplicates from a vector
5. Create a text analyzer that finds palindromes in a string
6. Implement a simple MapReduce-style data processing pipeline
7. Write a function that partitions data based on multiple criteria
8. Create a statistical analysis tool using numeric algorithms
