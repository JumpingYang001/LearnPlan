# Advanced Boost Libraries

*Duration: 2 weeks*

## Overview

This section covers specialized Boost libraries for advanced use cases including graph algorithms, geometric computations, interprocess communication, and optional/variant types.

## Learning Topics

### Boost.Graph
- Graph data structures and representations
- Graph algorithms (traversal, shortest path, etc.)
- Custom graph types and property maps
- Visualization and analysis tools

### Boost.Geometry
- Geometric algorithms and spatial operations
- Coordinate systems and transformations
- Spatial indexing and queries
- Integration with geographic information systems

### Boost.Interprocess
- Shared memory management
- Interprocess communication mechanisms
- Synchronization between processes
- Memory-mapped files and allocators

### Boost.Optional, Boost.Variant, Boost.Any
- Optional value handling
- Type-safe unions and variant types
- Type-erased value containers
- Comparisons with modern C++ equivalents

## Code Examples

### Boost.Graph - Basic Graph Operations
```cpp
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <iostream>
#include <vector>

// Define graph type
typedef boost::adjacency_list<
    boost::vecS,      // OutEdgeList
    boost::vecS,      // VertexList  
    boost::directedS, // Directed
    boost::no_property,     // VertexProperty
    boost::property<boost::edge_weight_t, int> // EdgeProperty
> Graph;

typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;

void demonstrate_basic_graph() {
    std::cout << "=== Basic Graph Operations ===\n";
    
    // Create graph
    Graph g(6); // 6 vertices
    
    // Add edges with weights
    boost::add_edge(0, 1, 7, g);
    boost::add_edge(0, 2, 9, g);
    boost::add_edge(0, 5, 14, g);
    boost::add_edge(1, 2, 10, g);
    boost::add_edge(1, 3, 15, g);
    boost::add_edge(2, 3, 11, g);
    boost::add_edge(2, 5, 2, g);
    boost::add_edge(3, 4, 6, g);
    boost::add_edge(4, 5, 9, g);
    
    // Graph properties
    std::cout << "Number of vertices: " << boost::num_vertices(g) << "\n";
    std::cout << "Number of edges: " << boost::num_edges(g) << "\n";
    
    // Iterate through vertices
    std::cout << "Vertices: ";
    auto vertices = boost::vertices(g);
    for (auto it = vertices.first; it != vertices.second; ++it) {
        std::cout << *it << " ";
    }
    std::cout << "\n";
    
    // Iterate through edges
    std::cout << "Edges (with weights):\n";
    auto edges = boost::edges(g);
    auto weight_map = boost::get(boost::edge_weight, g);
    
    for (auto it = edges.first; it != edges.second; ++it) {
        Vertex src = boost::source(*it, g);
        Vertex tgt = boost::target(*it, g);
        int weight = weight_map[*it];
        std::cout << "  " << src << " -> " << tgt << " (weight: " << weight << ")\n";
    }
}
```

### Graph Algorithms and Traversal
```cpp
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/visitors.hpp>
#include <iostream>
#include <vector>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UndirectedGraph;

// BFS Visitor
class BFSVisitor : public boost::default_bfs_visitor {
public:
    void discover_vertex(UndirectedGraph::vertex_descriptor v, const UndirectedGraph&) {
        std::cout << "BFS discovered vertex: " << v << "\n";
    }
    
    void examine_edge(UndirectedGraph::edge_descriptor e, const UndirectedGraph& g) {
        std::cout << "BFS examining edge: " << boost::source(e, g) 
                  << " -> " << boost::target(e, g) << "\n";
    }
};

// DFS Visitor
class DFSVisitor : public boost::default_dfs_visitor {
public:
    void discover_vertex(UndirectedGraph::vertex_descriptor v, const UndirectedGraph&) {
        std::cout << "DFS discovered vertex: " << v << "\n";
    }
    
    void finish_vertex(UndirectedGraph::vertex_descriptor v, const UndirectedGraph&) {
        std::cout << "DFS finished vertex: " << v << "\n";
    }
};

void demonstrate_graph_algorithms() {
    std::cout << "\n=== Graph Algorithms ===\n";
    
    UndirectedGraph g(6);
    
    // Add edges to create a connected graph
    boost::add_edge(0, 1, g);
    boost::add_edge(0, 2, g);
    boost::add_edge(1, 3, g);
    boost::add_edge(2, 4, g);
    boost::add_edge(3, 5, g);
    boost::add_edge(4, 5, g);
    
    // Breadth-First Search
    std::cout << "Breadth-First Search from vertex 0:\n";
    BFSVisitor bfs_visitor;
    boost::breadth_first_search(g, 0, boost::visitor(bfs_visitor));
    
    std::cout << "\n";
    
    // Depth-First Search  
    std::cout << "Depth-First Search from vertex 0:\n";
    DFSVisitor dfs_visitor;
    boost::depth_first_search(g, boost::visitor(dfs_visitor));
}
```

### Shortest Path Algorithms
```cpp
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <iostream>
#include <vector>

typedef boost::adjacency_list<
    boost::listS, 
    boost::vecS, 
    boost::directedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, int>
> WeightedGraph;

void demonstrate_shortest_paths() {
    std::cout << "\n=== Shortest Path Algorithms ===\n";
    
    WeightedGraph g(5);
    auto weight_map = boost::get(boost::edge_weight, g);
    
    // Add weighted edges
    boost::add_edge(0, 1, 10, g);
    boost::add_edge(0, 4, 5, g);
    boost::add_edge(1, 2, 1, g);
    boost::add_edge(1, 4, 2, g);
    boost::add_edge(2, 3, 4, g);
    boost::add_edge(3, 0, 7, g);
    boost::add_edge(3, 2, 6, g);
    boost::add_edge(4, 1, 3, g);
    boost::add_edge(4, 2, 9, g);
    boost::add_edge(4, 3, 2, g);
    
    // Dijkstra's algorithm
    std::vector<int> distances(boost::num_vertices(g));
    std::vector<WeightedGraph::vertex_descriptor> predecessors(boost::num_vertices(g));
    
    boost::dijkstra_shortest_paths(
        g, 0,
        boost::predecessor_map(&predecessors[0])
        .distance_map(&distances[0])
        .weight_map(weight_map)
    );
    
    std::cout << "Shortest distances from vertex 0 (Dijkstra):\n";
    for (size_t i = 0; i < distances.size(); ++i) {
        std::cout << "  To vertex " << i << ": " << distances[i] << "\n";
    }
    
    // Print shortest path to vertex 3
    std::cout << "Shortest path from 0 to 3: ";
    std::vector<int> path;
    int current = 3;
    while (current != 0) {
        path.push_back(current);
        current = predecessors[current];
    }
    path.push_back(0);
    
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        std::cout << *it;
        if (it != path.rend() - 1) std::cout << " -> ";
    }
    std::cout << "\n";
}
```

### Boost.Geometry - Spatial Operations
```cpp
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <iostream>
#include <vector>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 2, bg::cs::cartesian> Point;
typedef bg::model::polygon<Point> Polygon;
typedef bg::model::linestring<Point> Linestring;
typedef bg::model::box<Point> Box;

void demonstrate_basic_geometry() {
    std::cout << "=== Basic Geometry Operations ===\n";
    
    // Create points
    Point p1(1.0, 2.0);
    Point p2(4.0, 6.0);
    Point p3(0.0, 0.0);
    
    std::cout << "Point 1: " << bg::dsv(p1) << "\n";
    std::cout << "Point 2: " << bg::dsv(p2) << "\n";
    
    // Distance calculations
    double dist = bg::distance(p1, p2);
    std::cout << "Distance between p1 and p2: " << dist << "\n";
    
    // Create polygon
    Polygon poly;
    bg::exterior_ring(poly) = {{0, 0}, {0, 5}, {5, 5}, {5, 0}, {0, 0}};
    
    std::cout << "Polygon: " << bg::dsv(poly) << "\n";
    std::cout << "Polygon area: " << bg::area(poly) << "\n";
    std::cout << "Polygon perimeter: " << bg::perimeter(poly) << "\n";
    
    // Point-in-polygon test
    bool inside1 = bg::within(Point(2.5, 2.5), poly);
    bool inside2 = bg::within(Point(6.0, 6.0), poly);
    
    std::cout << "Point (2.5, 2.5) inside polygon: " << std::boolalpha << inside1 << "\n";
    std::cout << "Point (6.0, 6.0) inside polygon: " << std::boolalpha << inside2 << "\n";
    
    // Create linestring
    Linestring line = {{0, 0}, {1, 1}, {2, 0}, {3, 1}};
    std::cout << "Linestring: " << bg::dsv(line) << "\n";
    std::cout << "Linestring length: " << bg::length(line) << "\n";
    
    // Intersection
    std::vector<Point> intersections;
    bg::intersection(line, poly, intersections);
    std::cout << "Line-polygon intersections: " << intersections.size() << "\n";
}
```

### Spatial Indexing with R-tree
```cpp
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <iostream>
#include <vector>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<double, 2, bg::cs::cartesian> Point;
typedef bg::model::box<Point> Box;
typedef std::pair<Point, int> PointValue; // Point with associated ID

void demonstrate_spatial_indexing() {
    std::cout << "\n=== Spatial Indexing (R-tree) ===\n";
    
    // Create R-tree
    bgi::rtree<PointValue, bgi::quadratic<16>> rtree;
    
    // Insert points with IDs
    std::vector<PointValue> points = {
        {Point(1, 1), 1}, {Point(3, 2), 2}, {Point(5, 3), 3},
        {Point(2, 5), 4}, {Point(4, 4), 5}, {Point(6, 1), 6},
        {Point(1, 6), 7}, {Point(7, 7), 8}, {Point(2, 2), 9}
    };
    
    for (const auto& pv : points) {
        rtree.insert(pv);
    }
    
    std::cout << "Inserted " << rtree.size() << " points into R-tree\n";
    
    // Nearest neighbor search
    Point query_point(3, 3);
    std::vector<PointValue> nearest;
    rtree.query(bgi::nearest(query_point, 3), std::back_inserter(nearest));
    
    std::cout << "3 nearest neighbors to " << bg::dsv(query_point) << ":\n";
    for (const auto& pv : nearest) {
        double dist = bg::distance(query_point, pv.first);
        std::cout << "  ID " << pv.second << " at " << bg::dsv(pv.first) 
                  << " (distance: " << dist << ")\n";
    }
    
    // Range query
    Box query_box(Point(1, 1), Point(4, 4));
    std::vector<PointValue> in_range;
    rtree.query(bgi::intersects(query_box), std::back_inserter(in_range));
    
    std::cout << "Points within box " << bg::dsv(query_box) << ":\n";
    for (const auto& pv : in_range) {
        std::cout << "  ID " << pv.second << " at " << bg::dsv(pv.first) << "\n";
    }
    
    // Spatial predicate queries
    std::vector<PointValue> covered_points;
    rtree.query(bgi::covered_by(query_box), std::back_inserter(covered_points));
    
    std::cout << "Points covered by box: " << covered_points.size() << "\n";
}
```

### Boost.Interprocess - Shared Memory
```cpp
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <iostream>
#include <string>
#include <cstring>

namespace bip = boost::interprocess;

struct SharedData {
    bip::interprocess_mutex mutex;
    int counter;
    char message[256];
    
    SharedData() : counter(0) {
        std::strcpy(message, "Initial message");
    }
};

void demonstrate_shared_memory() {
    std::cout << "=== Shared Memory ===\n";
    
    try {
        // Remove any existing shared memory
        bip::shared_memory_object::remove("MySharedMemory");
        
        // Create shared memory object
        bip::shared_memory_object shm(
            bip::create_only,
            "MySharedMemory",
            bip::read_write
        );
        
        // Set size
        shm.truncate(sizeof(SharedData));
        
        // Map the shared memory
        bip::mapped_region region(shm, bip::read_write);
        
        // Construct the shared data
        SharedData* data = new(region.get_address()) SharedData;
        
        std::cout << "Created shared memory region\n";
        std::cout << "Initial counter: " << data->counter << "\n";
        std::cout << "Initial message: " << data->message << "\n";
        
        // Simulate access from "another process"
        {
            bip::scoped_lock<bip::interprocess_mutex> lock(data->mutex);
            data->counter++;
            std::strcpy(data->message, "Updated from process");
        }
        
        std::cout << "After update:\n";
        std::cout << "Counter: " << data->counter << "\n";
        std::cout << "Message: " << data->message << "\n";
        
        // Cleanup
        bip::shared_memory_object::remove("MySharedMemory");
        std::cout << "Cleaned up shared memory\n";
        
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Interprocess error: " << e.what() << "\n";
    }
}
```

### Shared Memory Containers
```cpp
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <iostream>

namespace bip = boost::interprocess;

// Shared memory allocator for int
typedef bip::allocator<int, bip::managed_shared_memory::segment_manager> ShmIntAllocator;
typedef bip::vector<int, ShmIntAllocator> ShmIntVector;

// Shared memory allocator for char
typedef bip::allocator<char, bip::managed_shared_memory::segment_manager> ShmCharAllocator;
typedef bip::basic_string<char, std::char_traits<char>, ShmCharAllocator> ShmString;

void demonstrate_shared_containers() {
    std::cout << "\n=== Shared Memory Containers ===\n";
    
    try {
        // Remove any existing shared memory
        bip::shared_memory_object::remove("MyManagedMemory");
        
        // Create managed shared memory
        bip::managed_shared_memory segment(
            bip::create_only,
            "MyManagedMemory",
            65536  // 64KB
        );
        
        // Create allocators
        ShmIntAllocator int_alloc(segment.get_segment_manager());
        ShmCharAllocator char_alloc(segment.get_segment_manager());
        
        // Create shared vector
        ShmIntVector* shared_vector = segment.construct<ShmIntVector>("MyVector")(int_alloc);
        
        // Add elements
        for (int i = 0; i < 10; ++i) {
            shared_vector->push_back(i * i);
        }
        
        std::cout << "Created shared vector with " << shared_vector->size() << " elements\n";
        std::cout << "Vector contents: ";
        for (const auto& val : *shared_vector) {
            std::cout << val << " ";
        }
        std::cout << "\n";
        
        // Create shared string
        ShmString* shared_string = segment.construct<ShmString>("MyString")(char_alloc);
        *shared_string = "Hello from shared memory!";
        
        std::cout << "Shared string: " << *shared_string << "\n";
        std::cout << "String length: " << shared_string->length() << "\n";
        
        // Memory usage info
        std::cout << "Free memory: " << segment.get_free_memory() << " bytes\n";
        std::cout << "Used memory: " << (65536 - segment.get_free_memory()) << " bytes\n";
        
        // Cleanup
        segment.destroy<ShmIntVector>("MyVector");
        segment.destroy<ShmString>("MyString");
        bip::shared_memory_object::remove("MyManagedMemory");
        std::cout << "Cleaned up shared containers\n";
        
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Interprocess error: " << e.what() << "\n";
    }
}
```

### Boost.Optional, Variant, and Any
```cpp
#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <boost/any.hpp>
#include <iostream>
#include <string>
#include <vector>

// Function that may or may not return a value
boost::optional<int> safe_divide(int numerator, int denominator) {
    if (denominator == 0) {
        return boost::none; // or boost::optional<int>()
    }
    return numerator / denominator;
}

// Variant visitor
class VariantPrinter : public boost::static_visitor<void> {
public:
    void operator()(int value) const {
        std::cout << "Integer: " << value << "\n";
    }
    
    void operator()(double value) const {
        std::cout << "Double: " << value << "\n";
    }
    
    void operator()(const std::string& value) const {
        std::cout << "String: " << value << "\n";
    }
};

void demonstrate_optional_variant_any() {
    std::cout << "\n=== Optional, Variant, and Any ===\n";
    
    // Boost.Optional
    std::cout << "--- Optional ---\n";
    
    auto result1 = safe_divide(10, 2);
    auto result2 = safe_divide(10, 0);
    
    if (result1) {
        std::cout << "Division result: " << *result1 << "\n";
    } else {
        std::cout << "Division failed\n";
    }
    
    if (result2) {
        std::cout << "Division result: " << *result2 << "\n";
    } else {
        std::cout << "Division by zero!\n";
    }
    
    // Optional with default value
    int value = result2.get_value_or(-1);
    std::cout << "Result with default: " << value << "\n";
    
    // Boost.Variant
    std::cout << "\n--- Variant ---\n";
    
    typedef boost::variant<int, double, std::string> MyVariant;
    
    std::vector<MyVariant> variants = {
        42,
        3.14,
        std::string("Hello Variant!")
    };
    
    VariantPrinter printer;
    for (const auto& var : variants) {
        boost::apply_visitor(printer, var);
    }
    
    // Type checking and extraction
    MyVariant v = 42;
    if (int* i = boost::get<int>(&v)) {
        std::cout << "Variant contains int: " << *i << "\n";
    }
    
    try {
        std::string s = boost::get<std::string>(v);
        std::cout << "This won't be printed\n";
    } catch (const boost::bad_get& e) {
        std::cout << "Bad variant access: " << e.what() << "\n";
    }
    
    // Boost.Any
    std::cout << "\n--- Any ---\n";
    
    std::vector<boost::any> any_values;
    any_values.push_back(42);
    any_values.push_back(3.14159);
    any_values.push_back(std::string("Any string"));
    any_values.push_back(std::vector<int>{1, 2, 3, 4, 5});
    
    for (size_t i = 0; i < any_values.size(); ++i) {
        const boost::any& val = any_values[i];
        std::cout << "Any[" << i << "] type: " << val.type().name() << "\n";
        
        // Type-safe extraction
        if (val.type() == typeid(int)) {
            std::cout << "  Value: " << boost::any_cast<int>(val) << "\n";
        } else if (val.type() == typeid(double)) {
            std::cout << "  Value: " << boost::any_cast<double>(val) << "\n";
        } else if (val.type() == typeid(std::string)) {
            std::cout << "  Value: " << boost::any_cast<std::string>(val) << "\n";
        } else if (val.type() == typeid(std::vector<int>)) {
            auto vec = boost::any_cast<std::vector<int>>(val);
            std::cout << "  Vector size: " << vec.size() << "\n";
        }
    }
    
    // Empty any
    boost::any empty_any;
    std::cout << "Empty any: " << std::boolalpha << empty_any.empty() << "\n";
}
```

## Practical Exercises

1. **Social Network Analyzer**
   - Model social connections using Boost.Graph
   - Implement friendship recommendations using graph algorithms
   - Calculate network metrics (centrality, clustering coefficient)

2. **GIS Application**
   - Build a geographic information system using Boost.Geometry
   - Implement spatial queries and route planning
   - Use R-tree indexing for efficient spatial lookups

3. **Distributed Computing Framework**
   - Create a work distribution system using Boost.Interprocess
   - Implement shared memory pools for data exchange
   - Add process monitoring and fault tolerance

4. **Configuration System**
   - Design a flexible configuration system using Boost.Variant
   - Support different value types and validation
   - Implement serialization and deserialization

## Performance Considerations

### Graph Operations
- Choose appropriate graph representations for your use case
- Consider memory layout and cache efficiency
- Use property maps efficiently for large graphs

### Geometric Computations
- Understand computational complexity of geometric algorithms
- Use spatial indexing for large datasets
- Consider numeric precision issues

### Interprocess Communication
- Minimize shared memory allocation/deallocation
- Design efficient synchronization strategies
- Consider memory alignment and padding

## Best Practices

1. **Graph Design**
   - Choose the right graph representation (adjacency list vs matrix)
   - Use property maps for vertex/edge attributes
   - Consider thread safety for concurrent access

2. **Geometric Programming**
   - Validate geometric inputs for robustness
   - Handle edge cases in geometric algorithms
   - Use appropriate coordinate systems

3. **Interprocess Programming**
   - Handle process failures gracefully
   - Design for data consistency across processes
   - Monitor shared memory usage

## Assessment

- Can model complex relationships using graph structures
- Understands spatial algorithms and indexing strategies
- Can implement interprocess communication safely
- Knows when to use optional/variant/any types appropriately

## Conclusion

This completes the comprehensive Boost Library learning track. You should now have:
- Deep understanding of Boost's core libraries
- Practical experience with advanced programming patterns
- Knowledge of performance considerations and best practices
- Ability to choose appropriate Boost libraries for specific problems

Continue practicing with real-world projects and stay updated with Boost releases and modern C++ alternatives.
