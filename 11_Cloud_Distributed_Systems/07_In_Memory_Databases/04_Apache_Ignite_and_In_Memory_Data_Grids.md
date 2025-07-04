# Apache Ignite and In-Memory Data Grids

*Duration: 2-3 weeks*

## Overview

Apache Ignite is a distributed database for high-performance computing with in-memory speed. It combines the capabilities of an in-memory data grid, compute grid, and streaming platform in a single, unified solution. This comprehensive guide will take you from basic concepts to advanced distributed computing scenarios.

### What is an In-Memory Data Grid (IMDG)?

An **In-Memory Data Grid** is a distributed computing architecture that:
- Stores data primarily in RAM across multiple machines
- Provides horizontal scalability by adding more nodes
- Offers data partitioning and replication for fault tolerance
- Enables real-time processing and analytics
- Supports ACID transactions across the distributed grid

### Why Apache Ignite?

```
Traditional Database          vs.          Apache Ignite
┌─────────────────┐                      ┌─────────────────┐
│ Disk Storage    │                      │ Memory Storage  │
│ Single Node     │                      │ Distributed     │
│ Limited Scale   │                      │ Horizontal Scale│
│ Slow Queries    │                      │ Fast Processing │
│ High Latency    │                      │ Low Latency     │
└─────────────────┘                      └─────────────────┘
```

## Core Concepts

### Apache Ignite Architecture

#### High-Level Architecture Diagram
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
├─────────────────────────────────────────────────────────────┤
│         Ignite Native APIs  │  SQL/JDBC  │  REST APIs      │
├─────────────────────────────────────────────────────────────┤
│                       Ignite Cluster                       │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐          │
│  │   Node 1  │    │   Node 2  │    │   Node 3  │          │
│  │┌─────────┐│    │┌─────────┐│    │┌─────────┐│          │
│  ││ Cache   ││    ││ Cache   ││    ││ Cache   ││          │
│  ││ Compute ││    ││ Compute ││    ││ Compute ││          │
│  ││Services ││    ││Services ││    ││Services ││          │
│  │└─────────┘│    │└─────────┘│    │└─────────┘│          │
│  └───────────┘    └───────────┘    └───────────┘          │
├─────────────────────────────────────────────────────────────┤
│              Discovery & Communication Layer                │
│                    (TCP/IP Discovery)                      │
└─────────────────────────────────────────────────────────────┘
```

#### Key Architectural Components

**1. Cluster Nodes**
- **Server Nodes**: Store data and execute computations
- **Client Nodes**: Connect to cluster without storing data
- **Baseline Topology**: Defines persistent nodes for data storage

**2. Data Grid Components**
```java
// Basic cluster setup
public class IgniteArchitectureDemo {
    public static void main(String[] args) {
        // Server node configuration
        IgniteConfiguration serverCfg = new IgniteConfiguration();
        serverCfg.setIgniteInstanceName("server-node");
        serverCfg.setClientMode(false); // Server mode
        
        // Client node configuration  
        IgniteConfiguration clientCfg = new IgniteConfiguration();
        clientCfg.setIgniteInstanceName("client-node");
        clientCfg.setClientMode(true); // Client mode
        
        // Data region configuration for memory management
        DataRegionConfiguration dataRegionCfg = new DataRegionConfiguration();
        dataRegionCfg.setName("default_region");
        dataRegionCfg.setInitialSize(100 * 1024 * 1024); // 100MB
        dataRegionCfg.setMaxSize(1024 * 1024 * 1024); // 1GB
        dataRegionCfg.setPersistenceEnabled(true); // Enable persistence
        
        DataStorageConfiguration dataStorageCfg = new DataStorageConfiguration();
        dataStorageCfg.setDefaultDataRegionConfiguration(dataRegionCfg);
        
        serverCfg.setDataStorageConfiguration(dataStorageCfg);
        
        // Start nodes
        Ignite serverNode = Ignition.start(serverCfg);
        Ignite clientNode = Ignition.start(clientCfg);
        
        System.out.println("Cluster topology: " + serverNode.cluster().nodes().size() + " nodes");
    }
}
```

**3. Memory Architecture**
```java
// Memory regions and data storage
public class MemoryManagementExample {
    public static void configureMemory() {
        IgniteConfiguration cfg = new IgniteConfiguration();
        
        // Configure multiple data regions
        DataRegionConfiguration hotDataRegion = new DataRegionConfiguration();
        hotDataRegion.setName("hot_data");
        hotDataRegion.setInitialSize(512 * 1024 * 1024); // 512MB
        hotDataRegion.setMaxSize(2L * 1024 * 1024 * 1024); // 2GB
        hotDataRegion.setEvictionPolicy(new LruEvictionPolicy(100000));
        
        DataRegionConfiguration coldDataRegion = new DataRegionConfiguration();
        coldDataRegion.setName("cold_data");
        coldDataRegion.setInitialSize(100 * 1024 * 1024); // 100MB
        coldDataRegion.setMaxSize(500 * 1024 * 1024); // 500MB
        coldDataRegion.setPersistenceEnabled(true);
        
        DataStorageConfiguration dataStorageCfg = new DataStorageConfiguration();
        dataStorageCfg.setDataRegionConfigurations(hotDataRegion, coldDataRegion);
        
        cfg.setDataStorageConfiguration(dataStorageCfg);
        
        Ignite ignite = Ignition.start(cfg);
    }
}
```

### Distributed Computing Capabilities

#### 1. Data Partitioning and Distribution

Apache Ignite automatically partitions data across cluster nodes using consistent hashing:

```java
public class DataPartitioningExample {
    public static void demonstratePartitioning() {
        Ignite ignite = Ignition.start();
        
        // Create cache with custom partitioning
        CacheConfiguration<Integer, Employee> cacheConfig = new CacheConfiguration<>();
        cacheConfig.setName("employee_cache");
        cacheConfig.setCacheMode(CacheMode.PARTITIONED);
        cacheConfig.setBackups(1); // One backup copy
        cacheConfig.setPartitionLossPolicy(PartitionLossPolicy.READ_ONLY_SAFE);
        
        // Custom affinity function for specific partitioning logic
        RendezvousAffinityFunction affinity = new RendezvousAffinityFunction();
        affinity.setPartitions(1024); // Number of partitions
        cacheConfig.setAffinity(affinity);
        
        IgniteCache<Integer, Employee> cache = ignite.getOrCreateCache(cacheConfig);
        
        // Insert data - automatically distributed across nodes
        for (int i = 0; i < 10000; i++) {
            Employee emp = new Employee(i, "Employee" + i, "Department" + (i % 10));
            cache.put(i, emp);
        }
        
        // Check data distribution
        for (ClusterNode node : ignite.cluster().nodes()) {
            Collection<Integer> keys = ignite.affinity("employee_cache")
                .mapKeysToNodes(cache.localEntries().stream()
                    .map(entry -> entry.getKey())
                    .collect(Collectors.toSet()))
                .get(node);
            
            System.out.println("Node " + node.id() + " has " + keys.size() + " keys");
        }
    }
}

class Employee {
    private int id;
    private String name;
    private String department;
    
    // Constructors, getters, setters
    public Employee(int id, String name, String department) {
        this.id = id;
        this.name = name; 
        this.department = department;
    }
    
    // ... getters and setters
}
```

#### 2. Compute Grid Capabilities

**Distributed Computing with Closures:**
```java
public class ComputeGridExample {
    public static void demonstrateDistributedComputing() {
        Ignite ignite = Ignition.start();
        IgniteCompute compute = ignite.compute();
        
        // Example 1: Simple distributed calculation
        Collection<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        
        // Distribute calculation across all nodes
        Collection<Integer> results = compute.apply(
            (Integer num) -> {
                System.out.println("Processing " + num + " on node: " + 
                    ignite.cluster().localNode().id());
                return num * num; // Square the number
            },
            numbers
        );
        
        System.out.println("Results: " + results);
        
        // Example 2: Map-Reduce pattern
        int sum = compute.execute(new SquareSumTask(), numbers);
        System.out.println("Sum of squares: " + sum);
        
        // Example 3: Broadcast computation
        compute.broadcast(() -> {
            System.out.println("Hello from node: " + ignite.cluster().localNode().id());
            // Perform node-specific operations
            return null;
        });
    }
}

// Custom compute task implementing Map-Reduce
class SquareSumTask extends ComputeTaskAdapter<Collection<Integer>, Integer> {
    @Override
    public Map<? extends ComputeJob, ClusterNode> map(List<ClusterNode> nodes, 
                                                     Collection<Integer> numbers) {
        Map<ComputeJob, ClusterNode> jobs = new HashMap<>();
        
        // Distribute numbers across available nodes
        Iterator<ClusterNode> nodeIter = nodes.iterator();
        for (Integer number : numbers) {
            if (!nodeIter.hasNext()) {
                nodeIter = nodes.iterator(); // Restart iterator
            }
            
            jobs.put(new SquareJob(number), nodeIter.next());
        }
        
        return jobs;
    }
    
    @Override
    public Integer reduce(List<ComputeJobResult> results) {
        int sum = 0;
        for (ComputeJobResult result : results) {
            sum += result.<Integer>getData();
        }
        return sum;
    }
}

class SquareJob extends ComputeJobAdapter {
    private final Integer number;
    
    public SquareJob(Integer number) {
        this.number = number;
    }
    
    @Override
    public Object execute() {
        System.out.println("Computing square of " + number);
        return number * number;
    }
}
```

#### 3. Collocated Computations
```java
public class CollocatedComputationExample {
    public static void demonstrateCollocation() {
        Ignite ignite = Ignition.start();
        
        // Create caches
        IgniteCache<Integer, Order> orderCache = ignite.getOrCreateCache("orders");
        IgniteCache<Integer, Customer> customerCache = ignite.getOrCreateCache("customers");
        
        // Insert test data
        populateData(orderCache, customerCache);
        
        // Collocated computation - process orders with customer data on same node
        IgniteCompute compute = ignite.compute();
        
        // Process orders for a specific customer
        Integer customerId = 1001;
        Collection<OrderSummary> summaries = compute.affinityCall(
            "orders", 
            customerId,
            () -> {
                // This closure runs on the node where customer data is stored
                Customer customer = customerCache.localPeek(customerId);
                Collection<Order> orders = getCustomerOrders(orderCache, customerId);
                
                return orders.stream()
                    .map(order -> new OrderSummary(order, customer))
                    .collect(Collectors.toList());
            }
        );
        
        summaries.forEach(System.out::println);
    }
    
    private static Collection<Order> getCustomerOrders(IgniteCache<Integer, Order> cache, 
                                                      Integer customerId) {
        // Use scan query to find orders for customer
        ScanQuery<Integer, Order> query = new ScanQuery<>(
            (key, order) -> order.getCustomerId().equals(customerId)
        );
        
        return cache.query(query).getAll().stream()
            .map(entry -> entry.getValue())
            .collect(Collectors.toList());
    }
}
```

### SQL Queries over In-Memory Data

#### 1. SQL Schema and Table Creation

```java
public class SQLQueriesExample {
    public static void setupSQLSchema() {
        Ignite ignite = Ignition.start();
        
        // Enable SQL for cache
        CacheConfiguration<PersonKey, Person> personCacheConfig = new CacheConfiguration<>();
        personCacheConfig.setName("person_cache");
        personCacheConfig.setIndexedTypes(PersonKey.class, Person.class);
        personCacheConfig.setSqlSchema("PUBLIC");
        
        IgniteCache<PersonKey, Person> personCache = ignite.getOrCreateCache(personCacheConfig);
        
        // Insert sample data
        for (int i = 1; i <= 1000; i++) {
            PersonKey key = new PersonKey(i);
            Person person = new Person(i, "Person" + i, i % 50 + 20, "Department" + (i % 10));
            personCache.put(key, person);
        }
        
        performSQLQueries(ignite);
    }
    
    private static void performSQLQueries(Ignite ignite) {
        // SQL Query examples
        
        // 1. Simple SELECT
        SqlFieldsQuery simpleQuery = new SqlFieldsQuery(
            "SELECT name, age, department FROM Person WHERE age > ? ORDER BY age"
        );
        simpleQuery.setArgs(30);
        
        try (QueryCursor<List<?>> cursor = ignite.cache("person_cache").query(simpleQuery)) {
            System.out.println("People older than 30:");
            for (List<?> row : cursor) {
                System.out.printf("Name: %s, Age: %d, Department: %s%n", 
                    row.get(0), row.get(1), row.get(2));
            }
        }
        
        // 2. Aggregation query
        SqlFieldsQuery aggregateQuery = new SqlFieldsQuery(
            "SELECT department, COUNT(*), AVG(age) FROM Person GROUP BY department"
        );
        
        try (QueryCursor<List<?>> cursor = ignite.cache("person_cache").query(aggregateQuery)) {
            System.out.println("\nDepartment statistics:");
            for (List<?> row : cursor) {
                System.out.printf("Department: %s, Count: %d, Avg Age: %.2f%n",
                    row.get(0), row.get(1), row.get(2));
            }
        }
        
        // 3. JOIN query (requires multiple caches)
        setupDepartmentCache(ignite);
        
        SqlFieldsQuery joinQuery = new SqlFieldsQuery(
            "SELECT p.name, p.age, d.manager " +
            "FROM Person p " +
            "JOIN Department d ON p.department = d.name " +
            "WHERE p.age > 35"
        );
        
        try (QueryCursor<List<?>> cursor = ignite.cache("person_cache").query(joinQuery)) {
            System.out.println("\nJoin query results:");
            for (List<?> row : cursor) {
                System.out.printf("Employee: %s, Age: %d, Manager: %s%n",
                    row.get(0), row.get(1), row.get(2));
            }
        }
    }
    
    private static void setupDepartmentCache(Ignite ignite) {
        CacheConfiguration<String, Department> deptConfig = new CacheConfiguration<>();
        deptConfig.setName("department_cache");
        deptConfig.setIndexedTypes(String.class, Department.class);
        deptConfig.setSqlSchema("PUBLIC");
        
        IgniteCache<String, Department> deptCache = ignite.getOrCreateCache(deptConfig);
        
        // Insert department data
        for (int i = 0; i < 10; i++) {
            String deptName = "Department" + i;
            Department dept = new Department(deptName, "Manager" + i, 10 + i);
            deptCache.put(deptName, dept);
        }
    }
}

// Data model classes with SQL annotations
class PersonKey {
    @AffinityKeyMapped
    private int id;
    
    public PersonKey(int id) { this.id = id; }
    // getters, setters, equals, hashCode
}

class Person {
    @QuerySqlField(index = true)
    private int id;
    
    @QuerySqlField
    private String name;
    
    @QuerySqlField(index = true)
    private int age;
    
    @QuerySqlField(index = true)
    private String department;
    
    public Person(int id, String name, int age, String department) {
        this.id = id;
        this.name = name;
        this.age = age;
        this.department = department;
    }
    
    // getters and setters
}

class Department {
    @QuerySqlField
    private String name;
    
    @QuerySqlField
    private String manager;
    
    @QuerySqlField
    private int budget;
    
    public Department(String name, String manager, int budget) {
        this.name = name;
        this.manager = manager;
        this.budget = budget;
    }
    
    // getters and setters
}
```

#### 2. Advanced SQL Features

```java
public class AdvancedSQLFeatures {
    public static void demonstrateAdvancedSQL() {
        Ignite ignite = Ignition.start();
        
        // 1. Custom indexes
        createCustomIndexes(ignite);
        
        // 2. Distributed JOINs
        performDistributedJoins(ignite);
        
        // 3. Streaming and continuous queries
        setupContinuousQueries(ignite);
        
        // 4. DML operations
        performDMLOperations(ignite);
    }
    
    private static void createCustomIndexes(Ignite ignite) {
        // Create composite index
        String createIndexSQL = 
            "CREATE INDEX person_age_dept_idx ON Person (age, department)";
        
        ignite.cache("person_cache").query(new SqlFieldsQuery(createIndexSQL));
        
        // Query using the index
        SqlFieldsQuery indexedQuery = new SqlFieldsQuery(
            "SELECT * FROM Person WHERE age BETWEEN ? AND ? AND department = ?"
        );
        indexedQuery.setArgs(25, 35, "Department1");
        
        // This query will use the composite index for efficient execution
    }
    
    private static void performDistributedJoins(Ignite ignite) {
        // Enable distributed joins for complex queries
        SqlFieldsQuery distributedJoinQuery = new SqlFieldsQuery(
            "SELECT p.name, p.age, d.manager, s.amount " +
            "FROM Person p " +
            "JOIN Department d ON p.department = d.name " +
            "JOIN Salary s ON p.id = s.personId " +
            "WHERE p.age > 30 AND s.amount > 50000"
        );
        distributedJoinQuery.setDistributedJoins(true);
        
        try (QueryCursor<List<?>> cursor = ignite.cache("person_cache").query(distributedJoinQuery)) {
            // Process results
        }
    }
    
    private static void setupContinuousQueries(Ignite ignite) {
        IgniteCache<PersonKey, Person> cache = ignite.cache("person_cache");
        
        // Continuous query for real-time monitoring
        ContinuousQuery<PersonKey, Person> continuousQuery = new ContinuousQuery<>();
        
        // Filter: only notify about senior employees
        continuousQuery.setRemoteFilterFactory(() -> (key, oldVal, newVal) -> {
            return newVal != null && newVal.getAge() > 45;
        });
        
        // Listener: handle events
        continuousQuery.setLocalListener((events) -> {
            for (CacheEntryEvent<? extends PersonKey, ? extends Person> event : events) {
                System.out.println("Senior employee update: " + event.getValue().getName());
            }
        });
        
        cache.query(continuousQuery);
    }
    
    private static void performDMLOperations(Ignite ignite) {
        // INSERT
        SqlFieldsQuery insertQuery = new SqlFieldsQuery(
            "INSERT INTO Person (id, name, age, department) VALUES (?, ?, ?, ?)"
        );
        ignite.cache("person_cache").query(insertQuery.setArgs(2001, "John Doe", 30, "IT"));
        
        // UPDATE
        SqlFieldsQuery updateQuery = new SqlFieldsQuery(
            "UPDATE Person SET age = age + 1 WHERE department = ?"
        );
        ignite.cache("person_cache").query(updateQuery.setArgs("IT"));
        
        // DELETE
        SqlFieldsQuery deleteQuery = new SqlFieldsQuery(
            "DELETE FROM Person WHERE age > ?"
        );
        ignite.cache("person_cache").query(deleteQuery.setArgs(65));
    }
}
```

### Compute Grid Applications

#### 1. Real-Time Analytics Pipeline

```java
public class RealTimeAnalyticsExample {
    public static void setupAnalyticsPipeline() {
        Ignite ignite = Ignition.start();
        
        // 1. Setup streaming data ingestion
        IgniteDataStreamer<String, Event> streamer = ignite.dataStreamer("events");
        streamer.allowOverwrite(true);
        streamer.receiver(StreamTransformer.from((key, val) -> {
            // Real-time data transformation
            Event event = (Event) val;
            event.setProcessedTimestamp(System.currentTimeMillis());
            return event;
        }));
        
        // 2. Simulate real-time data
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
        scheduler.scheduleAtFixedRate(() -> {
            // Generate sample events
            for (int i = 0; i < 100; i++) {
                String eventId = UUID.randomUUID().toString();
                Event event = new Event(eventId, "user" + (i % 50), 
                    "action" + (i % 10), System.currentTimeMillis());
                streamer.addData(eventId, event);
            }
        }, 0, 1, TimeUnit.SECONDS);
        
        // 3. Real-time aggregation using compute grid
        scheduler.scheduleAtFixedRate(() -> {
            performRealTimeAnalytics(ignite);
        }, 5, 5, TimeUnit.SECONDS);
    }
    
    private static void performRealTimeAnalytics(Ignite ignite) {
        IgniteCompute compute = ignite.compute();
        
        // Distributed analytics computation
        Map<String, Long> userActivityCount = compute.execute(
            new UserActivityAnalysisTask(), 
            System.currentTimeMillis() - 60000 // Last minute
        );
        
        System.out.println("User activity in last minute: " + userActivityCount);
        
        // Store results in analytics cache
        IgniteCache<String, AnalyticsResult> analyticsCache = 
            ignite.getOrCreateCache("analytics_results");
        
        userActivityCount.forEach((user, count) -> {
            AnalyticsResult result = new AnalyticsResult(user, count, System.currentTimeMillis());
            analyticsCache.put(user + "_" + System.currentTimeMillis(), result);
        });
    }
}

class UserActivityAnalysisTask extends ComputeTaskAdapter<Long, Map<String, Long>> {
    @Override
    public Map<? extends ComputeJob, ClusterNode> map(List<ClusterNode> nodes, Long since) {
        Map<ComputeJob, ClusterNode> jobs = new HashMap<>();
        
        // Create job for each node to analyze local data
        for (ClusterNode node : nodes) {
            jobs.put(new LocalAnalysisJob(since), node);
        }
        
        return jobs;
    }
    
    @Override
    public Map<String, Long> reduce(List<ComputeJobResult> results) {
        Map<String, Long> aggregated = new HashMap<>();
        
        for (ComputeJobResult result : results) {
            Map<String, Long> nodeResult = result.getData();
            nodeResult.forEach((user, count) -> 
                aggregated.merge(user, count, Long::sum));
        }
        
        return aggregated;
    }
}

class LocalAnalysisJob extends ComputeJobAdapter {
    private final Long since;
    
    public LocalAnalysisJob(Long since) {
        this.since = since;
    }
    
    @Override
    public Object execute() {
        // Analyze local events
        Ignite ignite = Ignition.localIgnite();
        IgniteCache<String, Event> cache = ignite.cache("events");
        
        Map<String, Long> userCounts = new HashMap<>();
        
        // Scan local entries
        for (Cache.Entry<String, Event> entry : cache.localEntries()) {
            Event event = entry.getValue();
            if (event.getTimestamp() >= since) {
                userCounts.merge(event.getUserId(), 1L, Long::sum);
            }
        }
        
        return userCounts;
    }
}
```

#### 2. Distributed Machine Learning

```java
public class DistributedMLExample {
    public static void distributedLinearRegression() {
        Ignite ignite = Ignition.start();
        
        // 1. Load training data distributed across cluster
        IgniteCache<Integer, TrainingData> trainingCache = 
            ignite.getOrCreateCache("training_data");
        
        loadTrainingData(trainingCache);
        
        // 2. Distributed gradient descent
        IgniteCompute compute = ignite.compute();
        
        // Initial weights
        double[] weights = {0.0, 0.0, 0.0}; // bias + 2 features
        double learningRate = 0.01;
        int epochs = 100;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Compute gradients on each node
            Collection<double[]> gradients = compute.apply(
                new GradientComputeJob(weights, learningRate),
                getDataPartitions(ignite, trainingCache)
            );
            
            // Aggregate gradients
            double[] avgGradient = aggregateGradients(gradients);
            
            // Update weights
            for (int i = 0; i < weights.length; i++) {
                weights[i] -= learningRate * avgGradient[i];
            }
            
            if (epoch % 10 == 0) {
                double loss = computeLoss(ignite, trainingCache, weights);
                System.out.printf("Epoch %d: Loss = %.6f%n", epoch, loss);
            }
        }
        
        System.out.println("Final weights: " + Arrays.toString(weights));
    }
    
    private static Collection<Collection<TrainingData>> getDataPartitions(
            Ignite ignite, IgniteCache<Integer, TrainingData> cache) {
        
        Map<ClusterNode, Collection<TrainingData>> partitions = new HashMap<>();
        
        // Group data by node
        for (Cache.Entry<Integer, TrainingData> entry : cache) {
            ClusterNode node = ignite.affinity("training_data").mapKeyToNode(entry.getKey());
            partitions.computeIfAbsent(node, k -> new ArrayList<>()).add(entry.getValue());
        }
        
        return partitions.values();
    }
    
    private static double[] aggregateGradients(Collection<double[]> gradients) {
        double[] result = new double[3];
        int count = 0;
        
        for (double[] gradient : gradients) {
            for (int i = 0; i < result.length; i++) {
                result[i] += gradient[i];
            }
            count++;
        }
        
        // Average the gradients
        for (int i = 0; i < result.length; i++) {
            result[i] /= count;
        }
        
        return result;
    }
}

class GradientComputeJob extends ComputeJobAdapter {
    private final double[] weights;
    private final double learningRate;
    
    public GradientComputeJob(double[] weights, double learningRate) {
        this.weights = weights.clone();
        this.learningRate = learningRate;
    }
    
    @Override
    public Object execute() {
        // Compute gradients for local data partition
        Collection<TrainingData> localData = argument(0);
        double[] gradients = new double[weights.length];
        
        for (TrainingData data : localData) {
            // Predict
            double prediction = weights[0] + // bias
                               weights[1] * data.getFeature1() +
                               weights[2] * data.getFeature2();
            
            // Error
            double error = prediction - data.getTarget();
            
            // Gradients
            gradients[0] += error; // bias gradient
            gradients[1] += error * data.getFeature1();
            gradients[2] += error * data.getFeature2();
        }
        
        // Normalize by local data size
        int dataSize = localData.size();
        for (int i = 0; i < gradients.length; i++) {
            gradients[i] /= dataSize;
        }
        
        return gradients;
    }
}
```

## Performance Optimization and Best Practices

### 1. Memory Management Best Practices

```java
public class PerformanceOptimization {
    public static void optimizeMemoryUsage() {
        IgniteConfiguration cfg = new IgniteConfiguration();
        
        // 1. Configure appropriate data regions
        DataRegionConfiguration defaultRegion = new DataRegionConfiguration();
        defaultRegion.setName("default");
        defaultRegion.setInitialSize(512 * 1024 * 1024); // 512MB
        defaultRegion.setMaxSize(2L * 1024 * 1024 * 1024); // 2GB
        defaultRegion.setPageEvictionMode(DataPageEvictionMode.RANDOM_2_LRU);
        
        // 2. Enable off-heap storage for large datasets
        DataRegionConfiguration offHeapRegion = new DataRegionConfiguration();
        offHeapRegion.setName("offheap");
        offHeapRegion.setMaxSize(8L * 1024 * 1024 * 1024); // 8GB off-heap
        offHeapRegion.setPersistenceEnabled(false);
        
        DataStorageConfiguration storageCfg = new DataStorageConfiguration();
        storageCfg.setDefaultDataRegionConfiguration(defaultRegion);
        storageCfg.setDataRegionConfigurations(offHeapRegion);
        
        // 3. Configure write-ahead logging for persistence
        storageCfg.setWalMode(WALMode.LOG_ONLY);
        storageCfg.setWalSegmentSize(128 * 1024 * 1024); // 128MB segments
        
        cfg.setDataStorageConfiguration(storageCfg);
        
        // 4. Optimize garbage collection
        System.setProperty("IGNITE_QUIET", "false");
        System.setProperty("IGNITE_PERFORMANCE_SUGGESTIONS_DISABLED", "false");
        
        Ignite ignite = Ignition.start(cfg);
        
        // 5. Monitor memory usage
        monitorMemoryUsage(ignite);
    }
    
    private static void monitorMemoryUsage(Ignite ignite) {
        DataRegionMetrics metrics = ignite.dataRegionMetrics("default");
        
        System.out.printf("Pages in memory: %d%n", metrics.getTotalAllocatedPages());
        System.out.printf("Memory usage: %.2f MB%n", 
            metrics.getTotalAllocatedSize() / (1024.0 * 1024.0));
        System.out.printf("Off-heap used: %.2f MB%n", 
            metrics.getOffHeapUsedSize() / (1024.0 * 1024.0));
    }
}
```

### 2. Query Performance Optimization

```java
public class QueryOptimization {
    public static void optimizeQueries() {
        Ignite ignite = Ignition.start();
        
        // 1. Use appropriate indexes
        createOptimalIndexes(ignite);
        
        // 2. Optimize SQL queries
        optimizeSQLQueries(ignite);
        
        // 3. Use scan queries for local processing
        useScanQueriesEfficiently(ignite);
    }
    
    private static void createOptimalIndexes(Ignite ignite) {
        // Composite indexes for multi-column WHERE clauses
        String compositeIndex = 
            "CREATE INDEX person_age_dept_salary_idx ON Person (age, department, salary)";
        
        // Partial indexes for filtered data
        String partialIndex = 
            "CREATE INDEX active_employees_idx ON Person (department) WHERE active = true";
        
        ignite.cache("person_cache").query(new SqlFieldsQuery(compositeIndex));
        ignite.cache("person_cache").query(new SqlFieldsQuery(partialIndex));
    }
    
    private static void optimizeSQLQueries(Ignite ignite) {
        // Good: Use indexed columns in WHERE clause
        SqlFieldsQuery optimizedQuery = new SqlFieldsQuery(
            "SELECT name, salary FROM Person " +
            "WHERE department = ? AND age BETWEEN ? AND ? " +
            "ORDER BY salary DESC LIMIT 10"
        );
        optimizedQuery.setArgs("Engineering", 25, 45);
        
        // Enable query parallelism for large datasets
        optimizedQuery.setDistributedJoins(true);
        optimizedQuery.setEnforceJoinOrder(true);
        
        // Use lazy execution for large result sets
        optimizedQuery.setLazy(true);
        
        try (QueryCursor<List<?>> cursor = ignite.cache("person_cache").query(optimizedQuery)) {
            cursor.forEach(row -> {
                // Process results lazily
                System.out.println("Employee: " + row.get(0) + ", Salary: " + row.get(1));
            });
        }
    }
    
    private static void useScanQueriesEfficiently(Ignite ignite) {
        IgniteCache<PersonKey, Person> cache = ignite.cache("person_cache");
        
        // Scan query with predicate - runs on each node locally
        ScanQuery<PersonKey, Person> scanQuery = new ScanQuery<>((key, person) -> {
            // This predicate runs locally on each node
            return person.getAge() > 30 && person.getDepartment().equals("Engineering");
        });
        
        // Set page size for memory efficiency
        scanQuery.setPageSize(1000);
        
        try (QueryCursor<Cache.Entry<PersonKey, Person>> cursor = cache.query(scanQuery)) {
            cursor.forEach(entry -> {
                // Process local results
                System.out.println("Found: " + entry.getValue().getName());
            });
        }
    }
}
```

### 3. Cluster Configuration Best Practices

```java
public class ClusterOptimization {
    public static void configureProductionCluster() {
        IgniteConfiguration cfg = new IgniteConfiguration();
        
        // 1. Network and discovery configuration
        TcpDiscoverySpi discoverySpi = new TcpDiscoverySpi();
        
        // Static IP finder for production
        TcpDiscoveryVmIpFinder ipFinder = new TcpDiscoveryVmIpFinder();
        ipFinder.setAddresses(Arrays.asList(
            "192.168.1.100:47500..47509",
            "192.168.1.101:47500..47509",
            "192.168.1.102:47500..47509"
        ));
        discoverySpi.setIpFinder(ipFinder);
        
        // Configure failure detection
        discoverySpi.setSocketTimeout(5000);
        discoverySpi.setAckTimeout(5000);
        discoverySpi.setNetworkTimeout(5000);
        
        cfg.setDiscoverySpi(discoverySpi);
        
        // 2. Communication SPI optimization
        TcpCommunicationSpi commSpi = new TcpCommunicationSpi();
        commSpi.setLocalPort(47100);
        commSpi.setLocalPortRange(10);
        commSpi.setMessageQueueLimit(1024);
        commSpi.setSocketSendBuffer(256 * 1024);
        commSpi.setSocketReceiveBuffer(256 * 1024);
        
        cfg.setCommunicationSpi(commSpi);
        
        // 3. Baseline topology for persistence
        cfg.setConsistentId("node-" + getLocalNodeId());
        cfg.setActiveOnStart(false); // Manual activation required
        
        // 4. Configure thread pools
        cfg.setPublicThreadPoolSize(Runtime.getRuntime().availableProcessors() * 2);
        cfg.setSystemThreadPoolSize(Runtime.getRuntime().availableProcessors());
        cfg.setQueryThreadPoolSize(Runtime.getRuntime().availableProcessors());
        
        // 5. Metrics and monitoring
        cfg.setMetricsLogFrequency(60000); // Log metrics every minute
        cfg.setMetricsUpdateFrequency(5000); // Update metrics every 5 seconds
        
        Ignite ignite = Ignition.start(cfg);
        
        // Activate cluster manually in production
        activateCluster(ignite);
    }
    
    private static void activateCluster(Ignite ignite) {
        // Wait for all baseline nodes to join
        ignite.cluster().active(true);
        
        // Set baseline topology
        Collection<ClusterNode> baselineNodes = ignite.cluster().forServers().nodes();
        ignite.cluster().setBaselineTopology(baselineNodes);
        
        System.out.println("Cluster activated with " + baselineNodes.size() + " baseline nodes");
    }
}
```

## Real-World Use Cases and Examples

### 1. Financial Trading System

```java
public class TradingSystemExample {
    public static void implementTradingSystem() {
        Ignite ignite = Ignition.start();
        
        // 1. Market data cache with expiry
        CacheConfiguration<String, MarketData> marketDataConfig = new CacheConfiguration<>();
        marketDataConfig.setName("market_data");
        marketDataConfig.setExpiryPolicyFactory(CreatedExpiryPolicy.factoryOf(Duration.ONE_MINUTE));
        marketDataConfig.setCacheMode(CacheMode.REPLICATED); // Low latency reads
        
        IgniteCache<String, MarketData> marketDataCache = 
            ignite.getOrCreateCache(marketDataConfig);
        
        // 2. Orders cache with persistence
        CacheConfiguration<String, Order> ordersConfig = new CacheConfiguration<>();
        ordersConfig.setName("orders");
        ordersConfig.setCacheMode(CacheMode.PARTITIONED);
        ordersConfig.setBackups(2); // High availability
        ordersConfig.setDataRegionName("persistent_region");
        
        IgniteCache<String, Order> ordersCache = ignite.getOrCreateCache(ordersConfig);
        
        // 3. Real-time risk calculation
        setupRiskCalculation(ignite, ordersCache, marketDataCache);
        
        // 4. Order matching engine
        setupOrderMatching(ignite, ordersCache);
    }
    
    private static void setupRiskCalculation(Ignite ignite, 
                                           IgniteCache<String, Order> ordersCache,
                                           IgniteCache<String, MarketData> marketDataCache) {
        
        // Continuous query for real-time risk monitoring
        ContinuousQuery<String, Order> riskQuery = new ContinuousQuery<>();
        
        riskQuery.setLocalListener((events) -> {
            for (CacheEntryEvent<? extends String, ? extends Order> event : events) {
                Order order = event.getValue();
                
                // Calculate position risk in real-time
                IgniteCompute compute = ignite.compute();
                Double risk = compute.call(() -> {
                    MarketData marketData = marketDataCache.get(order.getSymbol());
                    return calculateRisk(order, marketData);
                });
                
                if (risk > 100000) { // Risk threshold
                    System.out.println("HIGH RISK ORDER: " + order.getId() + 
                                     ", Risk: $" + risk);
                    // Trigger risk management actions
                }
            }
        });
        
        ordersCache.query(riskQuery);
    }
    
    private static Double calculateRisk(Order order, MarketData marketData) {
        if (marketData == null) return 0.0;
        
        double notional = order.getQuantity() * marketData.getPrice();
        double volatility = marketData.getVolatility();
        double timeToExpiry = order.getTimeToExpiry();
        
        // Simplified VaR calculation
        return notional * volatility * Math.sqrt(timeToExpiry) * 2.33; // 99% confidence
    }
}
```

### 2. IoT Data Processing Platform

```java
public class IoTDataProcessingExample {
    public static void setupIoTPlatform() {
        Ignite ignite = Ignition.start();
        
        // 1. Setup streaming for IoT device data
        IgniteDataStreamer<String, SensorReading> streamer = 
            ignite.dataStreamer("sensor_readings");
        
        streamer.allowOverwrite(true);
        streamer.receiver(StreamTransformer.from((key, val) -> {
            SensorReading reading = (SensorReading) val;
            
            // Real-time anomaly detection
            if (isAnomalous(reading)) {
                // Store in alerts cache
                ignite.cache("alerts").put(
                    reading.getDeviceId() + "_" + System.currentTimeMillis(),
                    new Alert(reading.getDeviceId(), "Anomalous reading", reading.getValue())
                );
            }
            
            return reading;
        }));
        
        // 2. Simulate IoT device data
        ScheduledExecutorService deviceSimulator = Executors.newScheduledThreadPool(10);
        
        for (int deviceId = 1; deviceId <= 1000; deviceId++) {
            final int id = deviceId;
            deviceSimulator.scheduleAtFixedRate(() -> {
                SensorReading reading = generateSensorReading(id);
                streamer.addData("device_" + id, reading);
            }, 0, 1, TimeUnit.SECONDS);
        }
        
        // 3. Real-time aggregation
        setupRealTimeAggregation(ignite);
        
        // 4. Predictive maintenance
        setupPredictiveMaintenance(ignite);
    }
    
    private static void setupRealTimeAggregation(Ignite ignite) {
        // Compute average sensor readings every 30 seconds
        ScheduledExecutorService aggregator = Executors.newScheduledThreadPool(1);
        
        aggregator.scheduleAtFixedRate(() -> {
            IgniteCompute compute = ignite.compute();
            
            // Distributed aggregation across all nodes
            Map<String, Double> deviceAverages = compute.execute(
                new SensorAggregationTask(),
                System.currentTimeMillis() - 30000 // Last 30 seconds
            );
            
            // Store aggregated results
            IgniteCache<String, DeviceStats> statsCache = ignite.cache("device_stats");
            deviceAverages.forEach((deviceId, avgValue) -> {
                DeviceStats stats = new DeviceStats(deviceId, avgValue, System.currentTimeMillis());
                statsCache.put(deviceId + "_" + System.currentTimeMillis(), stats);
            });
            
            System.out.println("Processed averages for " + deviceAverages.size() + " devices");
            
        }, 30, 30, TimeUnit.SECONDS);
    }
}
```

## Learning Objectives

By the end of this comprehensive study, you should be able to:

### Technical Mastery
- **Design and implement** distributed in-memory data grids using Apache Ignite
- **Configure optimal cluster topologies** for different use cases (OLTP, OLAP, hybrid)
- **Write efficient SQL queries** that leverage distributed execution and indexing
- **Implement compute grid applications** for parallel and distributed processing
- **Optimize memory management** and configure appropriate data regions
- **Handle cluster lifecycle management** including node discovery, baseline topology, and scaling

### Architectural Understanding
- **Explain the differences** between traditional databases and in-memory data grids
- **Design data models** that optimize for both performance and consistency
- **Implement proper data partitioning** strategies for load distribution
- **Configure replication and backup** strategies for high availability
- **Choose appropriate consistency models** (strong vs eventual consistency)

### Performance and Operations
- **Monitor and tune** cluster performance using metrics and profiling tools
- **Implement proper error handling** and failure recovery mechanisms
- **Design for horizontal scalability** and elastic cluster growth
- **Optimize network configuration** for low-latency operations
- **Implement security measures** including authentication, authorization, and encryption

### Self-Assessment Checklist

Before proceeding, ensure you can:

□ Set up a multi-node Ignite cluster with proper discovery configuration  
□ Design and implement a cache configuration with appropriate partitioning  
□ Write complex SQL queries with JOINs across distributed caches  
□ Implement a compute task that processes data across multiple nodes  
□ Configure memory regions and optimize for your use case  
□ Set up continuous queries for real-time data processing  
□ Implement proper error handling and cluster failure recovery  
□ Monitor cluster health and performance using built-in metrics  
□ Design a solution for a real-world distributed computing problem  
□ Explain trade-offs between consistency, availability, and partition tolerance  

## Practical Exercises

### Exercise 1: E-commerce Inventory System
**Objective**: Build a distributed inventory management system

```java
// TODO: Complete this implementation
public class InventorySystem {
    // Requirements:
    // 1. Product catalog cache with SQL queries
    // 2. Real-time inventory updates
    // 3. Distributed order processing
    // 4. Low-stock alerts using continuous queries
    // 5. Sales analytics using compute grid
    
    public void setupInventorySystem() {
        // Your implementation here
    }
}
```

### Exercise 2: Real-time Analytics Dashboard
**Objective**: Create a system for real-time web analytics

```java
// TODO: Implement a real-time analytics system
public class WebAnalytics {
    // Requirements:
    // 1. Ingest clickstream data using data streamers
    // 2. Real-time user session tracking
    // 3. Distributed computation for funnel analysis
    // 4. SQL queries for business intelligence
    // 5. Geographic analysis of user behavior
    
    public void buildAnalyticsPlatform() {
        // Your implementation here
    }
}
```

### Exercise 3: Financial Risk Management
**Objective**: Build a real-time risk calculation system

```java
// TODO: Create a risk management system
public class RiskManagement {
    // Requirements:
    // 1. Real-time portfolio valuation
    // 2. VaR (Value at Risk) calculations using compute grid
    // 3. Market data streaming and processing
    // 4. Stress testing using distributed computing
    // 5. Risk alerts and compliance monitoring
    
    public void implementRiskSystem() {
        // Your implementation here
    }
}
```

## Study Materials and Resources

### Primary Resources
- **Official Documentation**: [Apache Ignite Documentation](https://ignite.apache.org/docs/latest/)
- **Architecture Guide**: [Ignite Architecture Overview](https://ignite.apache.org/arch/multi-tier-storage.html)
- **Performance Guide**: [Performance and Tuning](https://ignite.apache.org/docs/latest/perf-and-troubleshooting/general-perf-tips)

### Books and Deep Learning
- **"Apache Ignite in Action"** - Comprehensive guide to Ignite features
- **"Designing Data-Intensive Applications"** by Martin Kleppmann - Chapter on distributed data systems
- **"High Performance Spark"** - Complementary knowledge for big data processing

### Video Resources
- **GridGain/Apache Ignite YouTube Channel** - Official tutorials and webinars
- **"In-Memory Computing with Apache Ignite"** - Conference presentations
- **"Building Scalable Data Platforms"** - Architecture patterns

### Hands-on Labs and Tutorials
- **GridGain University** - Free online courses
- **Apache Ignite Examples** - GitHub repository with sample projects
- **Docker Compose Labs** - Containerized cluster setups

### Community and Support
- **Apache Ignite User Mailing List** - Technical discussions
- **Stack Overflow** - Tagged questions and answers
- **GitHub Issues** - Bug reports and feature requests

### Development Setup

**Prerequisites:**
```bash
# Java 8 or higher
java -version

# Maven or Gradle for dependency management
mvn --version

# Docker for cluster testing
docker --version
```

**Maven Dependencies:**
```xml
<dependencies>
    <dependency>
        <groupId>org.apache.ignite</groupId>
        <artifactId>ignite-core</artifactId>
        <version>2.15.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.ignite</groupId>
        <artifactId>ignite-indexing</artifactId>
        <version>2.15.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.ignite</groupId>
        <artifactId>ignite-spring</artifactId>
        <version>2.15.0</version>
    </dependency>
</dependencies>
```

**Docker Cluster Setup:**
```yaml
# docker-compose.yml for testing
version: '3.8'
services:
  ignite-node1:
    image: apacheignite/ignite:2.15.0
    environment:
      - OPTION_LIBS=ignite-indexing,ignite-calcite
      - CONFIG_URI=https://raw.githubusercontent.com/apache/ignite/master/examples/config/example-cache.xml
    ports:
      - "10800:10800"
      - "47100:47100"
      - "47500:47500"
  
  ignite-node2:
    image: apacheignite/ignite:2.15.0
    environment:
      - OPTION_LIBS=ignite-indexing,ignite-calcite
      - CONFIG_URI=https://raw.githubusercontent.com/apache/ignite/master/examples/config/example-cache.xml
```

### Performance Testing Tools

**Benchmark Applications:**
```bash
# Yardstick benchmarks for Ignite
git clone https://github.com/gridgain/yardstick-ignite.git
cd yardstick-ignite
./bin/benchmark-run.sh config/benchmark.properties

# JMeter for load testing
# Custom Ignite samplers for testing distributed operations
```

**Monitoring and Profiling:**
```java
// JVM monitoring
-XX:+UseG1GC
-XX:+PrintGCDetails
-XX:+PrintGCTimeStamps
-Xloggc:gc.log

// Ignite-specific monitoring
-DIGNITE_QUIET=false
-DIGNITE_PERFORMANCE_SUGGESTIONS_DISABLED=false
-DIGNITE_UPDATE_NOTIFIER=false
```

## Next Steps

After mastering Apache Ignite fundamentals:

1. **Advanced Topics**:
   - Custom cache stores and write-through/write-behind patterns
   - Integration with Apache Kafka for event streaming
   - Machine learning with Apache Ignite ML
   - Integration with Apache Spark and Hadoop

2. **Production Deployment**:
   - Kubernetes deployment strategies
   - Monitoring with Prometheus and Grafana
   - Security configuration and SSL/TLS setup
   - Backup and disaster recovery planning

   
