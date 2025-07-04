# In-Memory Databases for Analytics

*Duration: 2 weeks*

## Overview

In-memory databases for analytics represent a paradigm shift from traditional disk-based storage systems, offering unprecedented performance for real-time analytics and business intelligence workloads. This module covers the fundamental concepts, technologies, and implementation patterns for building high-performance analytical systems.

## Learning Objectives

By the end of this section, you should be able to:
- **Understand columnar storage principles** and their advantages for analytical workloads
- **Design and implement** SAP HANA and similar in-memory analytical solutions
- **Build real-time analytics pipelines** using in-memory technologies
- **Optimize query performance** for large-scale analytical workloads
- **Implement hybrid transactional/analytical processing (HTAP)** systems
- **Choose appropriate technologies** for different analytical use cases

## Core Concepts

### What are In-Memory Databases for Analytics?

In-memory analytical databases store data primarily in RAM rather than on disk, enabling:
- **Sub-second query response times** for complex analytical queries
- **Real-time data processing** and analytics
- **Compressed columnar storage** for efficient memory utilization
- **Parallel processing** across multiple CPU cores
- **Direct integration** with business intelligence tools

### Traditional vs In-Memory Analytics Comparison

| Aspect | Traditional Disk-Based | In-Memory Analytics |
|--------|----------------------|-------------------|
| **Query Response** | Minutes to hours | Milliseconds to seconds |
| **Data Latency** | Batch processing (hourly/daily) | Real-time streaming |
| **Concurrency** | Limited concurrent users | Thousands of concurrent queries |
| **Data Volume** | Petabytes on disk | Terabytes in memory |
| **Cost** | Lower storage cost | Higher memory cost |
| **Complexity** | ETL pipelines required | Direct operational data access |

## Columnar In-Memory Storage

### Understanding Columnar Storage

Traditional row-based storage stores data records sequentially, while columnar storage organizes data by columns. This approach provides significant advantages for analytical workloads.

#### Row-Based vs Columnar Storage Visualization

```
Row-Based Storage (Traditional OLTP):
Record 1: [ID=1, Name="John", Age=25, Salary=50000, Dept="IT"]
Record 2: [ID=2, Name="Jane", Age=30, Salary=60000, Dept="HR"]
Record 3: [ID=3, Name="Bob",  Age=35, Salary=70000, Dept="IT"]

Columnar Storage (Analytics Optimized):
ID Column:     [1, 2, 3, ...]
Name Column:   ["John", "Jane", "Bob", ...]
Age Column:    [25, 30, 35, ...]
Salary Column: [50000, 60000, 70000, ...]
Dept Column:   ["IT", "HR", "IT", ...]
```

#### Key Advantages of Columnar Storage

**1. Compression Efficiency**
```sql
-- Example: Department column with repetitive values
-- Row storage: ["IT", "HR", "IT", "IT", "Finance", "IT", "HR", "IT"]
-- Columnar with dictionary encoding:
-- Dictionary: {1: "IT", 2: "HR", 3: "Finance"}
-- Encoded:    [1, 2, 1, 1, 3, 1, 2, 1]
-- Compression ratio: ~75% reduction in storage
```

**2. Query Performance for Analytics**
```sql
-- Analytical query: Average salary by department
SELECT Dept, AVG(Salary) 
FROM Employees 
GROUP BY Dept;

-- Columnar advantage: Only reads Dept and Salary columns
-- Row-based: Must read entire records (ID, Name, Age, Salary, Dept)
-- Performance improvement: 3-5x faster for selective column access
```

**3. Vectorized Processing**
```python
# Pseudo-code for vectorized operations in columnar storage
def calculate_bonus(salary_column):
    # Process entire column in single CPU instruction (SIMD)
    return salary_column * 0.1  # Apply to all values simultaneously

# Traditional row-by-row processing
def calculate_bonus_traditional(employees):
    bonuses = []
    for employee in employees:
        bonuses.append(employee.salary * 0.1)  # One operation per iteration
    return bonuses
```

### Implementation Example: Custom Columnar Store

```python
import numpy as np
from typing import Dict, List, Any
import pickle
import gzip

class ColumnStore:
    def __init__(self):
        self.columns: Dict[str, np.ndarray] = {}
        self.row_count = 0
        
    def add_column(self, name: str, data: List[Any], dtype=None):
        """Add a column to the store with optional compression"""
        column_array = np.array(data, dtype=dtype)
        
        # Apply compression based on data type
        if dtype == 'object':  # String data
            self.columns[name] = self._compress_string_column(column_array)
        else:
            self.columns[name] = column_array
            
        self.row_count = len(column_array)
    
    def _compress_string_column(self, column: np.ndarray):
        """Dictionary encoding for string columns"""
        unique_values = np.unique(column)
        value_to_code = {val: idx for idx, val in enumerate(unique_values)}
        
        # Store both dictionary and encoded values
        encoded = np.array([value_to_code[val] for val in column])
        return {
            'dictionary': unique_values,
            'encoded': encoded,
            'compression_ratio': len(column) / len(unique_values)
        }
    
    def select(self, columns: List[str], where_clause=None):
        """Efficient column selection with optional filtering"""
        result = {}
        
        # Only read requested columns (columnar advantage)
        for col_name in columns:
            if col_name in self.columns:
                column_data = self.columns[col_name]
                
                # Decompress if needed
                if isinstance(column_data, dict) and 'dictionary' in column_data:
                    # Decompress string column
                    dictionary = column_data['dictionary']
                    encoded = column_data['encoded']
                    result[col_name] = dictionary[encoded]
                else:
                    result[col_name] = column_data
        
        # Apply WHERE clause if provided
        if where_clause:
            mask = self._evaluate_where_clause(where_clause)
            for col_name in result:
                result[col_name] = result[col_name][mask]
        
        return result
    
    def aggregate(self, group_by: str, agg_column: str, operation: str):
        """Efficient grouping and aggregation"""
        group_col = self.columns[group_by]
        agg_col = self.columns[agg_column]
        
        # Vectorized groupby operation
        if operation == 'AVG':
            # Use pandas-like functionality for efficient grouping
            import pandas as pd
            df = pd.DataFrame({group_by: group_col, agg_column: agg_col})
            return df.groupby(group_by)[agg_column].mean().to_dict()
    
    def get_stats(self):
        """Get storage statistics"""
        stats = {
            'total_rows': self.row_count,
            'total_columns': len(self.columns),
            'memory_usage': {}
        }
        
        for col_name, col_data in self.columns.items():
            if isinstance(col_data, dict) and 'compression_ratio' in col_data:
                stats['memory_usage'][col_name] = {
                    'compressed_size': col_data['encoded'].nbytes,
                    'compression_ratio': col_data['compression_ratio']
                }
            else:
                stats['memory_usage'][col_name] = {
                    'size': col_data.nbytes,
                    'compression_ratio': 1.0
                }
        
        return stats

# Usage example
if __name__ == "__main__":
    # Create sample data
    store = ColumnStore()
    
    # Add columns with different data types
    store.add_column('employee_id', list(range(1, 10001)), dtype='int32')
    store.add_column('name', [f'Employee_{i}' for i in range(1, 10001)], dtype='object')
    store.add_column('department', ['IT', 'HR', 'Finance'] * 3334, dtype='object')
    store.add_column('salary', np.random.randint(40000, 120000, 10000), dtype='int32')
    store.add_column('age', np.random.randint(22, 65, 10000), dtype='int32')
    
    # Analytical query: Average salary by department
    print("Analytical Query Performance Test:")
    import time
    
    start_time = time.time()
    result = store.select(['department', 'salary'])
    avg_salary_by_dept = store.aggregate('department', 'salary', 'AVG')
    end_time = time.time()
    
    print(f"Query completed in {end_time - start_time:.4f} seconds")
    print("Average salary by department:", avg_salary_by_dept)
    
    # Show compression statistics
    print("\nStorage Statistics:")
    stats = store.get_stats()
    for col, info in stats['memory_usage'].items():
        if 'compression_ratio' in info:
            print(f"{col}: Compression ratio = {info['compression_ratio']:.2f}x")
```

### Advanced Columnar Techniques

#### 1. Zone Maps and Min/Max Pruning
```sql
-- Zone maps store min/max values for data segments
-- Enables query pruning without scanning data

-- Example: Sales data partitioned by date
CREATE COLUMN TABLE Sales (
    sale_date DATE,
    amount DECIMAL(10,2),
    product_id INT
) PARTITION BY RANGE(sale_date);

-- Query with date filter
SELECT SUM(amount) 
FROM Sales 
WHERE sale_date BETWEEN '2024-01-01' AND '2024-01-31';

-- Zone map optimization:
-- Partition 1: min_date='2024-01-01', max_date='2024-01-15' → INCLUDE
-- Partition 2: min_date='2024-01-16', max_date='2024-01-31' → INCLUDE  
-- Partition 3: min_date='2024-02-01', max_date='2024-02-15' → SKIP
```

#### 2. Bit-Vector Indexing
```python
class BitVectorIndex:
    def __init__(self, column_data: List[str]):
        self.unique_values = list(set(column_data))
        self.bit_vectors = {}
        
        # Create bit vector for each unique value
        for value in self.unique_values:
            self.bit_vectors[value] = [
                1 if x == value else 0 for x in column_data
            ]
    
    def query(self, conditions: List[str]):
        """Fast querying using bit operations"""
        result_bits = None
        
        for condition in conditions:
            if condition in self.bit_vectors:
                if result_bits is None:
                    result_bits = self.bit_vectors[condition].copy()
                else:
                    # AND operation for multiple conditions
                    result_bits = [a & b for a, b in 
                                 zip(result_bits, self.bit_vectors[condition])]
        
        # Return row indices where condition is true
        return [i for i, bit in enumerate(result_bits) if bit == 1]

# Example usage
departments = ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR']
index = BitVectorIndex(departments)

# Find all IT employees
it_employees = index.query(['IT'])
print(f"IT employees at positions: {it_employees}")  # [0, 2, 4]
```

## Practical Exercises and Projects

### Exercise 1: Build a Columnar Storage Engine

**Objective**: Implement a basic columnar storage engine with compression

```python
# Starter code - Complete the implementation
class ColumnarStorageEngine:
    def __init__(self):
        self.columns = {}
        
    def insert_batch(self, data_dict):
        """Insert a batch of records
        Args:
            data_dict: {'column_name': [values...]}
        """
        # TODO: Implement batch insertion with compression
        pass
    
    def select(self, columns, where_clause=None):
        """Select specific columns with optional filtering
        Args:
            columns: List of column names
            where_clause: Function that takes a row dict and returns bool
        """
        # TODO: Implement efficient column selection
        pass
    
    def aggregate(self, group_by_col, agg_col, operation):
        """Perform aggregation
        Args:
            group_by_col: Column to group by
            agg_col: Column to aggregate
            operation: 'SUM', 'AVG', 'COUNT', etc.
        """
        # TODO: Implement vectorized aggregation
        pass

# Test your implementation
test_data = {
    'customer_id': list(range(1, 10001)),
    'region': ['North', 'South', 'East', 'West'] * 2500,
    'amount': [100 + i % 500 for i in range(10000)]
}

storage = ColumnarStorageEngine()
# Add your test cases here
```

### Exercise 2: Real-Time Analytics Pipeline

**Objective**: Build a complete real-time analytics pipeline using Apache Kafka

```python
# TODO: Complete the real-time analytics system
from kafka import KafkaProducer, KafkaConsumer
import json
import time
from threading import Thread

class RealTimeAnalyticsSystem:
    def __init__(self):
        self.producer = None  # TODO: Initialize Kafka producer
        self.consumer = None  # TODO: Initialize Kafka consumer
        self.metrics = {}     # TODO: Design metrics storage structure
    
    def generate_events(self, num_events=1000):
        """Generate sample events for testing"""
        # TODO: Generate realistic sales events
        pass
    
    def process_events(self):
        """Process incoming events and update metrics"""
        # TODO: Implement real-time event processing
        pass
    
    def get_dashboard_data(self):
        """Return current analytics for dashboard"""
        # TODO: Return formatted dashboard data
        pass

# Implementation requirements:
# 1. Generate 1000 events per second
# 2. Calculate metrics with 1-minute windows
# 3. Detect anomalies (values > 3 standard deviations)
# 4. Provide REST API for dashboard queries
```

### Exercise 3: SAP HANA Performance Optimization

**Objective**: Optimize queries for large-scale analytical workloads

```sql
-- Given this slow query, optimize it for SAP HANA
-- Original query (taking 30+ seconds):

SELECT 
    c.customer_name,
    c.region,
    SUM(o.total_amount) as total_spent,
    COUNT(o.order_id) as order_count,
    AVG(o.total_amount) as avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2023-01-01'
  AND p.category = 'Electronics'
GROUP BY c.customer_name, c.region
HAVING SUM(o.total_amount) > 1000
ORDER BY total_spent DESC;

-- TODO: Optimize this query using:
-- 1. Column store techniques
-- 2. Proper indexing
-- 3. Query hints
-- 4. Materialized views
-- Target: Sub-second response time on 100M+ records
```

### Project 1: E-commerce Real-Time Analytics Dashboard

**Requirements**:
1. **Data Sources**: Customer actions, product views, purchases, inventory changes
2. **Real-Time Metrics**: 
   - Revenue per minute/hour/day
   - Conversion rates by traffic source
   - Product popularity trends
   - Inventory alerts
3. **Technologies**: Choose from SAP HANA, ClickHouse, or Apache Pinot
4. **Dashboard**: React-based with WebSocket updates
5. **Scale**: Handle 10,000 events per second

**Deliverables**:
- Architecture diagram
- Database schema design
- Real-time processing code
- Dashboard implementation
- Performance benchmarks

### Project 2: Financial Fraud Detection System

**Requirements**:
1. **Data Sources**: Transaction logs, user behavior, device fingerprints
2. **Analytics**: 
   - Pattern recognition using vector similarity
   - Real-time risk scoring
   - Anomaly detection
   - Customer segmentation
3. **Technologies**: Vector database + HTAP system
4. **Response Time**: <100ms for real-time decisions
5. **Accuracy**: >95% fraud detection rate, <2% false positives

**Deliverables**:
- ML model integration
- Real-time scoring pipeline
- Risk dashboard
- Performance metrics
- Compliance reporting

### Project 3: IoT Analytics Platform

**Requirements**:
1. **Data Sources**: Sensor data from 100,000 devices
2. **Analytics**:
   - Device health monitoring
   - Predictive maintenance
   - Usage pattern analysis
   - Geographic distribution insights
3. **Technologies**: Time-series optimized in-memory database
4. **Volume**: 1M data points per second
5. **Retention**: Real-time + 1 year historical

**Deliverables**:
- Time-series data model
- Streaming ingestion pipeline
- Alerting system
- Predictive analytics
- Operational dashboard

## Self-Assessment Checklist

Before proceeding, ensure you can:

□ **Explain columnar storage benefits** and implement compression techniques  
□ **Design SAP HANA schemas** for analytical workloads  
□ **Build real-time analytics pipelines** using stream processing  
□ **Implement HTAP systems** that serve both transactional and analytical queries  
□ **Optimize query performance** for sub-second response times  
□ **Create real-time dashboards** with live data updates  
□ **Handle high-velocity data streams** (>10K events/second)  
□ **Choose appropriate technologies** for different analytical use cases  
□ **Monitor and tune system performance** for large-scale deployments  
□ **Implement vector-based analytics** for recommendation systems  

### Advanced Challenges

1. **Build a distributed columnar database** that can scale across multiple nodes
2. **Implement automatic query optimization** using machine learning
3. **Create a unified SQL interface** that routes queries to appropriate engines
4. **Design a cost-based optimizer** for analytical query execution
5. **Build a real-time data lineage system** for governance and compliance

## Study Materials

### Essential Reading
- **"Designing Data-Intensive Applications"** by Martin Kleppmann - Chapters 3, 10
- **SAP HANA Developer Guide** - Official documentation
- **"Building Analytics Applications"** by Donald Farmer
- **Apache Pinot Documentation** - Real-time analytics patterns

### Video Resources
- **"Columnar Databases Deep Dive"** - Database internals series
- **"Real-Time Analytics at Scale"** - Netflix/Uber engineering talks
- **"SAP HANA Academy"** - Complete video series
- **"Stream Processing with Apache Kafka"** - Confluent workshops

### Hands-on Labs
- **SAP HANA Cloud Trial** - Free tier for learning
- **Apache Pinot Quick Start** - Local setup and examples  
- **ClickHouse Playground** - Online interactive environment
- **Apache Druid Tutorial** - Step-by-step implementation

### Research Papers
- **"C-Store: A Column-oriented DBMS"** - Foundational columnar storage paper
- **"The Design and Implementation of Modern Column-Oriented Database Systems"** - Comprehensive survey
- **"Apache Pinot: A Real-time Distributed OLAP Datastore"** - Architecture paper
- **"Amazon Redshift and the Case for Simpler Data Warehouses"** - Cloud analytics insights

### Practice Datasets
- **TPC-H Benchmark** - Standard analytical workload
- **NYC Taxi Dataset** - Time-series analytics practice
- **E-commerce Transaction Data** - Real-time analytics scenarios
- **IoT Sensor Data** - Stream processing practice

### Tools and Platforms
- **Development**: Docker, Kubernetes, Apache Kafka, Redis
- **Analytics**: SAP HANA Express, ClickHouse, Apache Pinot, Apache Druid
- **Monitoring**: Grafana, Prometheus, ELK Stack
- **Testing**: Apache JMeter, Artillery.js for load testing

### Performance Benchmarking
```bash
# Benchmark columnar vs row storage
sysbench --mysql-host=localhost --mysql-user=user --mysql-password=pass \
  --oltp-table-size=1000000 --oltp-read-only=on \
  --max-time=60 --max-requests=0 --num-threads=8 run

# Test real-time analytics performance
kafka-producer-perf-test --topic analytics-events \
  --num-records 1000000 --record-size 1024 \
  --throughput 10000 --producer-props bootstrap.servers=localhost:9092

# HANA query performance analysis
SELECT * FROM SYS.M_SQL_PLAN_STATISTICS 
WHERE STATEMENT_STRING LIKE '%your_query%'
ORDER BY TOTAL_EXECUTION_TIME DESC;
```

## Next Steps

After mastering in-memory analytics, consider exploring:

1. **[Distributed Query Engines](../08_Distributed_Query_Processing/01_Apache_Spark_SQL.md)** - Scale analytics across clusters
2. **[Data Lake Analytics](../09_Data_Lakes/01_Apache_Iceberg.md)** - Analytics on massive datasets
3. **[ML/AI Integration](../10_ML_AI_Integration/01_Feature_Stores.md)** - AI-powered analytics
4. **[Edge Analytics](../11_Edge_Computing/01_Edge_Analytics.md)** - Analytics at the edge

---

**Duration**: 2 weeks  
**Difficulty**: Intermediate to Advanced  
**Prerequisites**: Database fundamentals, SQL proficiency, distributed systems basics

## SAP HANA and Similar Technologies

### SAP HANA Architecture Overview

SAP HANA (High-Performance Analytic Appliance) is a multi-model, in-memory database that combines transactional and analytical processing in a single system.

#### Core Components Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SAP HANA System                         │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ HANA Studio │ │ Web Browser │ │ Business Apps       │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  SQL/MDX Interface Layer                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ SQL Processor │ MDX Engine │ Planning Engine        │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Engine Layer                                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Row Store   │ │ Column Store│ │ Text Engine         │   │
│  │ (OLTP)      │ │ (OLAP)      │ │ (Search)            │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Storage Layer                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ In-Memory Storage │ Persistence │ Recovery           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Hands-on SAP HANA Implementation

#### 1. Setting Up Column Store Tables

```sql
-- Create a column store table for analytics
CREATE COLUMN TABLE sales_analytics (
    transaction_id BIGINT PRIMARY KEY,
    customer_id INT NOT NULL,
    product_id INT NOT NULL,
    sale_date DATE NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    sales_rep_id INT,
    region VARCHAR(50),
    product_category VARCHAR(100)
) 
PARTITION BY RANGE(sale_date) (
    PARTITION p_2023 VALUES FROM ('2023-01-01') TO ('2024-01-01'),
    PARTITION p_2024 VALUES FROM ('2024-01-01') TO ('2025-01-01'),
    PARTITION p_current VALUES FROM ('2025-01-01') TO ('2026-01-01')
);

-- Create optimized indexes for analytical queries
CREATE INDEX idx_sales_date ON sales_analytics(sale_date);
CREATE INDEX idx_customer_region ON sales_analytics(customer_id, region);
CREATE INDEX idx_product_category ON sales_analytics(product_category, product_id);

-- Insert sample data
INSERT INTO sales_analytics VALUES
(1, 1001, 2001, '2024-01-15', 5, 29.99, 149.95, 101, 'North', 'Electronics'),
(2, 1002, 2002, '2024-01-16', 2, 59.99, 119.98, 102, 'South', 'Clothing'),
(3, 1003, 2001, '2024-01-17', 1, 29.99, 29.99, 101, 'East', 'Electronics');
```

#### 2. Advanced Analytical Queries

```sql
-- Complex analytical query with window functions
SELECT 
    product_category,
    region,
    sale_date,
    total_amount,
    -- Running total by category and region
    SUM(total_amount) OVER (
        PARTITION BY product_category, region 
        ORDER BY sale_date 
        ROWS UNBOUNDED PRECEDING
    ) as running_total,
    -- Rank products by sales within category
    RANK() OVER (
        PARTITION BY product_category 
        ORDER BY total_amount DESC
    ) as sales_rank,
    -- Moving average over last 30 days
    AVG(total_amount) OVER (
        PARTITION BY product_category 
        ORDER BY sale_date 
        RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW
    ) as moving_avg_30d
FROM sales_analytics
WHERE sale_date >= ADD_DAYS(CURRENT_DATE, -90)
ORDER BY product_category, region, sale_date;

-- Real-time dashboard query
SELECT 
    region,
    product_category,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_transaction_value,
    MAX(total_amount) as max_transaction,
    -- Time-based metrics
    SUM(CASE WHEN sale_date = CURRENT_DATE THEN total_amount ELSE 0 END) as today_revenue,
    SUM(CASE WHEN sale_date >= ADD_DAYS(CURRENT_DATE, -7) THEN total_amount ELSE 0 END) as week_revenue,
    -- Growth calculations
    (SUM(CASE WHEN sale_date >= ADD_DAYS(CURRENT_DATE, -7) THEN total_amount ELSE 0 END) - 
     SUM(CASE WHEN sale_date >= ADD_DAYS(CURRENT_DATE, -14) AND sale_date < ADD_DAYS(CURRENT_DATE, -7) THEN total_amount ELSE 0 END)) / 
     NULLIF(SUM(CASE WHEN sale_date >= ADD_DAYS(CURRENT_DATE, -14) AND sale_date < ADD_DAYS(CURRENT_DATE, -7) THEN total_amount ELSE 0 END), 0) * 100 
     as week_over_week_growth_pct
FROM sales_analytics
WHERE sale_date >= ADD_DAYS(CURRENT_DATE, -30)
GROUP BY region, product_category
ORDER BY total_revenue DESC;
```

#### 3. HANA-Specific Optimizations

```sql
-- Column compression and optimization
ALTER TABLE sales_analytics 
COMPRESS USING HANA_COMPRESSION;

-- Create calculation view for complex analytics
CREATE CALCULATION VIEW sales_kpi_view AS 
SELECT 
    customer_id,
    region,
    -- Customer lifetime value
    SUM(total_amount) as lifetime_value,
    COUNT(*) as transaction_count,
    AVG(total_amount) as avg_order_value,
    -- Customer segments
    CASE 
        WHEN SUM(total_amount) > 10000 THEN 'VIP'
        WHEN SUM(total_amount) > 5000 THEN 'Premium'
        WHEN SUM(total_amount) > 1000 THEN 'Standard'
        ELSE 'Basic'
    END as customer_segment,
    -- Recency, Frequency, Monetary analysis
    DAYS_BETWEEN(MAX(sale_date), CURRENT_DATE) as days_since_last_purchase,
    COUNT(DISTINCT MONTH(sale_date)) as active_months
FROM sales_analytics
GROUP BY customer_id, region;

-- Performance monitoring query
SELECT 
    statement_string,
    execution_time,
    records,
    cpu_time,
    memory_usage
FROM sys.m_sql_plan_statistics
WHERE execution_time > 1000  -- Queries taking more than 1 second
ORDER BY execution_time DESC;
```

### Similar Technologies Comparison

#### Apache Druid
```json
{
  "dataSource": "sales_analytics",
  "dimensionSpec": {
    "type": "default",
    "dimension": "product_category",
    "outputName": "category"
  },
  "aggregations": [
    {
      "type": "longSum",
      "name": "total_sales",
      "fieldName": "total_amount"
    },
    {
      "type": "count",
      "name": "transaction_count"
    }
  ],
  "filter": {
    "type": "interval",
    "dimension": "__time",
    "intervals": ["2024-01-01/2024-12-31"]
  },
  "queryType": "groupBy",
  "granularity": "day"
}
```

#### ClickHouse Implementation
```sql
-- ClickHouse table creation with optimal engine
CREATE TABLE sales_analytics_ch (
    transaction_id UInt64,
    customer_id UInt32,
    product_id UInt32,
    sale_date Date,
    quantity UInt16,
    unit_price Decimal(10,2),
    total_amount Decimal(12,2),
    sales_rep_id UInt32,
    region LowCardinality(String),  -- Optimized for repeated values
    product_category LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(sale_date)  -- Monthly partitions
ORDER BY (product_category, region, sale_date)  -- Clustering key
SETTINGS index_granularity = 8192;

-- Materialized view for real-time aggregations
CREATE MATERIALIZED VIEW sales_summary_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(sale_date)
ORDER BY (product_category, region, sale_date)
AS SELECT
    product_category,
    region,
    sale_date,
    sum(total_amount) as revenue,
    count() as transactions,
    uniq(customer_id) as unique_customers
FROM sales_analytics_ch
GROUP BY product_category, region, sale_date;
```

#### Amazon Redshift Spectrum
```sql
-- External table for data lake integration
CREATE EXTERNAL TABLE sales_data_lake (
    transaction_id bigint,
    customer_id int,
    product_id int,
    sale_date date,
    total_amount decimal(12,2),
    region varchar(50)
)
STORED AS PARQUET
LOCATION 's3://my-analytics-bucket/sales-data/'
TABLE PROPERTIES ('compression_type'='gzip');

-- Federated query combining Redshift and S3
SELECT 
    r.region,
    r.month,
    r.redshift_sales,
    dl.datalake_sales,
    (r.redshift_sales + dl.datalake_sales) as total_sales
FROM (
    SELECT region, 
           EXTRACT(month FROM sale_date) as month,
           SUM(total_amount) as redshift_sales
    FROM sales_analytics 
    GROUP BY region, EXTRACT(month FROM sale_date)
) r
JOIN (
    SELECT region,
           EXTRACT(month FROM sale_date) as month,
           SUM(total_amount) as datalake_sales
    FROM sales_data_lake
    GROUP BY region, EXTRACT(month FROM sale_date)
) dl ON r.region = dl.region AND r.month = dl.month;
```

## In-Memory Analytics Solutions

### Hybrid Transactional/Analytical Processing (HTAP)

HTAP systems combine transactional and analytical workloads in a single database, eliminating the need for separate OLTP and OLAP systems.

#### HTAP Architecture Patterns

```
Traditional Architecture (Separated):
┌─────────────┐    ETL    ┌─────────────┐    Query    ┌─────────────┐
│    OLTP     │ ────────→ │    OLAP     │ ──────────→ │  Analytics  │
│  Database   │           │ Data Warehouse│           │ Dashboard   │
│ (Real-time) │           │ (Batch-updated)│          │ (Historical)│
└─────────────┘           └─────────────┘           └─────────────┘

HTAP Architecture (Unified):
┌─────────────────────────────────────────────────────────────────┐
│                    HTAP Database                                │
│  ┌─────────────┐              ┌─────────────┐                  │
│  │    OLTP     │    Shared    │    OLAP     │                  │
│  │  Engine     │ ←──────────→ │   Engine    │                  │
│  │ (Row Store) │    Memory    │(Column Store)│                  │
│  └─────────────┘              └─────────────┘                  │
│         │                              │                       │
│         ▼                              ▼                       │
│  ┌─────────────┐              ┌─────────────┐                  │
│  │ Applications│              │  Analytics  │                  │
│  │ (Real-time) │              │ (Real-time) │                  │
│  └─────────────┘              └─────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

#### TiDB HTAP Implementation

```sql
-- TiDB: Hybrid row/column storage
-- Create table that supports both OLTP and OLAP workloads
CREATE TABLE customer_orders (
    order_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    order_date DATETIME NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    status VARCHAR(20) NOT NULL,
    region VARCHAR(50) NOT NULL,
    product_category VARCHAR(100),
    INDEX idx_customer (customer_id),
    INDEX idx_date_region (order_date, region)
);

-- Enable TiFlash (columnar storage) for analytical queries
ALTER TABLE customer_orders SET TIFLASH REPLICA 1;

-- OLTP Query (uses TiKV row storage)
SELECT * FROM customer_orders 
WHERE order_id = 12345;

-- Analytical Query (automatically uses TiFlash columnar storage)
SELECT 
    region,
    product_category,
    DATE(order_date) as order_date,
    COUNT(*) as order_count,
    SUM(total_amount) as daily_revenue
FROM customer_orders 
WHERE order_date >= '2024-01-01'
GROUP BY region, product_category, DATE(order_date)
ORDER BY daily_revenue DESC;

-- Real-time analytics on live transactional data
-- No ETL delay - queries run on current data
SELECT 
    customer_id,
    COUNT(*) as orders_today,
    SUM(total_amount) as spending_today
FROM customer_orders 
WHERE DATE(order_date) = CURDATE()
  AND status IN ('completed', 'processing')
GROUP BY customer_id
HAVING spending_today > 1000;
```

### Multi-Model Analytics Platforms

#### Apache Pinot for Real-Time Analytics

```javascript
// Pinot table configuration
{
  "tableName": "sales_analytics",
  "tableType": "REALTIME",
  "segmentsConfig": {
    "timeColumnName": "timestamp",
    "timeType": "MILLISECONDS",
    "retentionTimeUnit": "DAYS",
    "retentionTimeValue": "30"
  },
  "tenants": {
    "broker": "DefaultTenant",
    "server": "DefaultTenant"
  },
  "routing": {
    "instanceSelectorType": "strictReplicaGroup"
  },
  "metadata": {
    "customConfigs": {
      "realtime.segment.flush.threshold.rows": "50000",
      "realtime.segment.flush.threshold.time": "3600000"
    }
  }
}

// Schema definition
{
  "schemaName": "sales_analytics",
  "dimensionFieldSpecs": [
    {
      "name": "customer_id",
      "dataType": "INT"
    },
    {
      "name": "product_category",
      "dataType": "STRING"
    },
    {
      "name": "region",
      "dataType": "STRING"
    }
  ],
  "metricFieldSpecs": [
    {
      "name": "amount",
      "dataType": "DOUBLE"
    },
    {
      "name": "quantity",
      "dataType": "INT"
    }
  ],
  "timeFieldSpec": {
    "incomingGranularitySpec": {
      "name": "timestamp",
      "dataType": "LONG",
      "timeType": "MILLISECONDS"
    }
  }
}
```

```sql
-- Pinot SQL queries for real-time analytics
-- Sub-second response times on billions of events

-- Real-time customer segmentation
SELECT 
    CASE 
        WHEN total_spent > 10000 THEN 'VIP'
        WHEN total_spent > 5000 THEN 'Premium'
        WHEN total_spent > 1000 THEN 'Standard'
        ELSE 'Basic'
    END as customer_segment,
    COUNT(*) as customer_count,
    AVG(total_spent) as avg_spending
FROM (
    SELECT 
        customer_id,
        SUM(amount) as total_spent
    FROM sales_analytics
    WHERE timestamp >= fromDateTime('2024-01-01 00:00:00')
    GROUP BY customer_id
) customer_totals
GROUP BY customer_segment
ORDER BY customer_count DESC;

-- Real-time funnel analysis
SELECT 
    region,
    COUNT(CASE WHEN event_type = 'view' THEN 1 END) as views,
    COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_adds,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchases,
    -- Conversion rates
    COUNT(CASE WHEN event_type = 'cart' THEN 1 END) * 100.0 / 
        NULLIF(COUNT(CASE WHEN event_type = 'view' THEN 1 END), 0) as view_to_cart_rate,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) * 100.0 / 
        NULLIF(COUNT(CASE WHEN event_type = 'cart' THEN 1 END), 0) as cart_to_purchase_rate
FROM user_events
WHERE timestamp >= ago('PT1H')  -- Last 1 hour
GROUP BY region
ORDER BY purchases DESC;
```

### Vector Databases for AI-Powered Analytics

#### Building Recommendation Engine with Vector Analytics

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import redis
import json

class VectorAnalyticsEngine:
    def __init__(self, redis_host='localhost', vector_dim=512):
        self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
        self.vector_dim = vector_dim
        
        # Initialize FAISS index for similarity search
        self.faiss_index = faiss.IndexFlatIP(vector_dim)  # Inner product for cosine similarity
        self.item_mapping = {}  # item_id -> faiss_index_position
        
    def vectorize_products(self, products_data):
        """Convert product features to vectors"""
        # Combine product attributes into text
        product_texts = []
        for product in products_data:
            text = f"{product['name']} {product['category']} {product['description']} {' '.join(product.get('tags', []))}"
            product_texts.append(text)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=self.vector_dim, stop_words='english')
        vectors = vectorizer.fit_transform(product_texts).toarray()
        
        # Normalize for cosine similarity
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Add to FAISS index
        self.faiss_index.add(vectors.astype(np.float32))
        
        # Store mappings
        for idx, product in enumerate(products_data):
            self.item_mapping[product['id']] = idx
            
        return vectorizer, vectors
    
    def analyze_user_behavior(self, user_interactions):
        """Analyze user behavior using vector operations"""
        user_vectors = {}
        
        for user_id, interactions in user_interactions.items():
            # Create user preference vector based on interactions
            user_vector = np.zeros(self.vector_dim)
            total_weight = 0
            
            for interaction in interactions:
                item_id = interaction['item_id']
                weight = self._get_interaction_weight(interaction['type'])
                
                if item_id in self.item_mapping:
                    item_idx = self.item_mapping[item_id]
                    item_vector = self.faiss_index.reconstruct(item_idx)
                    user_vector += weight * item_vector
                    total_weight += weight
            
            if total_weight > 0:
                user_vector /= total_weight
                user_vectors[user_id] = user_vector
        
        return user_vectors
    
    def _get_interaction_weight(self, interaction_type):
        """Assign weights to different interaction types"""
        weights = {
            'view': 1.0,
            'cart': 3.0,
            'purchase': 5.0,
            'favorite': 4.0,
            'share': 2.0
        }
        return weights.get(interaction_type, 1.0)
    
    def get_recommendations(self, user_vector, k=10):
        """Get product recommendations using vector similarity"""
        # Reshape for FAISS query
        query_vector = user_vector.reshape(1, -1).astype(np.float32)
        
        # Search for similar items
        similarities, indices = self.faiss_index.search(query_vector, k)
        
        # Convert indices back to item IDs
        recommendations = []
        for idx, similarity in zip(indices[0], similarities[0]):
            item_id = next(item_id for item_id, mapped_idx in self.item_mapping.items() 
                          if mapped_idx == idx)
            recommendations.append({
                'item_id': item_id,
                'similarity_score': float(similarity)
            })
        
        return recommendations
    
    def real_time_recommendations(self, user_id, recent_interactions):
        """Generate real-time recommendations based on recent interactions"""
        # Update user vector with recent interactions
        if recent_interactions:
            recent_vector = np.zeros(self.vector_dim)
            total_weight = 0
            
            for interaction in recent_interactions:
                item_id = interaction['item_id']
                weight = self._get_interaction_weight(interaction['type'])
                
                if item_id in self.item_mapping:
                    item_idx = self.item_mapping[item_id]
                    item_vector = self.faiss_index.reconstruct(item_idx)
                    recent_vector += weight * item_vector
                    total_weight += weight
            
            if total_weight > 0:
                recent_vector /= total_weight
                
                # Get recommendations
                recommendations = self.get_recommendations(recent_vector)
                
                # Cache in Redis for fast retrieval
                self.redis_client.setex(
                    f'recommendations:{user_id}',
                    3600,  # 1 hour TTL
                    json.dumps(recommendations)
                )
                
                return recommendations
        
        # Fallback to cached recommendations
        cached = self.redis_client.get(f'recommendations:{user_id}')
        return json.loads(cached) if cached else []

# Advanced Analytics with Vector Operations
class VectorAnalytics:
    def __init__(self, vector_engine):
        self.engine = vector_engine
        
    def cluster_analysis(self, user_vectors, n_clusters=5):
        """Perform customer segmentation using vector clustering"""
        from sklearn.cluster import KMeans
        
        vectors = np.array(list(user_vectors.values()))
        user_ids = list(user_vectors.keys())
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Analyze clusters
        clusters = {}
        for user_id, label in zip(user_ids, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(user_id)
        
        # Calculate cluster characteristics
        cluster_analysis = {}
        for label, users in clusters.items():
            cluster_vector = kmeans.cluster_centers_[label]
            
            # Find representative products for this cluster
            recommendations = self.engine.get_recommendations(cluster_vector, k=5)
            
            cluster_analysis[label] = {
                'user_count': len(users),
                'representative_products': recommendations,
                'cluster_center': cluster_vector.tolist()
            }
        
        return cluster_analysis
    
    def similarity_trends(self, user_vectors, time_window_hours=24):
        """Analyze similarity trends over time"""
        current_time = time.time()
        
        # Calculate pairwise similarities
        vectors = np.array(list(user_vectors.values()))
        user_ids = list(user_vectors.keys())
        
        similarity_matrix = cosine_similarity(vectors)
        
        # Find trending similarities
        trending_pairs = []
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.8:  # High similarity threshold
                    trending_pairs.append({
                        'user1': user_ids[i],
                        'user2': user_ids[j],
                        'similarity': float(similarity)
                    })
        
        return sorted(trending_pairs, key=lambda x: x['similarity'], reverse=True)

# Usage example
if __name__ == "__main__":
    # Sample product data
    products = [
        {'id': 1, 'name': 'iPhone 15', 'category': 'Electronics', 'description': 'Smartphone with advanced camera', 'tags': ['mobile', 'apple', 'camera']},
        {'id': 2, 'name': 'MacBook Pro', 'category': 'Electronics', 'description': 'Professional laptop', 'tags': ['laptop', 'apple', 'professional']},
        {'id': 3, 'name': 'Nike Air Max', 'category': 'Fashion', 'description': 'Running shoes', 'tags': ['shoes', 'running', 'sports']}
    ]
    
    # Sample user interactions
    user_interactions = {
        'user1': [
            {'item_id': 1, 'type': 'view'},
            {'item_id': 2, 'type': 'cart'},
            {'item_id': 1, 'type': 'purchase'}
        ],
        'user2': [
            {'item_id': 3, 'type': 'view'},
            {'item_id': 3, 'type': 'purchase'}
        ]
    }
    
    # Initialize vector analytics
    engine = VectorAnalyticsEngine()
    vectorizer, product_vectors = engine.vectorize_products(products)
    user_vectors = engine.analyze_user_behavior(user_interactions)
    
    # Generate recommendations
    for user_id, user_vector in user_vectors.items():
        recommendations = engine.get_recommendations(user_vector, k=3)
        print(f"Recommendations for {user_id}: {recommendations}")
    
    # Perform cluster analysis
    analytics = VectorAnalytics(engine)
    clusters = analytics.cluster_analysis(user_vectors)
    print(f"Customer clusters: {clusters}")
```

### Enterprise Analytics Platforms

#### Building Custom Analytics Platform

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import asyncio
import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class QueryMetrics:
    query_id: str
    execution_time: float
    rows_processed: int
    memory_used: float
    cache_hit: bool

class EnterpriseAnalyticsPlatform:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = self._initialize_engines()
        self.cache = None
        self.query_executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        self.metrics_collector = []
        
    async def initialize_async_components(self):
        """Initialize async components like Redis cache"""
        self.cache = await aioredis.create_redis_pool(
            f"redis://{self.config['redis_host']}:{self.config['redis_port']}"
        )
        
    def _initialize_engines(self):
        """Initialize database engines for different data sources"""
        engines = {}
        
        for db_config in self.config['databases']:
            engine = create_engine(
                db_config['connection_string'],
                pool_size=db_config.get('pool_size', 20),
                max_overflow=db_config.get('max_overflow', 30),
                echo=db_config.get('debug', False)
            )
            engines[db_config['name']] = engine
            
        return engines
    
    async def execute_query(self, query: str, database: str, 
                           cache_ttl: Optional[int] = None) -> Dict[str, Any]:
        """Execute query with caching and performance monitoring"""
        query_id = f"{database}:{hash(query)}"
        start_time = time.time()
        
        # Check cache first
        if cache_ttl and self.cache:
            cached_result = await self.cache.get(f"query_cache:{query_id}")
            if cached_result:
                execution_time = time.time() - start_time
                self._record_metrics(QueryMetrics(
                    query_id=query_id,
                    execution_time=execution_time,
                    rows_processed=0,
                    memory_used=0,
                    cache_hit=True
                ))
                return eval(cached_result)  # In production, use proper serialization
        
        # Execute query
        engine = self.engines[database]
        with engine.connect() as conn:
            result = conn.execute(text(query))
            data = result.fetchall()
            columns = result.keys()
            
            # Convert to dictionary format
            result_dict = {
                'data': [dict(zip(columns, row)) for row in data],
                'columns': list(columns),
                'row_count': len(data)
            }
            
            # Cache result if TTL specified
            if cache_ttl and self.cache:
                await self.cache.setex(
                    f"query_cache:{query_id}",
                    cache_ttl,
                    str(result_dict)
                )
        
        execution_time = time.time() - start_time
        self._record_metrics(QueryMetrics(
            query_id=query_id,
            execution_time=execution_time,
            rows_processed=len(data),
            memory_used=0,  # Would need memory profiling in production
            cache_hit=False
        ))
        
        return result_dict
    
    async def execute_parallel_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple queries in parallel"""
        tasks = []
        for query_config in queries:
            task = self.execute_query(
                query_config['query'],
                query_config['database'],
                query_config.get('cache_ttl')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def create_dashboard_query(self, dashboard_config: Dict[str, Any]) -> str:
        """Generate optimized dashboard queries"""
        query_parts = []
        
        # Base metrics
        base_query = f"""
        SELECT 
            {dashboard_config['time_column']} as time_bucket,
            {dashboard_config['group_by']} as dimension,
            COUNT(*) as count,
            SUM({dashboard_config['measure_column']}) as total_value,
            AVG({dashboard_config['measure_column']}) as avg_value
        FROM {dashboard_config['table']}
        WHERE {dashboard_config['time_column']} >= '{dashboard_config['start_date']}'
        GROUP BY {dashboard_config['time_column']}, {dashboard_config['group_by']}
        ORDER BY {dashboard_config['time_column']} DESC
        """
        
        return base_query
    
    async def get_dashboard_data(self, dashboard_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Main metrics query
        main_query = self.create_dashboard_query(dashboard_config)
        
        # Additional queries for KPIs
        kpi_queries = [
            {
                'query': f"SELECT COUNT(*) as total_records FROM {dashboard_config['table']}",
                'database': dashboard_config['database'],
                'cache_ttl': 300
            },
            {
                'query': f"SELECT AVG({dashboard_config['measure_column']}) as overall_avg FROM {dashboard_config['table']}",
                'database': dashboard_config['database'], 
                'cache_ttl': 300
            }
        ]
        
        # Execute all queries in parallel
        main_result = await self.execute_query(
            main_query, 
            dashboard_config['database'],
            cache_ttl=60
        )
        
        kpi_results = await self.execute_parallel_queries(kpi_queries)
        
        return {
            'main_data': main_result,
            'kpis': kpi_results,
            'metadata': {
                'generated_at': time.time(),
                'query_count': len(kpi_queries) + 1
            }
        }
    
    def _record_metrics(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.metrics_collector.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_collector) > 1000:
            self.metrics_collector = self.metrics_collector[-1000:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analytics report"""
        if not self.metrics_collector:
            return {'message': 'No metrics available'}
        
        df = pd.DataFrame([
            {
                'query_id': m.query_id,
                'execution_time': m.execution_time,
                'rows_processed': m.rows_processed,
                'cache_hit': m.cache_hit
            }
            for m in self.metrics_collector
        ])
        
        return {
            'total_queries': len(df),
            'avg_execution_time': df['execution_time'].mean(),
            'cache_hit_rate': df['cache_hit'].mean(),
            'slowest_queries': df.nlargest(5, 'execution_time')[['query_id', 'execution_time']].to_dict('records'),
            'total_rows_processed': df['rows_processed'].sum()
        }

# Configuration example
platform_config = {
    'databases': [
        {
            'name': 'analytics_db',
            'connection_string': 'postgresql://user:password@localhost/analytics',
            'pool_size': 20
        },
        {
            'name': 'transactional_db', 
            'connection_string': 'mysql://user:password@localhost/transactions',
            'pool_size': 15
        }
    ],
    'redis_host': 'localhost',
    'redis_port': 6379,
    'max_workers': 10
}

# Dashboard configuration example
dashboard_config = {
    'table': 'sales_analytics',
    'database': 'analytics_db',
    'time_column': 'sale_date',
    'group_by': 'region',
    'measure_column': 'total_amount',
    'start_date': '2024-01-01'
}

# Usage example
async def main():
    platform = EnterpriseAnalyticsPlatform(platform_config)
    await platform.initialize_async_components()
    
    # Get dashboard data
    dashboard_data = await platform.get_dashboard_data(dashboard_config)
    print("Dashboard data:", dashboard_data)
    
    # Get performance metrics
    performance_report = platform.get_performance_report()
    print("Performance report:", performance_report)

# Run the example
# asyncio.run(main())
```
