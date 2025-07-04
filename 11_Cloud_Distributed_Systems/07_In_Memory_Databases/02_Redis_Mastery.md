# Redis Mastery

*Duration: 2-3 weeks*

## Description

Redis (Remote Dictionary Server) is an open-source, in-memory data structure store that can be used as a database, cache, message broker, and streaming engine. This comprehensive guide will take you from Redis basics to advanced clustering and real-world applications.

## Learning Path Overview

```
Week 1: Fundamentals & Data Structures
├── Redis Basics & Installation
├── Core Data Types (String, Hash, List, Set, Sorted Set)
├── Advanced Data Types (Streams, Bitmaps, HyperLogLog)
└── Redis Commands & CLI Mastery

Week 2: Advanced Features & Architecture
├── Persistence & Durability
├── Pub/Sub & Messaging
├── Transactions & Pipelines
├── Lua Scripting
└── Redis Modules

Week 3: Production & Scaling
├── Replication & High Availability
├── Clustering & Sharding
├── Monitoring & Performance
├── Security & Best Practices
└── Real-world Use Cases
```

## Prerequisites

- Basic understanding of databases and caching concepts
- Command line familiarity
- Basic programming knowledge (any language)
- Understanding of networking concepts (TCP/IP, ports)

## Redis Installation & Setup

### Installation Methods

**Docker (Recommended for Development):**
```bash
# Pull and run Redis
docker run --name redis-server -p 6379:6379 -d redis:latest

# Connect with Redis CLI
docker exec -it redis-server redis-cli

# Run with persistence
docker run --name redis-persistent \
  -p 6379:6379 \
  -v redis-data:/data \
  -d redis:latest redis-server --appendonly yes
```

**Linux Installation:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# CentOS/RHEL
sudo yum install epel-release
sudo yum install redis

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Check status
redis-cli ping  # Should return PONG
```

**Windows Installation:**
```bash
# Using WSL2 (recommended)
# Or download from GitHub releases
# Or use Redis Labs Redis for Windows
```

**Building from Source:**
```bash
wget http://download.redis.io/redis-stable.tar.gz
tar xzf redis-stable.tar.gz
cd redis-stable
make
make install

# Start Redis server
redis-server

# In another terminal, start CLI
redis-cli
```

## Redis Data Structures and Commands

### 1. Strings - The Foundation

Strings are the most basic Redis data type, capable of storing text, numbers, or binary data up to 512MB.

**Basic String Operations:**
```bash
# Setting and getting values
SET user:1:name "John Doe"
GET user:1:name                    # Returns "John Doe"

# Atomic increment/decrement
SET counter 10
INCR counter                       # Returns 11
INCRBY counter 5                   # Returns 16
DECR counter                       # Returns 15
DECRBY counter 3                   # Returns 12

# String length and substring
SET message "Hello, Redis World!"
STRLEN message                     # Returns 19
GETRANGE message 0 4              # Returns "Hello"
SETRANGE message 7 "Amazing"      # Modifies string

# Multiple operations
MSET user:1:name "John" user:1:age "30" user:1:city "NYC"
MGET user:1:name user:1:age user:1:city

# Expiration
SETEX session:abc123 3600 "user_data"  # Expires in 1 hour
TTL session:abc123                      # Check time to live
```

**Advanced String Use Cases:**
```bash
# Counters (page views, API rate limiting)
INCR page:home:views
INCRBY api:user:123:requests 1

# Caching with expiration
SETEX cache:user:123 300 '{"name":"John","email":"john@example.com"}'

# Atomic operations for distributed locks
SET lock:resource:1 "locked" NX EX 10  # Only set if not exists, expire in 10s
```

### 2. Hashes - Object Storage

Hashes are perfect for representing objects with multiple fields.

**Hash Operations:**
```bash
# Setting hash fields
HSET user:123 name "Alice" age 25 email "alice@example.com"
HGET user:123 name                 # Returns "Alice"
HGETALL user:123                   # Returns all fields and values

# Multiple field operations
HMSET product:456 name "Laptop" price 999.99 category "Electronics"
HMGET product:456 name price

# Increment numeric fields
HINCRBY user:123 age 1             # Increment age by 1
HINCRBYFLOAT product:456 price -50.00  # Decrease price

# Field existence and deletion
HEXISTS user:123 email             # Check if field exists
HDEL user:123 age                  # Delete age field
HLEN user:123                      # Number of fields
HKEYS user:123                     # Get all field names
HVALS user:123                     # Get all values
```

**Real-world Hash Examples:**
```bash
# User profile storage
HSET user:1001 
  name "Sarah Connor" 
  email "sarah@resistance.com" 
  last_login "2025-07-04T10:30:00Z" 
  login_count 42

# Shopping cart
HSET cart:session123 
  product:1 "2" 
  product:5 "1" 
  product:12 "3"
HINCRBY cart:session123 product:1 1  # Add one more item

# Application configuration
HSET config:app 
  db_host "localhost" 
  db_port "5432" 
  cache_ttl "3600" 
  debug_mode "false"
```

### 3. Lists - Ordered Collections

Lists are ordered collections of strings, perfect for queues, stacks, and timelines.

**List Operations:**
```bash
# Adding elements
LPUSH queue:tasks "task1" "task2" "task3"    # Add to left (beginning)
RPUSH queue:tasks "task4" "task5"            # Add to right (end)

# Removing elements
LPOP queue:tasks                             # Remove from left
RPOP queue:tasks                             # Remove from right
LLEN queue:tasks                             # Get length

# Accessing elements
LINDEX queue:tasks 0                         # Get first element
LRANGE queue:tasks 0 -1                      # Get all elements
LRANGE queue:tasks 0 2                       # Get first 3 elements

# Modifying lists
LSET queue:tasks 1 "modified_task"           # Set element at index
LTRIM queue:tasks 0 9                        # Keep only first 10 elements
```

**Advanced List Operations:**
```bash
# Blocking operations (for job queues)
BLPOP queue:high_priority queue:normal 5     # Block for 5 seconds
BRPOP queue:tasks 0                          # Block indefinitely

# Moving between lists
RPOPLPUSH source destination                 # Move element between lists
BRPOPLPUSH source destination 10             # Blocking move

# List as stack (LIFO)
LPUSH stack:items "item1" "item2" "item3"
LPOP stack:items                             # Returns "item3"

# List as queue (FIFO)
LPUSH queue:jobs "job1" "job2" "job3"
RPOP queue:jobs                              # Returns "job1"
```

**Real-world List Examples:**
```bash
# Activity feed/timeline
LPUSH user:123:timeline "Posted a new photo" "Liked John's post"
LRANGE user:123:timeline 0 9                # Get last 10 activities

# Task queue system
LPUSH queue:email_jobs '{"to":"user@example.com","subject":"Welcome"}'
BRPOP queue:email_jobs 0                    # Worker picks up job

# Recent searches
LPUSH user:456:recent_searches "redis tutorial" "docker commands"
LTRIM user:456:recent_searches 0 4          # Keep only 5 recent searches

# Chat messages
LPUSH chat:room:general '{"user":"Alice","msg":"Hello everyone","time":"..."}'
LRANGE chat:room:general 0 49               # Get last 50 messages
```

### 4. Sets - Unique Collections

Sets store unique, unordered collections of strings.

**Set Operations:**
```bash
# Basic set operations
SADD tags:article:123 "redis" "database" "nosql" "cache"
SMEMBERS tags:article:123                   # Get all members
SCARD tags:article:123                      # Get cardinality (count)
SISMEMBER tags:article:123 "redis"          # Check membership

# Set manipulation
SREM tags:article:123 "cache"               # Remove member
SPOP tags:article:123                       # Remove and return random member
SRANDMEMBER tags:article:123 2              # Get 2 random members

# Set operations between multiple sets
SADD skills:john "python" "redis" "docker"
SADD skills:jane "javascript" "redis" "kubernetes"

SINTER skills:john skills:jane              # Intersection (common skills)
SUNION skills:john skills:jane              # Union (all skills)
SDIFF skills:john skills:jane               # Difference (john's unique skills)
```

**Advanced Set Operations:**
```bash
# Store results of set operations
SINTERSTORE common_skills skills:john skills:jane
SUNIONSTORE all_skills skills:john skills:jane skills:bob

# Moving elements between sets
SMOVE skills:john skills:jane "docker"      # Move docker from john to jane
```

**Real-world Set Examples:**
```bash
# User permissions
SADD permissions:user:123 "read" "write" "delete"
SISMEMBER permissions:user:123 "admin"     # Check if user has admin rights

# Online users tracking
SADD online:users "user:123" "user:456" "user:789"
SCARD online:users                         # Count online users

# Article tags and recommendations
SADD tags:article:1 "programming" "python" "tutorial"
SADD tags:article:2 "programming" "javascript" "web"
SINTER tags:article:1 tags:article:2       # Find common tags

# Following/Followers system
SADD following:user:123 "user:456" "user:789"
SADD followers:user:456 "user:123" "user:321"
SINTER following:user:123 followers:user:123  # Mutual follows
```

### 5. Sorted Sets - Scored Collections

Sorted sets combine the uniqueness of sets with the ability to score and order elements.

**Sorted Set Operations:**
```bash
# Adding scored members
ZADD leaderboard 100 "player1" 85 "player2" 92 "player3"
ZRANGE leaderboard 0 -1                     # Get all members (by rank)
ZRANGE leaderboard 0 -1 WITHSCORES          # Include scores

# Score-based queries
ZRANGEBYSCORE leaderboard 90 100            # Members with score 90-100
ZREVRANGE leaderboard 0 2                   # Top 3 players (highest scores)
ZRANK leaderboard "player2"                 # Get rank of player2
ZSCORE leaderboard "player1"                # Get score of player1

# Score manipulation
ZINCRBY leaderboard 10 "player2"            # Increase player2's score by 10
ZCARD leaderboard                           # Count of members
ZCOUNT leaderboard 80 100                   # Count members with score 80-100

# Removing elements
ZREM leaderboard "player3"                  # Remove by member
ZREMRANGEBYRANK leaderboard 0 0             # Remove lowest ranked
ZREMRANGEBYSCORE leaderboard 0 50           # Remove low scores
```

**Advanced Sorted Set Operations:**
```bash
# Lexicographical operations (when scores are same)
ZADD words 0 "apple" 0 "banana" 0 "cherry"
ZRANGEBYLEX words "[a" "[c"                 # Words starting with a or b

# Set operations on sorted sets
ZINTERSTORE result:intersection 2 set1 set2 WEIGHTS 1 2  # Weighted intersection
ZUNIONSTORE result:union 2 set1 set2 AGGREGATE MAX      # Union with max scores
```

**Real-world Sorted Set Examples:**
```bash
# Gaming leaderboard
ZADD game:leaderboard 1500 "player:alice" 1200 "player:bob" 1800 "player:charlie"
ZREVRANGE game:leaderboard 0 9 WITHSCORES  # Top 10 players

# Trending articles (by view count)
ZADD trending:articles 1500 "article:123" 2300 "article:456" 890 "article:789"
ZREVRANGE trending:articles 0 4           # Top 5 trending articles

# Time-based data (using timestamp as score)
ZADD user:123:actions 1720099200 "login" 1720099260 "view_page" 1720099300 "logout"
ZRANGEBYSCORE user:123:actions 1720099000 1720099999  # Actions in time range

# Rate limiting with sliding window
ZADD api:user:123:requests 1720099200 "req1" 1720099210 "req2" 1720099220 "req3"
ZREMRANGEBYSCORE api:user:123:requests 0 (1720099200-3600)  # Remove requests older than 1 hour
ZCARD api:user:123:requests                                  # Count requests in last hour

# Price tracking
ZADD product:123:price_history 999.99 "2025-01-01" 899.99 "2025-02-01" 949.99 "2025-03-01"
ZRANGE product:123:price_history 0 -1 WITHSCORES   # Price history
```

### 6. Advanced Data Types

#### Streams - Event Sourcing & Message Queues

Redis Streams provide a powerful abstraction for managing streams of data with built-in persistence and consumer groups.

**Stream Basics:**
```bash
# Adding entries to stream
XADD mystream * field1 value1 field2 value2
XADD mystream * user "alice" action "login" timestamp "2025-07-04T10:30:00Z"

# Reading from stream
XREAD STREAMS mystream 0                    # Read all entries
XREAD STREAMS mystream $                    # Read new entries
XRANGE mystream - +                         # Read all entries in range
XLEN mystream                               # Get stream length

# Consumer groups
XGROUP CREATE mystream mygroup 0            # Create consumer group
XREADGROUP GROUP mygroup consumer1 STREAMS mystream >  # Read as consumer
XACK mystream mygroup entry-id              # Acknowledge processed message
```

**Advanced Stream Operations:**
```bash
# Stream with maxlen
XADD events * event "user_signup" MAXLEN ~ 1000  # Keep approximately 1000 entries

# Reading with blocking
XREAD BLOCK 5000 STREAMS events $            # Block for 5 seconds waiting for new data

# Consumer group with multiple consumers
XREADGROUP GROUP processors worker1 COUNT 5 STREAMS tasks >
XREADGROUP GROUP processors worker2 COUNT 5 STREAMS tasks >

# Pending messages handling
XPENDING tasks processors                    # Check pending messages
XCLAIM tasks processors worker2 3600000 entry-id  # Claim pending message
```

**Real-world Stream Examples:**
```bash
# User activity tracking
XADD user_activities * user_id 123 action "page_view" page "/products" timestamp "2025-07-04T10:30:00Z"
XADD user_activities * user_id 456 action "purchase" product_id 789 amount 99.99

# IoT sensor data
XADD sensor:temperature:room1 * temp 23.5 humidity 45 timestamp "2025-07-04T10:30:00Z"
XRANGE sensor:temperature:room1 - + COUNT 100  # Get last 100 readings

# Order processing pipeline
XADD orders * order_id 12345 customer_id 678 status "pending" items '[{"id":1,"qty":2}]'
XGROUP CREATE orders payment_processor 0
XGROUP CREATE orders inventory_manager 0
XGROUP CREATE orders shipping_handler 0
```

#### Bitmaps - Efficient Binary Data

Bitmaps allow you to perform bit-level operations on strings, perfect for tracking boolean states efficiently.

**Bitmap Operations:**
```bash
# Setting bits
SETBIT user:active:2025-07-04 123 1         # User 123 was active today
SETBIT user:active:2025-07-04 456 1         # User 456 was active today
SETBIT user:active:2025-07-03 123 1         # User 123 was active yesterday

# Getting bits
GETBIT user:active:2025-07-04 123           # Check if user 123 was active

# Counting set bits
BITCOUNT user:active:2025-07-04             # Count active users today
BITCOUNT user:active:2025-07-04 0 100       # Count active users in range

# Bitwise operations
BITOP AND result user:active:2025-07-04 user:active:2025-07-03  # Users active both days
BITOP OR result user:active:2025-07-04 user:active:2025-07-03   # Users active either day
BITCOUNT result                             # Count result
```

**Real-world Bitmap Examples:**
```bash
# Daily active users tracking
SETBIT dau:2025-07-04 123 1                 # User 123 active today
BITCOUNT dau:2025-07-04                     # Total DAU count

# Feature flags per user
SETBIT features:user:123 0 1                # Feature 0 enabled for user 123
SETBIT features:user:123 5 1                # Feature 5 enabled for user 123
GETBIT features:user:123 0                  # Check if feature 0 is enabled

# A/B testing groups
SETBIT experiment:variant_a 123 1           # User 123 in variant A
SETBIT experiment:variant_b 456 1           # User 456 in variant B
BITCOUNT experiment:variant_a               # Count users in variant A
```

#### HyperLogLog - Cardinality Estimation

HyperLogLog provides approximate cardinality (unique count) with very small memory footprint.

**HyperLogLog Operations:**
```bash
# Adding elements
PFADD unique_visitors "user123" "user456" "user789"
PFADD unique_visitors "user123" "user999"   # Duplicates are handled

# Counting unique elements
PFCOUNT unique_visitors                     # Approximate count of unique visitors

# Merging HyperLogLogs
PFADD page1_visitors "user1" "user2" "user3"
PFADD page2_visitors "user2" "user3" "user4"
PFMERGE total_visitors page1_visitors page2_visitors
PFCOUNT total_visitors                      # Unique visitors across both pages
```

**Real-world HyperLogLog Examples:**
```bash
# Website analytics
PFADD daily_unique_visitors:2025-07-04 "ip:192.168.1.1" "ip:10.0.0.1"
PFCOUNT daily_unique_visitors:2025-07-04   # Estimate unique visitors

# API endpoint usage
PFADD api_users:get_users "user123" "user456"
PFADD api_users:post_data "user789" "user123"
PFMERGE api_users:all api_users:get_users api_users:post_data
PFCOUNT api_users:all                      # Unique API users

# Search query tracking
PFADD search_queries:unique "redis tutorial" "docker guide" "kubernetes basics"
PFCOUNT search_queries:unique              # Approximate unique search terms
```

#### Geospatial - Location-based Data

Redis provides geospatial data types for location-based applications.

**Geospatial Operations:**
```bash
# Adding geospatial data
GEOADD cities -122.4194 37.7749 "San Francisco" -74.0059 40.7128 "New York"
GEOADD cities -0.1276 51.5074 "London" 2.3522 48.8566 "Paris"

# Distance calculations
GEODIST cities "San Francisco" "New York" km    # Distance in kilometers
GEODIST cities "London" "Paris" mi              # Distance in miles

# Finding nearby locations
GEORADIUS cities -122.4194 37.7749 1000 km WITHDIST WITHCOORD  # Within 1000km
GEORADIUSBYMEMBER cities "London" 500 km        # Cities within 500km of London

# Getting coordinates
GEOPOS cities "San Francisco" "London"          # Get coordinates
GEOHASH cities "New York"                       # Get geohash
```

**Real-world Geospatial Examples:**
```bash
# Restaurant finder
GEOADD restaurants -122.4194 37.7749 "Pizza Place" -122.4094 37.7849 "Burger Joint"
GEORADIUS restaurants -122.4144 37.7799 1 km WITHDIST  # Restaurants within 1km

# Delivery tracking
GEOADD drivers 2.3522 48.8566 "driver:123" 2.3622 48.8666 "driver:456"
GEORADIUSBYMEMBER drivers "driver:123" 5 km    # Other drivers within 5km

# Store locator
GEOADD stores -74.0059 40.7128 "store:nyc" -118.2437 34.0522 "store:la"
GEORADIUS stores -74.0000 40.7000 50 km WITHCOORD  # Stores near coordinates
```

## Redis Commands Mastery

### Essential Command Categories

#### Key Management
```bash
# Key operations
EXISTS mykey                               # Check if key exists
TYPE mykey                                 # Get key type
KEYS pattern                               # Find keys (avoid in production)
SCAN 0 MATCH user:* COUNT 10              # Iterate keys safely
DEL key1 key2 key3                        # Delete keys
UNLINK key1 key2                          # Async delete (non-blocking)

# Expiration
EXPIRE mykey 3600                          # Set expiration in seconds
EXPIREAT mykey 1720099200                  # Set expiration at timestamp
TTL mykey                                  # Time to live in seconds
PEXPIRE mykey 60000                        # Set expiration in milliseconds
PERSIST mykey                              # Remove expiration

# Key renaming
RENAME oldkey newkey                       # Rename key
RENAMENX oldkey newkey                     # Rename only if new key doesn't exist
```

#### Database Management
```bash
# Database selection
SELECT 0                                   # Switch to database 0 (default)
SELECT 1                                   # Switch to database 1

# Database operations
FLUSHDB                                    # Clear current database
FLUSHALL                                   # Clear all databases
DBSIZE                                     # Get number of keys in current DB

# Key migration
MOVE mykey 1                               # Move key to database 1
MIGRATE host port key destination-db timeout  # Migrate key to another Redis instance
```

#### Information and Monitoring
```bash
# Server information
INFO                                       # General server info
INFO memory                                # Memory usage info
INFO replication                           # Replication info
INFO stats                                 # Statistics

# Monitoring
MONITOR                                    # Monitor all commands (debug only)
CLIENT LIST                                # List connected clients
CLIENT KILL ip:port                        # Kill client connection
SLOWLOG GET 10                            # Get slow queries
CONFIG GET parameter                       # Get configuration
CONFIG SET parameter value                 # Set configuration
```

## Redis Persistence & Durability

Redis offers multiple persistence options to ensure data durability.

### RDB (Redis Database) Snapshots

RDB creates point-in-time snapshots of your dataset.

**Configuration:**
```bash
# In redis.conf
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Manual snapshots
BGSAVE                                     # Background save
LASTSAVE                                   # Get timestamp of last save
SAVE                                       # Synchronous save (blocks server)
```

**RDB Advantages & Disadvantages:**
```
✅ Advantages:
- Compact single-file backup
- Faster restart times
- Good for backups and disaster recovery
- Better performance (minimal impact on Redis)

❌ Disadvantages:
- Potential data loss between snapshots
- Can be CPU intensive for large datasets
- Not suitable for minimal data loss requirements
```

### AOF (Append Only File)

AOF logs every write operation for complete data recovery.

**Configuration:**
```bash
# In redis.conf
appendonly yes                             # Enable AOF
appendfilename "appendonly.aof"            # AOF filename

# Sync policies
appendfsync always      # Sync every write (slowest, safest)
appendfsync everysec    # Sync every second (good balance)
appendfsync no          # Let OS decide when to sync (fastest, least safe)

# AOF rewrite
auto-aof-rewrite-percentage 100            # Rewrite when size doubles
auto-aof-rewrite-min-size 64mb             # Minimum size for rewrite
```

**AOF Commands:**
```bash
BGREWRITEAOF                               # Trigger AOF rewrite
```

**Hybrid Persistence (RDB + AOF):**
```bash
# Best of both worlds
aof-use-rdb-preamble yes                   # Use RDB format for AOF rewrite
```

### Persistence Examples

**Backup Strategy:**
```bash
# Daily RDB backup script
#!/bin/bash
REDIS_CLI="redis-cli"
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d)

$REDIS_CLI BGSAVE
sleep 5  # Wait for save to complete
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump-$DATE.rdb
```

**Disaster Recovery:**
```bash
# Restore from RDB
sudo systemctl stop redis
cp /backups/redis/dump-20250704.rdb /var/lib/redis/dump.rdb
sudo systemctl start redis

# Restore from AOF
sudo systemctl stop redis
cp /backups/redis/appendonly-20250704.aof /var/lib/redis/appendonly.aof
sudo systemctl start redis
```

## Redis Pub/Sub & Messaging

Redis provides publish/subscribe messaging paradigm for real-time communication.

### Basic Pub/Sub

**Publishing Messages:**
```bash
# Publish to channel
PUBLISH news "Breaking: Redis 7.0 released!"
PUBLISH sports "Lakers win championship"
PUBLISH weather "Sunny, 25°C"
```

**Subscribing to Channels:**
```bash
# Subscribe to specific channels
SUBSCRIBE news sports                      # Subscribe to news and sports
UNSUBSCRIBE news                          # Unsubscribe from news

# Pattern subscriptions
PSUBSCRIBE news:*                         # Subscribe to all news channels
PSUBSCRIBE user:*:notifications           # Subscribe to user notification patterns
PUNSUBSCRIBE user:*:notifications         # Unsubscribe from pattern
```

**Checking Subscriptions:**
```bash
PUBSUB CHANNELS                           # List active channels
PUBSUB CHANNELS news:*                    # List channels matching pattern
PUBSUB NUMSUB news sports                 # Number of subscribers per channel
PUBSUB NUMPAT                             # Number of pattern subscriptions
```

### Real-world Pub/Sub Examples

**Chat Application:**
```python
import redis

# Publisher (when user sends message)
r = redis.Redis()
def send_message(room, user, message):
    payload = f"{user}: {message}"
    r.publish(f"chat:{room}", payload)

# Subscriber (chat client)
def listen_to_chat(room):
    pubsub = r.pubsub()
    pubsub.subscribe(f"chat:{room}")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"Room {room}: {message['data'].decode()}")
```

**Live Notifications:**
```python
# Publisher (when event occurs)
def notify_user(user_id, notification):
    r.publish(f"notifications:{user_id}", notification)

# Subscriber (user's browser/app)
def listen_for_notifications(user_id):
    pubsub = r.pubsub()
    pubsub.subscribe(f"notifications:{user_id}")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            # Send to user's browser via WebSocket
            send_to_browser(message['data'])
```

**Microservices Communication:**
```python
# Service A publishes events
def user_created(user_data):
    event = {
        "type": "user_created",
        "user_id": user_data["id"],
        "timestamp": time.time(),
        "data": user_data
    }
    r.publish("events:users", json.dumps(event))

# Service B subscribes to events
def handle_user_events():
    pubsub = r.pubsub()
    pubsub.subscribe("events:users")
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            event = json.loads(message['data'])
            if event['type'] == 'user_created':
                create_user_profile(event['data'])
```

## Redis Transactions & Pipelines

### Transactions with MULTI/EXEC

Redis transactions allow you to execute multiple commands atomically.

**Basic Transactions:**
```bash
# Transaction example
MULTI                                     # Start transaction
SET account:1:balance 100
SET account:2:balance 50
INCR transaction:count
EXEC                                      # Execute all commands atomically

# Transaction with conditional execution
WATCH account:1:balance                   # Watch key for changes
GET account:1:balance
MULTI
SET account:1:balance 90
EXEC                                      # Only executes if watched key unchanged
```

**Transaction Examples:**
```python
import redis

r = redis.Redis()

# Money transfer with transaction
def transfer_money(from_account, to_account, amount):
    with r.pipeline() as pipe:
        while True:
            try:
                pipe.watch(from_account, to_account)
                
                from_balance = float(pipe.get(from_account) or 0)
                to_balance = float(pipe.get(to_account) or 0)
                
                if from_balance < amount:
                    raise ValueError("Insufficient funds")
                
                pipe.multi()
                pipe.set(from_account, from_balance - amount)
                pipe.set(to_account, to_balance + amount)
                pipe.execute()
                break
                
            except redis.WatchError:
                # Retry if keys were modified
                continue

# Atomic counter increment
def atomic_increment_with_limit(key, limit):
    with r.pipeline() as pipe:
        while True:
            try:
                pipe.watch(key)
                current = int(pipe.get(key) or 0)
                
                if current >= limit:
                    return False  # Limit reached
                
                pipe.multi()
                pipe.incr(key)
                pipe.execute()
                return True
                
            except redis.WatchError:
                continue
```

### Pipelines for Performance

Pipelines allow you to send multiple commands without waiting for replies.

**Basic Pipeline:**
```python
# Without pipeline (slow - multiple round trips)
r.set("key1", "value1")
r.set("key2", "value2")
r.set("key3", "value3")
# 3 network round trips

# With pipeline (fast - single round trip)
pipe = r.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.set("key3", "value3")
results = pipe.execute()
# 1 network round trip
```

**Real-world Pipeline Examples:**
```python
# Bulk data loading
def bulk_load_users(users):
    pipe = r.pipeline()
    for user in users:
        pipe.hset(f"user:{user['id']}", mapping=user)
        pipe.sadd("all_users", user['id'])
    pipe.execute()

# Batch analytics update
def update_analytics(events):
    pipe = r.pipeline()
    for event in events:
        pipe.incr(f"stats:page_views:{event['page']}")
        pipe.sadd(f"active_users:{event['date']}", event['user_id'])
        pipe.lpush(f"user_activity:{event['user_id']}", event['action'])
    pipe.execute()

# Performance comparison
import time

def benchmark_pipeline():
    # Without pipeline
    start = time.time()
    for i in range(1000):
        r.set(f"key:{i}", f"value:{i}")
    no_pipeline_time = time.time() - start
    
    # With pipeline
    start = time.time()
    pipe = r.pipeline()
    for i in range(1000):
        pipe.set(f"key:{i}", f"value:{i}")
    pipe.execute()
    pipeline_time = time.time() - start
    
    print(f"Without pipeline: {no_pipeline_time:.2f}s")
    print(f"With pipeline: {pipeline_time:.2f}s")
    print(f"Speedup: {no_pipeline_time/pipeline_time:.2f}x")
```

## Lua Scripting in Redis

Redis allows you to execute Lua scripts atomically on the server side.

### Basic Lua Scripting

**Simple Script Example:**
```bash
# Inline script
EVAL "return redis.call('GET', KEYS[1])" 1 mykey

# Script with logic
EVAL "
local current = redis.call('GET', KEYS[1])
if current == false then
    redis.call('SET', KEYS[1], ARGV[1])
    return 1
else
    return 0
end
" 1 mykey "initial_value"
```

**Script Management:**
```bash
# Load script and get SHA
SCRIPT LOAD "return redis.call('GET', KEYS[1])"
# Returns: "6b1bf486c81ceb7edf3c093f4c48582e38c0e791"

# Execute loaded script by SHA
EVALSHA 6b1bf486c81ceb7edf3c093f4c48582e38c0e791 1 mykey

# Check if script exists
SCRIPT EXISTS 6b1bf486c81ceb7edf3c093f4c48582e38c0e791

# Clear script cache
SCRIPT FLUSH
```

### Advanced Lua Scripts

**Rate Limiting Script:**
```lua
-- Rate limiting with sliding window
-- KEYS[1]: rate limit key
-- ARGV[1]: window size in seconds
-- ARGV[2]: max requests in window
-- ARGV[3]: current timestamp

local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

-- Count current requests
local current = redis.call('ZCARD', key)

if current < limit then
    -- Add current request
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return {1, limit - current - 1}  -- [allowed, remaining]
else
    return {0, 0}  -- [not allowed, remaining]
end
```

**Atomic Counter with Bounds:**
```lua
-- Atomic increment with upper and lower bounds
-- KEYS[1]: counter key
-- ARGV[1]: increment amount
-- ARGV[2]: min value
-- ARGV[3]: max value

local key = KEYS[1]
local increment = tonumber(ARGV[1])
local min_val = tonumber(ARGV[2])
local max_val = tonumber(ARGV[3])

local current = tonumber(redis.call('GET', key) or 0)
local new_val = current + increment

if new_val < min_val then
    new_val = min_val
elseif new_val > max_val then
    new_val = max_val
end

redis.call('SET', key, new_val)
return new_val
```

**Distributed Lock Script:**
```lua
-- Acquire distributed lock
-- KEYS[1]: lock key
-- ARGV[1]: lock value (unique identifier)
-- ARGV[2]: expiration time in milliseconds

local key = KEYS[1]
local value = ARGV[1]
local ttl = tonumber(ARGV[2])

local result = redis.call('SET', key, value, 'PX', ttl, 'NX')
if result then
    return 1  -- Lock acquired
else
    return 0  -- Lock not acquired
end
```

### Python Integration with Lua Scripts

```python
import redis
import time
import uuid

r = redis.Redis()

# Rate limiting example
rate_limit_script = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local current = redis.call('ZCARD', key)

if current < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return {1, limit - current - 1}
else
    return {0, 0}
end
"""

def is_rate_limited(user_id, window=60, limit=10):
    key = f"rate_limit:{user_id}"
    now = time.time()
    
    result = r.eval(rate_limit_script, 1, key, window, limit, now)
    allowed, remaining = result
    
    return not allowed, remaining

# Distributed lock example
lock_script = """
local key = KEYS[1]
local value = ARGV[1]
local ttl = tonumber(ARGV[2])

return redis.call('SET', key, value, 'PX', ttl, 'NX')
"""

class DistributedLock:
    def __init__(self, redis_client, key, ttl=30000):
        self.redis = redis_client
        self.key = key
        self.value = str(uuid.uuid4())
        self.ttl = ttl
        
    def acquire(self):
        result = self.redis.eval(lock_script, 1, self.key, self.value, self.ttl)
        return result is not None
        
    def release(self):
        release_script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """
        return self.redis.eval(release_script, 1, self.key, self.value)

# Usage
lock = DistributedLock(r, "resource:123")
if lock.acquire():
    try:
        # Critical section
        print("Lock acquired, doing work...")
        time.sleep(5)
    finally:
        lock.release()
else:
    print("Could not acquire lock")
```

## Redis Clustering & High Availability

### Redis Sentinel - High Availability

Redis Sentinel provides monitoring, notification, and automatic failover for Redis deployments.

#### Sentinel Configuration

**Master Configuration (redis-master.conf):**
```
port 6379
bind 0.0.0.0
# Enable AOF for better durability in HA setup
appendonly yes
appendfsync everysec
```

**Replica Configuration (redis-replica.conf):**
```
port 6380
bind 0.0.0.0
replicaof 127.0.0.1 6379
replica-read-only yes
```

**Sentinel Configuration (sentinel.conf):**
```
port 26379
bind 0.0.0.0

# Monitor master
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
sentinel parallel-syncs mymaster 1

# Authentication (if needed)
sentinel auth-pass mymaster your_password

# Notification scripts
sentinel notification-script mymaster /scripts/notify.sh
sentinel client-reconfig-script mymaster /scripts/reconfig.sh
```

#### Starting Sentinel Setup

```bash
# Start master
redis-server redis-master.conf

# Start replicas
redis-server redis-replica.conf --port 6380
redis-server redis-replica.conf --port 6381

# Start sentinels (minimum 3 for quorum)
redis-sentinel sentinel1.conf --port 26379
redis-sentinel sentinel2.conf --port 26380
redis-sentinel sentinel3.conf --port 26381
```

#### Sentinel Commands

```bash
# Connect to sentinel
redis-cli -p 26379

# Sentinel information
SENTINEL masters                           # List monitored masters
SENTINEL slaves mymaster                   # List slaves for master
SENTINEL sentinels mymaster               # List other sentinels
SENTINEL get-master-addr-by-name mymaster  # Get current master address

# Manual operations
SENTINEL failover mymaster                 # Force failover
SENTINEL reset mymaster                    # Reset sentinel state
```

#### Application Integration with Sentinel

```python
import redis.sentinel

# Configure sentinel
sentinel = redis.sentinel.Sentinel([
    ('localhost', 26379),
    ('localhost', 26380),
    ('localhost', 26381)
])

# Get master and slave connections
master = sentinel.master_for('mymaster', socket_timeout=0.1)
slave = sentinel.slave_for('mymaster', socket_timeout=0.1)

# Use master for writes, slave for reads
def write_data(key, value):
    master.set(key, value)

def read_data(key):
    try:
        return slave.get(key)
    except:
        # Fallback to master if slave unavailable
        return master.get(key)

# Handle failover events
def sentinel_event_handler():
    pubsub = sentinel.pubsub()
    pubsub.subscribe('__sentinel__:hello')
    
    for message in pubsub.listen():
        print(f"Sentinel event: {message}")
```

### Redis Cluster - Horizontal Scaling

Redis Cluster provides automatic data sharding across multiple Redis nodes.

#### Cluster Setup

**Node Configuration (redis-cluster.conf):**
```
port 7000
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
```

**Creating a Cluster:**
```bash
# Start 6 nodes (3 masters + 3 replicas)
for port in 7000 7001 7002 7003 7004 7005; do
    redis-server --port $port --cluster-enabled yes \
                 --cluster-config-file nodes-$port.conf \
                 --cluster-node-timeout 5000 \
                 --appendonly yes \
                 --daemonize yes
done

# Create cluster
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
                          127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
                          --cluster-replicas 1
```

#### Cluster Management

```bash
# Cluster information
redis-cli -c -p 7000 CLUSTER NODES          # List all nodes
redis-cli -c -p 7000 CLUSTER INFO           # Cluster status
redis-cli -c -p 7000 CLUSTER SLOTS          # Slot distribution

# Adding nodes
redis-cli --cluster add-node 127.0.0.1:7006 127.0.0.1:7000
redis-cli --cluster reshard 127.0.0.1:7000  # Redistribute slots

# Removing nodes
redis-cli --cluster del-node 127.0.0.1:7000 node-id

# Cluster health check
redis-cli --cluster check 127.0.0.1:7000
```

#### Application Integration with Cluster

```python
from rediscluster import RedisCluster

# Cluster connection
startup_nodes = [
    {"host": "127.0.0.1", "port": "7000"},
    {"host": "127.0.0.1", "port": "7001"},
    {"host": "127.0.0.1", "port": "7002"}
]

rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

# Normal operations work transparently
rc.set("user:1001", "John Doe")
rc.set("user:1002", "Jane Smith")
print(rc.get("user:1001"))

# Hash tags for keeping related keys on same slot
rc.set("user:{1001}:profile", "profile_data")
rc.set("user:{1001}:preferences", "pref_data")
# Both keys will be on the same node due to hash tag {1001}

# Pipeline with cluster
def cluster_pipeline_example():
    pipe = rc.pipeline()
    for i in range(100):
        pipe.set(f"key:{i}", f"value:{i}")
    results = pipe.execute()
    return results

# Handling cluster events
def monitor_cluster():
    try:
        for key in rc.scan_iter("user:*"):
            print(f"Found key: {key}")
    except Exception as e:
        print(f"Cluster error: {e}")
        # Implement retry logic or fallback
```

### Replication Strategies

#### Master-Slave Replication

**Setting up Replication:**
```bash
# On slave server
redis-cli REPLICAOF master-ip 6379

# Check replication status
redis-cli INFO replication

# Make slave read-only (default)
CONFIG SET replica-read-only yes
```

**Replication Configuration:**
```
# In redis.conf for replica
replicaof master-ip 6379
replica-read-only yes
replica-serve-stale-data yes
replica-priority 100

# Replication settings for master
repl-diskless-sync no
repl-diskless-sync-delay 5
repl-ping-replica-period 10
repl-timeout 60
```

#### Read/Write Splitting

```python
import redis
import random

class RedisReadWriteSplit:
    def __init__(self, master_config, slave_configs):
        self.master = redis.Redis(**master_config)
        self.slaves = [redis.Redis(**config) for config in slave_configs]
    
    def write(self, *args, **kwargs):
        """All writes go to master"""
        return self.master.execute_command(*args, **kwargs)
    
    def read(self, *args, **kwargs):
        """Reads can go to any slave (or master if no slaves)"""
        if self.slaves:
            slave = random.choice(self.slaves)
            try:
                return slave.execute_command(*args, **kwargs)
            except:
                # Fallback to master if slave fails
                return self.master.execute_command(*args, **kwargs)
        else:
            return self.master.execute_command(*args, **kwargs)

# Usage
redis_split = RedisReadWriteSplit(
    master_config={'host': 'master.redis.com', 'port': 6379},
    slave_configs=[
        {'host': 'slave1.redis.com', 'port': 6379},
        {'host': 'slave2.redis.com', 'port': 6379}
    ]
)

# Writes go to master
redis_split.write('SET', 'user:123', 'John Doe')

# Reads can go to slaves
user_data = redis_split.read('GET', 'user:123')
```

## Redis Modules and Extensions

Redis modules extend Redis functionality with custom data types and commands.

### Popular Redis Modules

#### RedisJSON - JSON Data Type

```bash
# Install RedisJSON
# Download from https://redisjson.io/
redis-server --loadmodule /path/to/redisjson.so

# JSON operations
JSON.SET user:1001 $ '{"name":"John","age":30,"skills":["Python","Redis"]}'
JSON.GET user:1001                         # Get entire object
JSON.GET user:1001 $.name                 # Get specific field
JSON.SET user:1001 $.age 31               # Update field
JSON.ARRAPPEND user:1001 $.skills '"JavaScript"'  # Add to array
JSON.DEL user:1001 $.age                  # Delete field
```

**Python with RedisJSON:**
```python
import redis
import json
import time

r = redis.Redis()

# Store complex JSON
user_data = {
    "id": 1001,
    "profile": {
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {
            "theme": "dark",
            "notifications": True
        }
    },
    "tags": ["developer", "redis-expert"]
}

r.execute_command('JSON.SET', 'user:1001', '$', json.dumps(user_data))

# Query JSON
name = r.execute_command('JSON.GET', 'user:1001', '$.profile.name')
theme = r.execute_command('JSON.GET', 'user:1001', '$.profile.preferences.theme')

# Update nested fields
r.execute_command('JSON.SET', 'user:1001', '$.profile.preferences.theme', '"light"')
```

#### RedisTimeSeries - Time Series Data

```bash
# Load module
redis-server --loadmodule /path/to/redistimeseries.so

# Create time series
TS.CREATE temperature:sensor1 RETENTION 86400000 LABELS sensor_id 1 location room1

# Add data points
TS.ADD temperature:sensor1 * 23.5          # Current timestamp
TS.ADD temperature:sensor1 1720099200000 24.1  # Specific timestamp

# Query data
TS.RANGE temperature:sensor1 - +           # All data
TS.RANGE temperature:sensor1 1720099200000 1720099800000  # Time range

# Aggregation
TS.RANGE temperature:sensor1 - + AGGREGATION avg 3600000  # Hourly averages
```

#### RediSearch - Full-Text Search

```bash
# Load module
redis-server --loadmodule /path/to/redisearch.so

# Create index
FT.CREATE books_idx ON HASH PREFIX 1 book: SCHEMA title TEXT WEIGHT 5.0 author TEXT year NUMERIC

# Add documents
HSET book:1 title "Redis in Action" author "Josiah Carlson" year 2013
HSET book:2 title "Redis Essentials" author "Maxwell Dayvson Da Silva" year 2015

# Search
FT.SEARCH books_idx "redis"                # Simple search
FT.SEARCH books_idx "@title:redis @year:[2010 2020]"  # Field-specific search
FT.SEARCH books_idx "redis" LIMIT 0 10     # Paginated results
```

### Custom Module Development

**Simple Module Example (C):**
```c
#include "redismodule.h"

// Command: HELLO.WORLD
int HelloWorld_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModule_ReplyWithSimpleString(ctx, "Hello, Redis World!");
    return REDISMODULE_OK;
}

// Module initialization
int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_Init(ctx, "hello", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    
    if (RedisModule_CreateCommand(ctx, "hello.world", HelloWorld_RedisCommand, 
                                  "readonly", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    
    return REDISMODULE_OK;
}
```

**Compile and Load:**
```bash
# Compile module
gcc -fPIC -shared -o hello.so hello.c -I/path/to/redis/src

# Load in Redis
MODULE LOAD /path/to/hello.so

# Use command
HELLO.WORLD
```

## Performance Optimization & Monitoring

### Performance Best Practices

#### Memory Optimization

```bash
# Memory analysis
MEMORY USAGE keyname                       # Memory used by specific key
MEMORY STATS                               # Overall memory statistics
INFO memory                                # Memory section info

# Configuration for memory efficiency
maxmemory 2gb                              # Set memory limit
maxmemory-policy allkeys-lru               # Eviction policy
hash-max-ziplist-entries 512               # Optimize hash encoding
set-max-intset-entries 512                 # Optimize set encoding
```

**Memory-Efficient Data Structures:**
```python
# Use hashes for objects instead of multiple keys
# Bad: Multiple keys
r.set("user:1001:name", "John")
r.set("user:1001:email", "john@example.com")
r.set("user:1001:age", "30")

# Good: Single hash
r.hset("user:1001", mapping={
    "name": "John",
    "email": "john@example.com", 
    "age": "30"
})

# Use appropriate data types
# For sets of integers, use intsets (automatic)
r.sadd("user:1001:friends", 123, 456, 789)  # Stored as intset

# For large strings, consider compression
import zlib
compressed_data = zlib.compress(large_string.encode())
r.set("large_data:1", compressed_data)
```

#### Network Optimization

```python
# Use pipelining for bulk operations
def bulk_operations_optimized():
    pipe = r.pipeline()
    for i in range(1000):
        pipe.set(f"key:{i}", f"value:{i}")
    pipe.execute()  # Single network round trip

# Connection pooling
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=20,
    socket_keepalive=True,
    socket_keepalive_options={}
)

r = redis.Redis(connection_pool=pool)
```

### Monitoring and Alerting

#### Key Metrics to Monitor

```bash
# Performance metrics
INFO stats                                 # Operations per second, hits/misses
INFO clients                               # Connected clients
INFO memory                                # Memory usage
INFO replication                           # Replication lag
INFO persistence                           # Last save time, changes since save

# Latency monitoring
LATENCY DOCTOR                             # Latency analysis
LATENCY HISTORY command-name               # Command latency history
CONFIG SET latency-monitor-threshold 100   # Monitor commands > 100ms
```

**Monitoring Script Example:**
```python
import redis
import time
import json

def redis_monitor():
    r = redis.Redis()
    
    while True:
        info = r.info()
        
        metrics = {
            'timestamp': time.time(),
            'used_memory': info['used_memory'],
            'used_memory_peak': info['used_memory_peak'],
            'connected_clients': info['connected_clients'],
            'ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
            'keyspace_hits': info['keyspace_hits'],
            'keyspace_misses': info['keyspace_misses'],
            'hit_rate': info['keyspace_hits'] / (info['keyspace_hits'] + info['keyspace_misses']) if (info['keyspace_hits'] + info['keyspace_misses']) > 0 else 0
        }
        
        # Send metrics to monitoring system
        print(json.dumps(metrics))
        
        # Alert if memory usage > 80%
        if info['used_memory'] > info['maxmemory'] * 0.8:
            send_alert("Redis memory usage high")
        
        time.sleep(10)

def send_alert(message):
    # Implement alerting (email, Slack, etc.)
    print(f"ALERT: {message}")
```

#### Slow Query Analysis

```bash
# Enable slow log
CONFIG SET slowlog-log-slower-than 10000   # Log queries > 10ms
CONFIG SET slowlog-max-len 128             # Keep last 128 slow queries

# View slow queries
SLOWLOG GET 10                             # Get last 10 slow queries
SLOWLOG LEN                                # Number of slow queries
SLOWLOG RESET                              # Clear slow log
```

**Slow Query Analysis Script:**
```python
def analyze_slow_queries():
    r = redis.Redis()
    slow_queries = r.slowlog_get(100)
    
    # Analyze patterns
    command_counts = {}
    for query in slow_queries:
        command = query['command'][0].decode()
        command_counts[command] = command_counts.get(command, 0) + 1
    
    print("Slow query patterns:")
    for command, count in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{command}: {count} times")
    
    # Find longest queries
    longest_queries = sorted(slow_queries, key=lambda x: x['duration'], reverse=True)[:5]
    print("\nTop 5 slowest queries:")
    for query in longest_queries:
        print(f"Duration: {query['duration']}μs, Command: {' '.join(query['command'])}")
```

## Real-world Redis Use Cases

### 1. Caching Layer

Redis is commonly used as a cache to reduce database load and improve application performance.

#### Application-Level Caching

```python
import redis
import json
import time
from functools import wraps

r = redis.Redis()

def cache_result(expiration=3600):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"cache:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = r.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            r.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage example
@cache_result(expiration=1800)  # Cache for 30 minutes
def get_user_profile(user_id):
    # Expensive database operation
    time.sleep(2)  # Simulate slow DB query
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }

# Cache-aside pattern implementation
class UserCache:
    def __init__(self, redis_client, db_client):
        self.redis = redis_client
        self.db = db_client
        self.ttl = 3600  # 1 hour
    
    def get_user(self, user_id):
        cache_key = f"user:{user_id}"
        
        # Try cache first
        cached_user = self.redis.get(cache_key)
        if cached_user:
            return json.loads(cached_user)
        
        # Cache miss - get from database
        user = self.db.get_user(user_id)
        if user:
            # Store in cache
            self.redis.setex(cache_key, self.ttl, json.dumps(user))
        
        return user
    
    def update_user(self, user_id, user_data):
        # Update database
        self.db.update_user(user_id, user_data)
        
        # Invalidate cache
        self.redis.delete(f"user:{user_id}")
        
        # Or update cache with new data
        # self.redis.setex(f"user:{user_id}", self.ttl, json.dumps(user_data))
```

#### HTTP Response Caching

```python
from flask import Flask, request, jsonify
import hashlib

app = Flask(__name__)

def cache_response(ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from request
            cache_key = f"http_cache:{request.endpoint}:{hashlib.md5(request.url.encode()).hexdigest()}"
            
            # Check cache
            cached_response = r.get(cache_key)
            if cached_response:
                return json.loads(cached_response)
            
            # Generate response
            response = func(*args, **kwargs)
            
            # Cache response
            r.setex(cache_key, ttl, json.dumps(response))
            return response
        return wrapper
    return decorator

@app.route('/api/products')
@cache_response(ttl=600)  # Cache for 10 minutes
def get_products():
    # Expensive operation
    products = fetch_products_from_database()
    return {"products": products}
```

### 2. Session Management

Redis excels at storing user sessions in web applications.

```python
import uuid
import json
from datetime import datetime, timedelta

class RedisSessionManager:
    def __init__(self, redis_client, default_ttl=3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
    
    def create_session(self, user_id, user_data=None):
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "data": user_data or {}
        }
        
        session_key = f"session:{session_id}"
        self.redis.setex(session_key, self.default_ttl, json.dumps(session_data))
        return session_id
    
    def get_session(self, session_id):
        session_key = f"session:{session_id}"
        session_data = self.redis.get(session_key)
        
        if session_data:
            session = json.loads(session_data)
            # Update last activity
            session["last_activity"] = datetime.now().isoformat()
            self.redis.setex(session_key, self.default_ttl, json.dumps(session))
            return session
        
        return None
    
    def update_session(self, session_id, data):
        session_key = f"session:{session_id}"
        session_data = self.redis.get(session_key)
        
        if session_data:
            session = json.loads(session_data)
            session["data"].update(data)
            session["last_activity"] = datetime.now().isoformat()
            self.redis.setex(session_key, self.default_ttl, json.dumps(session))
            return True
        
        return False
    
    def delete_session(self, session_id):
        session_key = f"session:{session_id}"
        return self.redis.delete(session_key)
    
    def cleanup_expired_sessions(self):
        # This is handled automatically by Redis TTL
        # But you might want to track active sessions
        pass

# Flask integration
from flask import Flask, request, session as flask_session

app = Flask(__name__)
session_manager = RedisSessionManager(r)

@app.before_request
def load_session():
    session_id = request.cookies.get('session_id')
    if session_id:
        session_data = session_manager.get_session(session_id)
        if session_data:
            flask_session.update(session_data)

@app.after_request
def save_session(response):
    session_id = request.cookies.get('session_id')
    if session_id and flask_session:
        session_manager.update_session(session_id, dict(flask_session))
    return response
```

### 3. Real-time Analytics & Leaderboards

```python
import time
from datetime import datetime, timedelta

class RealTimeAnalytics:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def track_page_view(self, page, user_id=None):
        now = int(time.time())
        
        # Increment total page views
        self.redis.incr(f"stats:page_views:{page}:total")
        
        # Track hourly views
        hour_key = f"stats:page_views:{page}:hourly:{now // 3600}"
        self.redis.incr(hour_key)
        self.redis.expire(hour_key, 86400)  # Keep for 24 hours
        
        # Track unique daily visitors using HyperLogLog
        if user_id:
            daily_key = f"stats:unique_visitors:{page}:{datetime.now().date()}"
            self.redis.pfadd(daily_key, user_id)
            self.redis.expire(daily_key, 86400 * 7)  # Keep for 7 days
    
    def get_page_stats(self, page):
        total_views = self.redis.get(f"stats:page_views:{page}:total") or 0
        
        # Get last 24 hours
        now = int(time.time())
        hourly_views = []
        for i in range(24):
            hour = now // 3600 - i
            hour_key = f"stats:page_views:{page}:hourly:{hour}"
            views = self.redis.get(hour_key) or 0
            hourly_views.append({"hour": hour, "views": int(views)})
        
        # Get unique visitors today
        daily_key = f"stats:unique_visitors:{page}:{datetime.now().date()}"
        unique_visitors = self.redis.pfcount(daily_key)
        
        return {
            "total_views": int(total_views),
            "hourly_views": hourly_views,
            "unique_visitors_today": unique_visitors
        }

class GameLeaderboard:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def update_score(self, game_id, player_id, score):
        leaderboard_key = f"leaderboard:{game_id}"
        
        # Update player score (Redis will handle sorting)
        self.redis.zadd(leaderboard_key, {player_id: score})
        
        # Track player's best score
        best_score_key = f"player:{player_id}:best_score:{game_id}"
        current_best = self.redis.get(best_score_key)
        if not current_best or score > int(current_best):
            self.redis.set(best_score_key, score)
    
    def get_leaderboard(self, game_id, start=0, end=9):
        leaderboard_key = f"leaderboard:{game_id}"
        
        # Get top players with scores
        top_players = self.redis.zrevrange(
            leaderboard_key, start, end, withscores=True
        )
        
        leaderboard = []
        for rank, (player_id, score) in enumerate(top_players, start=start+1):
            leaderboard.append({
                "rank": rank,
                "player_id": player_id.decode(),
                "score": int(score)
            })
        
        return leaderboard
    
    def get_player_rank(self, game_id, player_id):
        leaderboard_key = f"leaderboard:{game_id}"
        rank = self.redis.zrevrank(leaderboard_key, player_id)
        return rank + 1 if rank is not None else None
    
    def get_players_around(self, game_id, player_id, range_size=5):
        """Get players ranked around a specific player"""
        player_rank = self.get_player_rank(game_id, player_id)
        if not player_rank:
            return []
        
        start = max(0, player_rank - range_size - 1)
        end = player_rank + range_size - 1
        
        return self.get_leaderboard(game_id, start, end)

# Usage examples
analytics = RealTimeAnalytics(r)
leaderboard = GameLeaderboard(r)

# Track page views
analytics.track_page_view("/home", user_id="user123")
analytics.track_page_view("/products", user_id="user456")

# Update game scores
leaderboard.update_score("space_invaders", "player1", 15000)
leaderboard.update_score("space_invaders", "player2", 12000)
leaderboard.update_score("space_invaders", "player1", 18000)  # New high score

# Get results
stats = analytics.get_page_stats("/home")
top_players = leaderboard.get_leaderboard("space_invaders")
player_rank = leaderboard.get_player_rank("space_invaders", "player1")
```

### 4. Rate Limiting & API Throttling

```python
import time
import math

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def fixed_window_limiter(self, key, limit, window):
        """Fixed window rate limiter"""
        current_window = int(time.time()) // window
        window_key = f"rate_limit:{key}:{current_window}"
        
        current_requests = self.redis.incr(window_key)
        if current_requests == 1:
            self.redis.expire(window_key, window)
        
        return current_requests <= limit
    
    def sliding_window_limiter(self, key, limit, window):
        """Sliding window rate limiter using sorted sets"""
        now = time.time()
        pipeline = self.redis.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {f"{now}:{hash(now)}": now})
        
        # Set expiration
        pipeline.expire(key, window + 1)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit
    
    def token_bucket_limiter(self, key, capacity, refill_rate, requested_tokens=1):
        """Token bucket rate limiter"""
        bucket_key = f"bucket:{key}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = self.redis.hmget(bucket_key, 'tokens', 'last_refill')
        tokens = float(bucket_data[0] or capacity)
        last_refill = float(bucket_data[1] or now)
        
        # Add tokens based on time passed
        time_passed = now - last_refill
        tokens = min(capacity, tokens + time_passed * refill_rate)
        
        # Check if we have enough tokens
        if tokens >= requested_tokens:
            tokens -= requested_tokens
            
            # Update bucket state
            self.redis.hmset(bucket_key, {
                'tokens': tokens,
                'last_refill': now
            })
            self.redis.expire(bucket_key, 3600)  # Keep bucket for 1 hour
            
            return True
        else:
            # Update last_refill even if request is denied
            self.redis.hmset(bucket_key, {
                'tokens': tokens,
                'last_refill': now
            })
            self.redis.expire(bucket_key, 3600)
            
            return False

# Flask integration
from flask import Flask, request, jsonify
import functools

app = Flask(__name__)
rate_limiter = RateLimiter(r)

def rate_limit(limit=100, window=3600, limiter_type="sliding"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use client IP as key (you might want user ID instead)
            client_key = request.remote_addr
            
            if limiter_type == "sliding":
                allowed = rate_limiter.sliding_window_limiter(client_key, limit, window)
            elif limiter_type == "fixed":
                allowed = rate_limiter.fixed_window_limiter(client_key, limit, window)
            elif limiter_type == "token_bucket":
                allowed = rate_limiter.token_bucket_limiter(client_key, limit, limit/window)
            
            if not allowed:
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@app.route('/api/data')
@rate_limit(limit=10, window=60, limiter_type="sliding")  # 10 requests per minute
def get_data():
    return jsonify({"data": "some data"})

# Advanced rate limiting with multiple tiers
class TieredRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.limits = {
            "free": {"requests": 100, "window": 3600},
            "premium": {"requests": 1000, "window": 3600},
            "enterprise": {"requests": 10000, "window": 3600}
        }
    
    def check_limits(self, user_id, user_tier):
        limits = self.limits.get(user_tier, self.limits["free"])
        
        return rate_limiter.sliding_window_limiter(
            f"user:{user_id}",
            limits["requests"],
            limits["window"]
        )

tiered_limiter = TieredRateLimiter(r)

@app.route('/api/premium-data')
def premium_data():
    user_id = request.headers.get('X-User-ID')
    user_tier = request.headers.get('X-User-Tier', 'free')
    
    if not tiered_limiter.check_limits(user_id, user_tier):
        return jsonify({"error": "Rate limit exceeded for your tier"}), 429
    
    return jsonify({"premium_data": "valuable information"})
```

### 5. Queue Systems & Job Processing

```python
import json
import uuid
import time
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RedisJobQueue:
    def __init__(self, redis_client, queue_name="default"):
        self.redis = redis_client
        self.queue_name = queue_name
        self.queue_key = f"queue:{queue_name}"
        self.processing_key = f"queue:{queue_name}:processing"
        self.jobs_key = f"jobs:{queue_name}"
    
    def enqueue(self, job_type, data, delay=0, priority=0):
        """Add job to queue"""
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "type": job_type,
            "data": data,
            "status": JobStatus.PENDING.value,
            "created_at": time.time(),
            "priority": priority
        }
        
        # Store job details
        self.redis.hset(self.jobs_key, job_id, json.dumps(job))
        
        if delay > 0:
            # Delayed job using sorted set with timestamp
            execute_at = time.time() + delay
            self.redis.zadd(f"{self.queue_key}:delayed", {job_id: execute_at})
        else:
            # Immediate job using list (with priority support)
            if priority > 0:
                self.redis.lpush(f"{self.queue_key}:high", job_id)
            else:
                self.redis.rpush(self.queue_key, job_id)
        
        return job_id
    
    def dequeue(self, timeout=10):
        """Get next job from queue"""
        # Check delayed jobs first
        self._process_delayed_jobs()
        
        # Try high priority queue first, then normal queue
        job_id = self.redis.blpop([
            f"{self.queue_key}:high",
            self.queue_key
        ], timeout=timeout)
        
        if job_id:
            job_id = job_id[1].decode()
            
            # Move to processing
            self.redis.sadd(self.processing_key, job_id)
            
            # Get job details
            job_data = self.redis.hget(self.jobs_key, job_id)
            if job_data:
                job = json.loads(job_data)
                job["status"] = JobStatus.PROCESSING.value
                job["started_at"] = time.time()
                self.redis.hset(self.jobs_key, job_id, json.dumps(job))
                
                return job
        
        return None
    
    def complete_job(self, job_id, result=None):
        """Mark job as completed"""
        self._update_job_status(job_id, JobStatus.COMPLETED, result)
        self.redis.srem(self.processing_key, job_id)
    
    def fail_job(self, job_id, error=None):
        """Mark job as failed"""
        self._update_job_status(job_id, JobStatus.FAILED, error)
        self.redis.srem(self.processing_key, job_id)
    
    def _update_job_status(self, job_id, status, data=None):
        job_data = self.redis.hget(self.jobs_key, job_id)
        if job_data:
            job = json.loads(job_data)
            job["status"] = status.value
            job["finished_at"] = time.time()
            if data:
                job["result"] = data
            self.redis.hset(self.jobs_key, job_id, json.dumps(job))
    
    def _process_delayed_jobs(self):
        """Move ready delayed jobs to main queue"""
        now = time.time()
        ready_jobs = self.redis.zrangebyscore(
            f"{self.queue_key}:delayed", 0, now
        )
        
        if ready_jobs:
            pipeline = self.redis.pipeline()
            for job_id in ready_jobs:
                pipeline.zrem(f"{self.queue_key}:delayed", job_id)
                pipeline.rpush(self.queue_key, job_id)
            pipeline.execute()
    
    def get_job_status(self, job_id):
        """Get job status and details"""
        job_data = self.redis.hget(self.jobs_key, job_id)
        return json.loads(job_data) if job_data else None
    
    def get_queue_stats(self):
        """Get queue statistics"""
        return {
            "pending": self.redis.llen(self.queue_key),
            "high_priority": self.redis.llen(f"{self.queue_key}:high"),
            "delayed": self.redis.zcard(f"{self.queue_key}:delayed"),
            "processing": self.redis.scard(self.processing_key)
        }

# Worker implementation
class JobWorker:
    def __init__(self, redis_client, queue_name="default"):
        self.queue = RedisJobQueue(redis_client, queue_name)
        self.handlers = {}
    
    def register_handler(self, job_type, handler_func):
        """Register handler for specific job type"""
        self.handlers[job_type] = handler_func
    
    def start_processing(self):
        """Start processing jobs"""
        print(f"Worker started, waiting for jobs...")
        
        while True:
            try:
                job = self.queue.dequeue(timeout=5)
                if job:
                    self.process_job(job)
            except KeyboardInterrupt:
                print("Worker stopped")
                break
            except Exception as e:
                print(f"Worker error: {e}")
    
    def process_job(self, job):
        """Process a single job"""
        job_id = job["id"]
        job_type = job["type"]
        
        print(f"Processing job {job_id} of type {job_type}")
        
        try:
            if job_type in self.handlers:
                result = self.handlers[job_type](job["data"])
                self.queue.complete_job(job_id, result)
                print(f"Job {job_id} completed successfully")
            else:
                raise ValueError(f"No handler for job type: {job_type}")
                
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            self.queue.fail_job(job_id, str(e))

# Usage example
job_queue = RedisJobQueue(r, "email_queue")
worker = JobWorker(r, "email_queue")

# Define job handlers
def send_email_handler(data):
    email = data["email"]
    subject = data["subject"]
    body = data["body"]
    
    # Simulate email sending
    time.sleep(1)
    print(f"Email sent to {email}: {subject}")
    
    return {"status": "sent", "timestamp": time.time()}

def generate_report_handler(data):
    report_type = data["type"]
    user_id = data["user_id"]
    
    # Simulate report generation
    time.sleep(5)
    print(f"Report {report_type} generated for user {user_id}")
    
    return {"report_url": f"/reports/{uuid.uuid4()}.pdf"}

# Register handlers
worker.register_handler("send_email", send_email_handler)
worker.register_handler("generate_report", generate_report_handler)

# Enqueue jobs
email_job_id = job_queue.enqueue("send_email", {
    "email": "user@example.com",
    "subject": "Welcome!",
    "body": "Thank you for signing up"
})

report_job_id = job_queue.enqueue("generate_report", {
    "type": "monthly_summary",
    "user_id": "user123"
}, delay=300)  # Process in 5 minutes

# Check job status
print(job_queue.get_job_status(email_job_id))
print(job_queue.get_queue_stats())

# Start worker (in production, this would be a separate process)
# worker.start_processing()
```

## Security Best Practices

### Authentication and Authorization

```bash
# Enable authentication
requirepass your_strong_password

# Use ACL (Redis 6+) for fine-grained access control
ACL SETUSER alice on >password123 ~user:* +@read
ACL SETUSER bob on >password456 ~config:* +@all
ACL LIST
```

### Network Security

```bash
# Bind to specific interfaces
bind 127.0.0.1 192.168.1.100

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG "CONFIG_b9c3c8e1f2a"

# Enable TLS
port 0
tls-port 6380
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
tls-ca-cert-file /path/to/ca.crt
```

### Data Protection

```python
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet

class SecureRedisClient:
    def __init__(self, redis_client, encryption_key=None):
        self.redis = redis_client
        self.cipher = Fernet(encryption_key) if encryption_key else None
    
    def secure_set(self, key, value, ttl=None):
        """Set value with optional encryption"""
        if self.cipher:
            value = self.cipher.encrypt(value.encode())
        
        if ttl:
            self.redis.setex(key, ttl, value)
        else:
            self.redis.set(key, value)
    
    def secure_get(self, key):
        """Get value with optional decryption"""
        value = self.redis.get(key)
        if value and self.cipher:
            try:
                value = self.cipher.decrypt(value).decode()
            except:
                # Decryption failed, might be unencrypted data
                pass
        return value
    
    def set_with_integrity(self, key, value, secret_key):
        """Set value with HMAC for integrity checking"""
        signature = hmac.new(
            secret_key.encode(),
            value.encode(),
            hashlib.sha256
        ).hexdigest()
        
        data = {
            "value": value,
            "signature": signature
        }
        
        self.redis.set(key, json.dumps(data))
    
    def get_with_integrity(self, key, secret_key):
        """Get value and verify integrity"""
        data = self.redis.get(key)
        if not data:
            return None
        
        try:
            data = json.loads(data)
            value = data["value"]
            signature = data["signature"]
            
            expected_signature = hmac.new(
                secret_key.encode(),
                value.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if hmac.compare_digest(signature, expected_signature):
                return value
            else:
                raise ValueError("Data integrity check failed")
                
        except (json.JSONDecodeError, KeyError):
            raise ValueError("Invalid data format")

# Usage
encryption_key = Fernet.generate_key()
secure_redis = SecureRedisClient(r, encryption_key)

# Encrypted storage
secure_redis.secure_set("sensitive_data", "credit_card_number", ttl=3600)
decrypted_data = secure_redis.secure_get("sensitive_data")

# Integrity protected storage
secure_redis.set_with_integrity("important_data", "user_balance:1000", "secret_key")
verified_data = secure_redis.get_with_integrity("important_data", "secret_key")
```

## Learning Objectives & Assessment

### Self-Assessment Checklist

Before completing this Redis mastery course, ensure you can:

**Fundamentals:**
□ Install and configure Redis on different platforms  
□ Understand Redis data structures and their use cases  
□ Write efficient Redis commands for CRUD operations  
□ Implement proper key naming conventions  

**Advanced Features:**
□ Configure Redis persistence (RDB vs AOF)  
□ Implement pub/sub messaging patterns  
□ Write and execute Lua scripts  
□ Use transactions and pipelines effectively  

**Scalability & Production:**
□ Set up Redis clustering for horizontal scaling  
□ Configure Sentinel for high availability  
□ Implement proper monitoring and alerting  
□ Apply security best practices  

**Real-world Applications:**
□ Design and implement caching strategies  
□ Build session management systems  
□ Create real-time analytics solutions  
□ Implement rate limiting and job queues  

### Practice Projects

**Project 1: E-commerce Cache Layer**
Build a caching layer for an e-commerce application including:
- Product catalog caching
- User session management
- Shopping cart storage
- Real-time inventory tracking

**Project 2: Social Media Analytics**
Create a real-time analytics system featuring:
- User activity tracking
- Trending content identification
- Real-time leaderboards
- Geographic user distribution

**Project 3: Microservices Communication**
Implement inter-service communication using:
- Event-driven architecture with pub/sub
- Distributed job queue
- Circuit breaker pattern
- Service discovery

**Project 4: Gaming Platform**
Build a gaming platform backend with:
- Player matchmaking queues
- Real-time leaderboards
- Achievement tracking
- Session management

### Additional Resources

**Documentation & References:**
- [Official Redis Documentation](https://redis.io/documentation)
- [Redis Commands Reference](https://redis.io/commands)
- [Redis Modules Hub](https://redis.io/modules)

**Books:**
- "Redis in Action" by Josiah L. Carlson
- "Redis Essentials" by Maxwell Dayvson Da Silva
- "Mastering Redis" by Jeremy Nelson

**Online Courses:**
- Redis University (free courses)
- Pluralsight Redis Path
- Udemy Redis Masterclass

**Tools & Libraries:**
- Redis Desktop Manager (GUI client)
- RedisInsight (official GUI)
- Language-specific Redis clients
- Monitoring tools (RedisLive, Redis Monitor)

**Community & Support:**
- Redis Community Slack
- Stack Overflow Redis tag
- Reddit r/redis
- Redis GitHub repository

## Next Steps

After mastering Redis, consider exploring:

1. **Other In-Memory Databases:**
   - Apache Ignite
   - Hazelcast
   - Memcached
   - Amazon ElastiCache

2. **Related Technologies:**
   - Apache Kafka for event streaming
   - RabbitMQ for message queuing
   - Elasticsearch for search and analytics
   - Apache Cassandra for distributed databases

3. **Advanced Topics:**
   - Redis on Kubernetes
   - Multi-cloud Redis deployments
   - Custom Redis module development
   - Redis performance tuning at scale

Congratulations on completing the Redis Mastery course! You now have the knowledge and skills to effectively use Redis in production environments and build scalable, high-performance applications.
