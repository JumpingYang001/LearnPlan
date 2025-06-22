# Memcached and Distributed Caching

## Description
Covers Memcached architecture, commands, consistent hashing, sharding, and cache invalidation.

## Topics
- Memcached architecture and commands
- Consistent hashing and sharding
- Cache invalidation strategies
- Distributed caching solutions

## Example Code
```python
# Example: Using python-memcached
import memcache
mc = memcache.Client(['127.0.0.1:11211'])
mc.set('key', 'value')
print(mc.get('key'))  # Output: value
```
