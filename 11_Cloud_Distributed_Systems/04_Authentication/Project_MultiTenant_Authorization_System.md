# Project: Multi-tenant Authorization System

## Description
Build a system supporting multiple tenants with RBAC and dynamic policy management.

## Example: Multi-tenant RBAC (Python)
```python
tenants = {'tenant1': {'alice': 'admin'}, 'tenant2': {'bob': 'user'}}
def can_access(tenant, user):
    return tenants.get(tenant, {}).get(user) == 'admin'
print(can_access('tenant1', 'alice'))  # True
```
