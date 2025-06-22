# Authorization Models

## RBAC, ABAC, PBAC
- RBAC: Role-Based Access Control
- ABAC: Attribute-Based Access Control
- PBAC: Policy-Based Access Control

## Example: Simple RBAC (Python)
```python
roles = {'alice': 'admin', 'bob': 'user'}
def can_access(user, resource):
    if roles.get(user) == 'admin':
        return True
    return False
print(can_access('alice', 'server'))  # True
```
