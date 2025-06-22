# Project: Database Access Layer Testing

## Description
Build a database access layer with proper isolation. Use in-memory databases and mocking strategies to verify database operations without a real database.

## Example (Python with SQLite)
```python
import sqlite3
import pytest

def get_user_count(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    return cursor.fetchone()[0]

def test_get_user_count():
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY)')
    conn.execute('INSERT INTO users (id) VALUES (1)')
    assert get_user_count(conn) == 1
```
