# Project: Database Testing Project

## Description
Implement tests for database operations using fixtures for setup/teardown and test complex queries.

## Example
```python
import pytest
import sqlite3

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    yield conn
    conn.close()

def test_insert_user(db):
    db.execute("INSERT INTO users VALUES (?, ?)", (1, "Alice"))
    user = db.execute("SELECT name FROM users WHERE id=1").fetchone()
    assert user[0] == "Alice"
```
