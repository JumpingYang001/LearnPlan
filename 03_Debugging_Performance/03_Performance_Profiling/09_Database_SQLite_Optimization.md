# Database and SQLite Optimization

Covers SQLite performance tuning and database access patterns.

## SQLite Indexing Example (C)
```c
#include <sqlite3.h>
sqlite3_exec(db, "CREATE INDEX idx_col ON mytable(col);", 0, 0, 0);
```

## Prepared Statement Example (C)
```c
sqlite3_stmt *stmt;
sqlite3_prepare_v2(db, "SELECT * FROM mytable WHERE col = ?;", -1, &stmt, 0);
sqlite3_bind_int(stmt, 1, 42);
while (sqlite3_step(stmt) == SQLITE_ROW) {
    // process row
}
sqlite3_finalize(stmt);
```
