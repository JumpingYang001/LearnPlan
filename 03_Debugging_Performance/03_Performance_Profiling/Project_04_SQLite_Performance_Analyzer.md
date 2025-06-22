# Project: SQLite Performance Analyzer

Build a tool to analyze and optimize SQLite usage, including automatic index suggestion.

## Example: SQLite Query Timing in C
```c
#include <sqlite3.h>
#include <stdio.h>
#include <time.h>
int main() {
    sqlite3 *db;
    sqlite3_open("test.db", &db);
    clock_t start = clock();
    sqlite3_exec(db, "SELECT * FROM mytable;", 0, 0, 0);
    clock_t end = clock();
    printf("Query time: %f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    sqlite3_close(db);
}
```
