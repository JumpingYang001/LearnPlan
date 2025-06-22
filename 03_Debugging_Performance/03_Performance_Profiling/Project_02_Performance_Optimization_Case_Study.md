# Project: Performance Optimization Case Study

Analyze and optimize a real-world application. Document methodology and results.

## Example: Optimizing a Sorting Algorithm in C
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void bubble_sort(int* arr, int n) {
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j)
            if (arr[j] > arr[j+1]) {
                int t = arr[j]; arr[j] = arr[j+1]; arr[j+1] = t;
            }
}
int main() {
    int n = 10000;
    int* arr = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) arr[i] = rand();
    clock_t start = clock();
    bubble_sort(arr, n);
    clock_t end = clock();
    printf("Elapsed: %f s\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(arr);
}
```
