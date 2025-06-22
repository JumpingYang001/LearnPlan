# Project: Task-Based Parallel Algorithm Library

## Description
Build a library of common algorithms using OpenMP tasks. Implement sorting, searching, and graph algorithms. Create benchmarking and comparison tools.

## Example Code
```c
// Example: Parallel quicksort with OpenMP tasks
#include <omp.h>
#include <stdio.h>

void quicksort(int* arr, int left, int right) {
    if (left < right) {
        int pivot = arr[right];
        int i = left - 1;
        for (int j = left; j < right; j++) {
            if (arr[j] < pivot) {
                i++;
                int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
        }
        int tmp = arr[i+1]; arr[i+1] = arr[right]; arr[right] = tmp;
        int pi = i + 1;
        #pragma omp task shared(arr)
        quicksort(arr, left, pi - 1);
        #pragma omp task shared(arr)
        quicksort(arr, pi + 1, right);
    }
}

void parallel_quicksort(int* arr, int n) {
    #pragma omp parallel
    {
        #pragma omp single
        quicksort(arr, 0, n - 1);
    }
}
```
