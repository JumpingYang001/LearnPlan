# Project: Parallel Image Processing Application

## Description
Build an image processing application using OpenMP. Implement different image filters and transformations. Optimize for performance and scalability.

## Example Code
```c
// Example: Parallel grayscale filter
#include <omp.h>
#include <stdio.h>
#define WIDTH 1024
#define HEIGHT 768
unsigned char image[HEIGHT][WIDTH][3];
unsigned char gray[HEIGHT][WIDTH];

void grayscale() {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            gray[y][x] = (image[y][x][0] + image[y][x][1] + image[y][x][2]) / 3;
        }
    }
}
```
