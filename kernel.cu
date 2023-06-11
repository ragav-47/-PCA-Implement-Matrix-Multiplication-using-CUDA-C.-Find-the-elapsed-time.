#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <device_launch_parameters.h>
#include <windows.h>
#define N 1024

// Host Matrix Multiplication
void matrixMulCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                float a_elem = a[i * n + k];
                float b_elem = b[k * n + j];
                sum += a_elem * b_elem;
            }
            c[i * n + j] = sum;
        }
    }
}

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulGPU(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}
int main() {
    float* h_a, * h_b, * h_c;
    float* d_a, * d_b, * d_c;

    // Allocate memory on host
    h_a = (float*)malloc(N * N * sizeof(float));
    h_b = (float*)malloc(N * N * sizeof(float));
    h_c = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices on host
    for (int i = 0; i < N * N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Allocate memory on device
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    printf("Number of blocks: %dx%d\n", numBlocks.x, numBlocks.y);
    printf("Number of threads per block: %dx%d\n", threadsPerBlock.x, threadsPerBlock.y);

    cudaEvent_t start, stop;
    float elapsedTimeCPU, elapsedTimeGPU;

    // Measure time for CPU Matrix Multiplication
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    matrixMulCPU(h_a, h_b, h_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeCPU, start, stop);
    printf("CPU Time: %f ms\n", elapsedTimeCPU);

    // Measure time for GPU Matrix Multiplication
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    matrixMulGPU << <numBlocks, threadsPerBlock >> > (d_a, d_b, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
    printf("GPU Time: %f ms\n", elapsedTimeGPU);
    // Copy result from device to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    float eps = 1e-5;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += h_a[i * N + k] * h_b[k * N + j];
            }
            float diff = fabs(expected - h_c[i * N + j]);
            if (diff > eps) {
                printf("Verification failed at (%d, %d): expected %f, got %f\n", i, j, expected, h_c[i * N + j]);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("Verification succeeded!\n");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
