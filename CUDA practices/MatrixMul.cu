
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

cudaError_t MulWithCuda(int *c, const int *a, const int *b, unsigned int M, unsigned int N, unsigned int R);

__global__ void MulKernel(int *c, const int *a, const int *b, unsigned int M, unsigned int N, unsigned int R)
{
    size_t i = threadIdx.y;
    size_t j = threadIdx.x;
    if (i >= M || j >= R)return;
    int tem = 0;
    for (size_t n = 0; n < N; n++) {
        
        tem = tem + a[i * N + n] * b[j + n * R];
    }
    c[j + R * i] = tem;
}

int main()
{
    const unsigned int M = 2;
    const unsigned int N = 3;
    const unsigned int R = 4;
    
    const int a[M * N] = { 1, 2, 3, 1, 2, 3};
    const int b[N * R] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    int c[M * R] = { 1,2,3,4,5,6,7,8};

    
    // Add vectors in parallel.
    cudaError_t cudaStatus = MulWithCuda(c, a, b, M, N, R);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MulWithCuda failed!");
        return 1;
    }
    printf("The results of multiplication is :\n");
    for (int i = 0; i < M * R; i++)
        printf("%d\t",c[i]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MulWithCuda(int *c, const int *a, const int *b, unsigned int M, unsigned int N, unsigned int R)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, M * R * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, M * N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * R * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, M * N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * R * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threads(R, M);

    // Launch a kernel on the GPU with one thread for each element.
    MulKernel<<<1, threads>>>(dev_c, dev_a, dev_b, M, N, R);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, M * R * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
