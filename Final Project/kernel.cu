
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mat.h"
#include "matrix.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__host__ void matread(const char* file, unsigned short* v, int* arraySize)
{
    // open MAT-file
    MATFile* pmat = matOpen(file, "r");
    //std::vector<double> v;
    if (pmat == NULL) return;

    // extract the specified variable
    mxArray* arr = matGetVariable(pmat, "Material_Map");
    if (arr != NULL  && !mxIsEmpty(arr)) {
    //if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {
        // copy data
        mwSize num = mxGetNumberOfElements(arr);
        const mwSize *dim = mxGetDimensions(arr);
        
        //std::cout<< "Dimension is : " << *dim << *(dim+1) << *(dim+2);
        *arraySize = num;
        //double *pr = mxGetPr(arr);
        //double* pr = (double*)mxGetData(arr);
        mxUint16* pr = mxGetUint16s(arr);
        //int* pr_int = (int*)pr;
        for (int i = 0 ;i < num;i++) {
            if (*(pr + i) > 0) {
                std::cout << "Index: " << i << std::endl;
                std::cout << "Value: " << *(pr + i) << std::endl;
                break;
            }
            //std::cout << *(pr+i+79917) << " ";
        }

        //double* v = (double *)malloc(num * sizeof(double));
        memcpy(v, pr+79917, 200*sizeof(unsigned short));
        //if (pr != NULL) {
        //    v.reserve(num); //is faster than resize :-)
        //    std::cout << "max size is: " << int(v.capacity()) << std :: endl;
        //    v.assign(pr, pr + num);
        //}
    }

    // cleanup
    mxDestroyArray(arr);
    matClose(pmat);
}





__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__host__ int main()
{
    const char* file = "G:\\Ji Chen's Lab\\Chen's Lab\\ActiveMRIHeating\\Bioheat Equation\\Material_Map.mat";
    unsigned short* v =  (unsigned short*)malloc(sizeof(unsigned short));
    //std::vector<double> v;
    int arraySize;

    matread(file, v, &arraySize);

    for (int i = 0;i < 200;i++) {
        std:: cout << "printing read results: " ;
        std:: cout << *(v+i) << " "<< std:: endl;
        
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
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
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
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
