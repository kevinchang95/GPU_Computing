
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "Argument number error!";
        return 1;                    //Return if argument number is not 3
    }
    //string filename = argv[1];                   //the 2nd argument is the figure to be processed
    string filename(argv[1]);
    string version;
    string comment;
    char c;
    //char * c = (char * )malloc(sizeof(char) * 1);
    int sigma = atoi(argv[2]);                  //the 3rd argument is the sigma value for filter

    cout << "The filename for the figure is " << filename << endl;
    cout << "The sigma value for the filter is " << sigma << endl << endl;

   
    fstream figure(filename , ifstream::in|ifstream::binary);

    getline(figure, version);
    if (version == "P6") {
        
        cout << "Version correct!\n";
    }
    else {
        cout << "Version wrong! Exiting...\n";
        return 1;
    }

    do {
        getline(figure, comment);
            
    } while (comment[0] == '#');

    //cout << comment <<endl;

    // read figure width and length
    int width = 0;
    int length = 0;
    const char* size_string = comment.c_str();                             //Convert string to const char, which sccanf signature needed
    sscanf(size_string, "%d %d", &width, &length);                         //read integer pattern from the string
    cout << "The width of the figure is " << width << ", and the length is " << length << endl;
    string intensity ;
    getline(figure, intensity);
    cout << "The intensity of the figure is " << intensity << endl;

    int size = width * length * 3;
    char *RGB = (char *)malloc(sizeof(char) * size) ;

    // read data as a block:
    figure.read(RGB, size);

    cout <<  RGB << endl;

    figure.close();

    matrix_read(RGB, size);

 



    return 0;
}


int* matrix_read(char* RGB, int width, int length) {
    int i = 0;

    int R[width][length], G, B;
    while (i < 12) {
        rem = remainder(i, 3);
        switch (rem) {
        case 0:
            &R = 
        }
        


    }



}



//////////////Stratches//////////////////
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
// // Add vectors in parallel.
//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//if (cudaStatus != cudaSuccess) {
//    fprintf(stderr, "addWithCuda failed!");
//    return 1;
//}
//
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
//{
//    int* dev_a = 0;
//    int* dev_b = 0;
//    int* dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel << <1, size >> > (dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//
//    return cudaStatus;
//}
//
//
//
//const int arraySize = 5;
//const int a[arraySize] = { 1, 2, 3, 4, 5 };
//const int b[arraySize] = { 10, 20, 30, 40, 50 };
//int c[arraySize] = { 0 };
//
//
//
//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//    c[0], c[1], c[2], c[3], c[4]);
//
//// cudaDeviceReset must be called before exiting in order for profiling and
//// tracing tools such as Nsight and Visual Profiler to show complete traces.
//cudaStatus = cudaDeviceReset();
//if (cudaStatus != cudaSuccess) {
//    fprintf(stderr, "cudaDeviceReset failed!");
//    return 1;
//}


 //  while (Figure.get(c)) {
    //getline(Figure, filename);
    //for(int i = 0 ; i < filename.size(); i++){
    //    //Figure.get(c,1024);
      //  cout << c ;
    //    cout << filename[i];
    //    //cout << i << ":" << c << endl;
    //    //break;
    //}

    /*while (getline(Figure, fileline)) {

        cout << fileline << '\t' << "The length is: "<< fileline.length() << endl;

    }

    Figure.close();*/