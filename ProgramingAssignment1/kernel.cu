
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <math.h>
using namespace std;



__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

////////////////Function Declarations/////////////
unsigned char* matrix_read(char* RGB, int size, char color);
float* Gaussian_kernel(int sigma , int* kernel_size);
unsigned char* convolve(unsigned char* monochrome, float* k, int kernel_size,int width, int length);

/////////////////////////////////////////////////

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
    
    cout << "Reading file...\n";

    if (!(figure.is_open())) {

        cout << "File open failed!!!" << endl;
    }
    else {
        cout << "File open success!!" << endl;
    }


    getline(figure, version);
    if (version == "P6") {
        
        cout << "Version correct! Version is " << version <<"\n";
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

    int size_mat = width * length * 3;
    char *RGB = (char *)malloc(sizeof(char) * size_mat) ;

    // read data as a block:
    figure.read(RGB, size_mat);

    //cout <<  RGB << endl;

    figure.close();

    unsigned char* R;
    unsigned char* G;
    unsigned char* B;

    R = matrix_read(RGB,size_mat,'R');
    G = matrix_read(RGB, size_mat, 'G');
    B = matrix_read(RGB, size_mat, 'B');
 
    /*for (int i = 0; i < 10; i++)
    cout << +R[i] << " " << +G[i] << " " << +B[i] <<" ";*/
    
    float* k ;       
    int kernel_size;
    k = Gaussian_kernel(sigma,&kernel_size);                                         //The Gaussian Kernel
    //cout << "The size of the kernel is " << kernel_size << endl;

    /*for (int i = 0; i < 100; i++) {

        cout << k[i] << " ";
    }*/

    unsigned char* R_fil;
    unsigned char* G_fil;
    unsigned char* B_fil;

    //R_fil = convolve(R, k, kernel_size,width,length);
   //G_fil = convolve(G, k, kernel_size,width,length);
   //B_fil = convolve(B, k, kernel_size,width,length);

     /*for (int i = 0; i < 100; i++) {

        cout << R_fil[i] << " ";
    }*/

     
     string outputFilename = "hereford256_fil.ppm";
     fstream figure_out(outputFilename, ofstream::out | ofstream::binary);
     string version1 = version + "\n";
     char* version_out = &version1[0];
     //string width_out = to_string(width - kernel_size + 1) + " ";
     //string length_out = to_string(length - kernel_size + 1) + "\n";
     string width_out =  "256 ";
     string length_out = "128\n";
     char* width_write = &width_out[0];
     char* length_write = &length_out[0];
     cout << width_out << endl;
     cout << length_out << endl;
     size_t size_width_out = width_out.size();
     size_t size_length_out = length_out.size();
     figure_out.write(version_out, version1.size());
     figure_out.write( width_write, 4);
     figure_out.write( length_write, 4);

     figure_out.close();


     
     
     
     figure.close();
    return 0;
}














unsigned char* matrix_read(char* RGB, int size, char color) {
    
    unsigned char* monochrome = (unsigned char*)malloc(size * sizeof(char));
    int size_RGB = size / 3;
    //int size_RGB = 30;
    int i;
    switch (color)
    {
    case 'R':
        i = 0;
        break;
    case 'G':
        i = 1;
        break;
    case 'B':
        i = 2;
        break;
    }
    int inc = 3;
    int j = 0;
    for (i ; i < size_RGB ; i+=inc ) {
        
        *(monochrome + j) = unsigned char(RGB[i]);
        j += 1;
        

    }

    return monochrome;

}

float* Gaussian_kernel(int sigma,int* kernel_size) {
    int k = 6 * sigma;                  //The length of the kernel, covering 99% of the Gaussian values        
    if (k % 2 == 0) k++;                //Make k odd which is easier for calculation
    *kernel_size = k;
    int mu = (k - 1) / 2;               //The mu value
    float* K = (float*)malloc(k * sizeof(float));

    for (int i = 0; i < k; i++)
        K[i] = exp(-pow((i - mu), 2) / 2 / sigma / sigma)/(sqrt(2 * sigma * sigma * M_PI));

    return K;
}


unsigned char* convolve(unsigned char* monochrome, float* k, int kernel_size,int width,int length) {
    unsigned char* monochrome_fil1 = (unsigned char*)malloc( (width - kernel_size) * length * sizeof(unsigned char));
    unsigned char* monochrome_fil2 = (unsigned char*)malloc((width - kernel_size) * (length - kernel_size) * sizeof(unsigned char));
    int index = 0;
    //int dim1, dim2, flag1, flag2, flag3;
    /*switch (pattern) {
        case('x') :
            dim1 = length;
            dim2 = width;
            flag1 = width;
            flag2 = 1;
            flag3 = 1;
            break;
        case('y'):
            dim1 = width;
            dim2 = length;
            flag1 = 1;
            flag2 = width;
            flag3 = width;
            break;
        default:
            cout << "Convolution Pattern doesn't apply!!\n";
            exit(1);
        
    }*/

    ////////Convolution on X direction
    for (int row = 0; row < length; row++) {
        for (int i = 0; i < width - kernel_size; i++){
        
            float r = 0;
            for (int j = 0; j < kernel_size; j++) {

                r += monochrome[row * width + i + j ] * k[j];
                
            }
            
            monochrome_fil1[index] = round(r);
            index++;
        }

    }
    index = 0;
    ////////Convolution on Y direction
    for (int row = 0; row < width - kernel_size; row++) {
        for (int i = 0; i < length - kernel_size; i++) {

            float r = 0;
            for (int j = 0; j < kernel_size; j++) {

                r += monochrome_fil1[row  + i * width + j * width] * k[j];

            }

            monochrome_fil2[index] = round(r);
            index++;
        }

    }

    return monochrome_fil2;

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