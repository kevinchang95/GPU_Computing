
#define _USE_MATH_DEFINES                           //Use math constant Pi

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>                                   //string file is for introducing the variabe type String
#include <iostream>
#include <stdio.h>
#include <fstream>                                  //fstream is the stream for file I/O
#include <sstream>
#include <cmath>                                    //cmath and math.h is for exponential and other math functions
#include <math.h>
using namespace std;                                //Using namespace, in simpliflication of call functions 



__global__ void Kernel(unsigned char *b, unsigned char *a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = a[i] ;

}


__global__ void Conv_x(unsigned char* mono, unsigned char* kernel, unsigned char* fil, int length, int width, int size_kernel)
{
    //size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    //size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index1 = blockIdx.x * blockDim.x + threadIdx.x;
    int index;
    //int index1;
    float r = 0;
    float ker = 0;
    int width_fil = width - size_kernel + 1;
    //if (row >= length || column >= width_fil)   return;
    if (index1 >= (length * width_fil))   return;
    for (int i = 0; i < size_kernel; i++) {
        
        index = index1 % width_fil + index1 / width_fil * width + i;
        ker = kernel[i];
        r += mono[index] * ker;

    }


    //index1 = row * width_fil + column;
    fil[index1] = round(r);
    

}



////////////////Function Declarations/////////////
unsigned char* matrix_read(char* RGB, int size, char color);                //function for reading R,G,B matrix from original image
float* Gaussian_kernel(int sigma , int* kernel_size);                       //function for generating standard Gaussian function for given parameter
unsigned char* convolve_GPU(unsigned char* monochrome, float* k, int kernel_size,int width, int length);    //The CPU-executed convolution function 

/////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cout << "Argument number error!";
        return 1;                                       //Return if argument number is not 3
    }
    string filename(argv[1]);                           //the 2nd argument is the figure to be processed
    
    
    int sigma = atoi(argv[2]);                          //the 3rd argument is the sigma value for filter

    cout << "The filename for the figure is " << filename << endl;
    cout << "The sigma value for the filter is " << sigma << endl << endl;

   
    fstream figure(filename , ifstream::in|ifstream::binary);       //Open file using fstream
    
    cout << "Reading file...\n";                                    //Prompt in command window

    if (!(figure.is_open())) {

        cout << "File open failed!!!" << endl;                      //Check if file is open, if not, then return function
        exit(1);
    }
    else {
        cout << "File open success!!" << endl;
    }

    string version;
    string comment;

    getline(figure, version);                                       //Reading the first line, which is the version of the image
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

    int size_mat = sizeof(char) * width * length * 3;
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
    int size_kernel;
    k = Gaussian_kernel(sigma , &size_kernel);                                         //The Gaussian Kernel
    //cout << "The size of the kernel is " << kernel_size << endl;

    /*for (int i = 0; i < 100; i++) {

        cout << k[i] << " ";
    }*/

    unsigned char* R_fil;
    unsigned char* G_fil;
    unsigned char* B_fil;

    R_fil = convolve_GPU(R, k, size_kernel, width, length);
    G_fil = convolve_GPU(G, k, size_kernel, width, length);
    B_fil = convolve_GPU(B, k, size_kernel, width, length);

     for (int i = 0; i < 100; i++) {

        cout << R_fil[i] << G_fil[i] << B_fil[i];
    }

     
     string outputFilename = "hereford256_GPU_conv_x.ppm";
     fstream figure_out(outputFilename, ofstream::out | ofstream::binary);
     string version1 = version + "\n";
     char* version_out = &version1[0];
     string width_out = to_string(width - size_kernel + 1) + " ";
     //string width_out = to_string(width ) + " ";
     //string length_out = to_string(length - size_kernel + 1) + "\n";
     string length_out = to_string(length) + "\n";
     
     char* width_write = &width_out[0];
     char* length_write = &length_out[0];
     //cout << width_out << endl;
     //cout << length_out << endl;
     size_t size_width_out = width_out.size();
     size_t size_length_out = length_out.size();
     figure_out.write( version_out, version1.size());
     figure_out.write( width_write, width_out.size());
     figure_out.write( length_write, length_out.size());
     string intensity_1 = intensity + '\n';
     char* intensity_out = &intensity_1[0];
     figure_out.write(intensity_out, intensity_1.size());
     //size_t size_out = (width - size_kernel + 1) * (length - size_kernel + 1);
     size_t size_out = (width - size_kernel + 1) * length;
     //size_t size_out = width  * length;
     
     //figure_out.write(RGB, size_mat);                             //Test the correctness of RGB[] read, correct
   /*  for (int i = 0; i < size_out; i++) {                         
         figure_out.write((char*)(R + i), sizeof(unsigned char));   //Test the correctness of R[],G[] and B[], correct
         figure_out.write((char*)(G + i), sizeof(unsigned char));
         figure_out.write((char*)(B + i), sizeof(unsigned char));
     }*/

     for (int i = 0; i < size_out; i++) {
         figure_out.write((char*)(R_fil + i), sizeof(unsigned char));
         figure_out.write((char*)(G_fil + i), sizeof(unsigned char));
         figure_out.write((char*)(B_fil + i), sizeof(unsigned char));
     }




     figure_out.close();


     

    return 0;
}














unsigned char* matrix_read(char* RGB, int size, char color) {
    
    int size_RGB = size;
    unsigned char* monochrome = (unsigned char*)malloc(size_RGB * sizeof(char));
    
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


unsigned char* convolve_GPU(unsigned char* monochrome, float* k, int size_kernel,int width,int length) {
    
    int size_monochrome = width * length;
    int size_fil1 = (width - size_kernel + 1) * length;
    unsigned char* monochrome_fil1 = (unsigned char*)malloc( size_fil1 * sizeof(unsigned char));
    unsigned char* monochrome_fil2 = (unsigned char*)malloc( size_monochrome * sizeof(unsigned char));
       
    unsigned char* dev_mono = 0;
    unsigned char* dev_kernel = 0;
    unsigned char* dev_fil1 = 0;
    unsigned char* dev_fil2 = 0;
    cudaError_t cudaStatus;
    

    cudaStatus = cudaMalloc((void**) &dev_mono, size_monochrome * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc monochrome failed!");
       
    }

    cudaStatus = cudaMalloc((void**) &dev_kernel, size_kernel * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc kernel failed!");
        
    }

    cudaStatus = cudaMalloc((void**) &dev_fil1, size_fil1 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_fil failed!");

    }

    /*cudaStatus = cudaMalloc((void**)&dev_fil2, (width - size_kernel + 1) * (length - size_kernel + 1) * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");

    }*/

    cudaStatus = cudaMemcpy(dev_mono, monochrome, size_monochrome * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy monochrome failed!");
        exit(1);
    }

    cudaStatus = cudaMemcpy(dev_kernel, k, size_kernel * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy kernel failed!");
        exit(1);
    }
    
    dim3 threads( (width - size_kernel + 1), length);
    int BlkNum = size_monochrome / 1024 + 1;
    int BlkNum_fil1 = size_fil1 / 1024 + 1;


    //Kernel<<< BlkNum, 1024 >>>(dev_mono, dev_kernel);
    Conv_x <<< BlkNum_fil1, 1024 >>> (dev_mono, dev_kernel, dev_fil1, length, width, size_kernel);


    // Check for any errors launching the kernel

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        exit(1);
    }

    cudaStatus = cudaMemcpy(monochrome_fil1, dev_fil1, size_fil1 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(1);
    }

    return monochrome_fil1;


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