
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




__global__ void Conv_x(unsigned char* mono, float* kernel, unsigned char* fil, int length, int width, int size_kernel)
{
   
    int index1 = blockIdx.x * blockDim.x + threadIdx.x;                 //In 1D grid, index1 is the element index of X-filtered matrix
    int index;
    float r = 0;
    float ker = 0;
    int width_fil = width - size_kernel + 1;
    
    if (index1 >= (length * width_fil))   return;                       //Avoid segmentation fault
    
    for (int i = 0; i < size_kernel; i++) {

        index = index1 % width_fil + index1 / width_fil * width + i;    //Get element index in original image matrix from x-filtered element index
        ker = kernel[i];
        r += mono[index] * ker;

    }

    
    fil[index1] = round(r);                                             //Store element into GPU pointer

}

__global__ void Conv_y(unsigned char* fil1, float* kernel, unsigned char* fil2, int length, int width, int size_kernel)
{
   
    int index1 = blockIdx.x * blockDim.x + threadIdx.x;                 //In 1D grid, index1 is the element index of Y-filtered matrix
    int index;
   
    float r = 0;
    float ker = 0;
    int width_fil = width - size_kernel + 1;
    int length_fil = length - size_kernel + 1;
   
    if (index1 >= (length_fil * width_fil))   return;                   //Avoid segment fault
    for (int i = 0; i < size_kernel; i++) {

        index = index1 + i * width_fil;                                 //Get element index of X-filtered matrix from Y-filtered matrix
        ker = kernel[i];
        r += fil1[index] * ker;

    }

    
    fil2[index1] = round(r);                                            //Store element into GPU pointer

}





////////////////Function Declarations/////////////
unsigned char* matrix_read(char* RGB, int size, char color);                //function for reading R,G,B matrix from original image
float* Gaussian_kernel(int sigma , int* kernel_size);                       //function for generating standard Gaussian function for given parameter
unsigned char* convolve_CPU(unsigned char* monochrome, float* k, int kernel_size,int width, int length);    //The CPU-executed convolution function 
unsigned char* convolve_GPU(unsigned char* monochrome, float* k, int kernel_size, int width, int length);    //The GPU-executed convolution function 

                                                                                                             
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
        cout << "File open success!!" << endl;                      //If file is open, prompt message
    }

    string version;
    string comment;

    getline(figure, version);                                       //Reading the first line, which is the version of the image
    if (version == "P6") {
        
        cout << "Version correct! Version is " << version <<"\n";   //If version is the desired version, prompt the version number
    }
    else {
        cout << "Version wrong! Exiting...\n";                      //If version is not the desired version, return function
        return 1;
    }

    do {
        getline(figure, comment);                                   //If the string starts with #, it is the comment, so read until all comment is gone
            
    } while (comment[0] == '#');

    

    // read figure width, length and image color intensity level
    int width = 0;
    int length = 0;
    const char* size_string = comment.c_str();                             //Convert string to const char, which sccanf signature needed
    sscanf(size_string, "%d %d", &width, &length);                         //read integer pattern from the string
    cout << "The width of the figure is " << width << ", and the length is " << length << endl;
    string intensity ;
    getline(figure, intensity);                                             //read intensity level
    cout << "The intensity of the figure is " << intensity << endl;

    int size_mat = sizeof(char) * width * length * 3;                       //The total size of RGB matrix is 3 times of the image size
    char *RGB = (char *)malloc(sizeof(char) * size_mat);                    //allocate space for RGB matrix read from file

    // read data as a block:
    figure.read(RGB, size_mat);

    figure.close();                                         //Close original image file

    unsigned char* R;                                       //unsigned char pointer for Red pixels
    unsigned char* G;                                       //unsigned char pointer for Green pixels
    unsigned char* B;                                       //unsigned char pointer for Blue pixel

    R = matrix_read(RGB,size_mat,'R');                      //Read red pixel from RGB matrix
    G = matrix_read(RGB, size_mat, 'G');                    //Read green pixel from RGB matrix
    B = matrix_read(RGB, size_mat, 'B');                    //Read blue pixel from RGB matrix
 
   
    
    float* k ;                                              //float pointer for Gaussian kernel
    int size_kernel;
    k = Gaussian_kernel(sigma,&size_kernel);                //Get Gaussian Kernel and store it to *k
  

    unsigned char* R_fil_CPU;                                   //pointer for CPU filtered Red pixels
    unsigned char* G_fil_CPU;                                   //pointer for CPU filtered Green pixels
    unsigned char* B_fil_CPU;                                   //pointer for CPU filtered Blue pixels

    unsigned char* R_fil_GPU;                                   //pointer for GPU filtered Red pixels
    unsigned char* G_fil_GPU;                                   //pointer for GPU filtered Green pixels
    unsigned char* B_fil_GPU;                                   //pointer for GPU filtered Blue pixels

    cout << "\n\n" << "Starting CPU Gaussian filtering..." << endl;

    R_fil_CPU = convolve_CPU(R, k, size_kernel, width, length);   //CPU Gaussian filter for Red pixel
    G_fil_CPU = convolve_CPU(G, k, size_kernel, width, length);   //CPU Gaussian filter for Green pixel
    B_fil_CPU = convolve_CPU(B, k, size_kernel, width, length);   //CPU Gaussian filter for Blue pixel

    cout << "\n\n" <<"CPU Gaussian filtering finished!!!" << endl;

    cout << "\n\n" <<"Starting GPU Gaussian filtering..." << endl;

    R_fil_GPU = convolve_GPU(R, k, size_kernel, width, length);   //GPU Gaussian filter for Red pixel
    G_fil_GPU = convolve_GPU(G, k, size_kernel, width, length);   //GPU Gaussian filter for Green pixel
    B_fil_GPU = convolve_GPU(B, k, size_kernel, width, length);   //GPU Gaussian filter for Blue pixel
    
    cout << "\n\n" << "GPU Gaussian filtering finished!!!" << endl;
 
    ////////// Write the filtered pixels to new ppm file ///////
    cout << "\n\n" << "Writing files..." << endl;

    string outputFile_CPU = "hereford256_fil_CPU.ppm";                      //define CPU filtered filename
    string outputFile_GPU = "hereford256_fil_GPU.ppm";                      //define GPU filtered filename

    fstream figure_CPU(outputFile_CPU, ofstream::out | ofstream::binary);   //Open CPU filtered file
    fstream figure_GPU(outputFile_GPU, ofstream::out | ofstream::binary);   //Open GPU filtered file

    string version1 = version + "\n";                                       //Same version number with input, add endline 
    char* version_out = &version1[0];
    string width_out = to_string(width - size_kernel + 1) + " ";            //Modified width in file
    string length_out = to_string(length - size_kernel + 1) + "\n";         //Modified length in file
    char* width_write = &width_out[0];
    char* length_write = &length_out[0];
    string intensity_1 = intensity + '\n';                                  //Same intensity with input, add endline
    char* intensity_out = &intensity_1[0];

    figure_CPU.write( version_out, version1.size());                        //Write version, width, length, indensity for CPU filtered file
    figure_CPU.write( width_write, width_out.size());
    figure_CPU.write( length_write, length_out.size());
    figure_CPU.write( intensity_out, intensity_1.size());

    figure_GPU.write( version_out, version1.size());                        //Write version, width, length, indensity for GPU filtered file
    figure_GPU.write( width_write, width_out.size());
    figure_GPU.write( length_write, length_out.size());
    figure_GPU.write( intensity_out, intensity_1.size());

    size_t size_out = (width - size_kernel + 1) * (length - size_kernel + 1);   //New filtered image size
    

    for (int i = 0; i < size_out; i++) {
        figure_CPU.write((char*)(R_fil_CPU + i), sizeof(unsigned char));    //Write R,G,B pixel iteratively in CPU file
        figure_CPU.write((char*)(G_fil_CPU + i), sizeof(unsigned char));
        figure_CPU.write((char*)(B_fil_CPU + i), sizeof(unsigned char));
    }

    for (int i = 0; i < size_out; i++) {
        figure_GPU.write((char*)(R_fil_GPU + i), sizeof(unsigned char));    //Write R,G,B pixel iteratively in GPU file
        figure_GPU.write((char*)(G_fil_GPU + i), sizeof(unsigned char));
        figure_GPU.write((char*)(B_fil_GPU + i), sizeof(unsigned char));
    }


    figure_CPU.close();                     //Close CPU file
    figure_GPU.close();                     //Close GPU file

          
                           

    return 0;
}






unsigned char* matrix_read(char* RGB, int size, char color) {                                   //Read monochrome matrix from RGB matrix
    
    int size_monochrome = size / 3;                                                             //initialize the size of monochrome, which is 1/3 of the pixel size
    unsigned char* monochrome = (unsigned char*)malloc(size_monochrome * sizeof(char));         //allocate monochrome pointer
    
    int i;                          //i is the initial read position from RGB matrix
    switch (color)                  //The order of RGB matrix is : R, G, B, R, G, B,...etc
    {
        case 'R':
            i = 0;                  //Red color matrix starts from first position
            break;
        case 'G':
            i = 1;                  //Green color matrix starts from second position
            break;
        case 'B':
            i = 2;                  //Blue color matrix starts from third position
            break;
    }
    int inc = 3;                    //increment of monochrome is 3
    int j = 0;                      //initial position of monochrome
    for (i ; i < size ; i+=inc ) {
        
        *(monochrome + j) = unsigned char(RGB[i]);          //Read monochrome matrix from RGB matrix
        j += 1;                                             //increment monochrome pointer
        

    }

    return monochrome;

}

float* Gaussian_kernel(int sigma,int* kernel_size) {        //Generate Gaussian filter kernel
    int k = 6 * sigma;                                      //The length of the kernel, covering 99% of the Gaussian values, which is 3*sigma each side      
    if (k % 2 == 0) k++;                                    //Make k odd which is easier for calculation
    *kernel_size = k;                                       //Return kernel size 
    int mu = (k - 1) / 2;                                   //The mu value
    float* K = (float*)malloc(k * sizeof(float));           //allocate pointer for storing kernel

    for (int i = 0; i < k; i++)
        K[i] = exp(-pow((i - mu), 2) / 2 / sigma / sigma)/(sqrt(2 * sigma * sigma * M_PI));
                                                            //Calculation of discrete Gaussian kernel
    return K;
}


unsigned char* convolve_CPU(unsigned char* monochrome, float* k, int size_kernel,int width,int length) {
    
    unsigned char* monochrome_fil1 = (unsigned char*)malloc( (width - size_kernel + 1) * length * sizeof(unsigned char));
    unsigned char* monochrome_fil2 = (unsigned char*)malloc((width - size_kernel + 1) * (length - size_kernel + 1) * sizeof(unsigned char));
    //allocate the pointer to store the image matrix after X-direction convolution and Y-direction convolution
    
    int width_cut = width - size_kernel + 1;                    //X-direction filtered image width should decrease size by (size_kernel - 1)
    int length_cut = length - size_kernel + 1;                  //Y-direction filtered image length should decrease size by (size_kernel - 1)
    int index = 0;
    float r;
    int index0 = 0;
    float ker = 0;
    
    ////////Convolution on X direction
    for (int row = 0; row < length; row++) {                    //the row index of the matrix , the original and X filtered image have the same row number
        for (int i = 0; i < width_cut; i++){                    //the column index of the matrix
            r = 0;                                              //Empty the sum before every iteration
            for (int j = 0; j < size_kernel; j++) {             //sum of the multiplication of kernel and original image 
                index0 = row * width + i + j;                   //index0 is the element index of the original image
                ker = k[j];
                r += monochrome[index0] * ker;
                
            }
            
            monochrome_fil1[index] = round(r);                  //Assign the result of every element to new pointer iteratively
            index++;
        }

    }

    
    ////////Convolution on Y direction
    
    index = 0;                                                  //Redefine all the loop counters
  
    index0 = 0;
    ker = 0;

    for (int row = 0; row < length_cut; row++) {                //The reduced row index of Y-filtered image
        for (int i = 0; i < width_cut; i++) {
            r = 0;                                              //Empty the sum before every iteration
            for (int j = 0; j < size_kernel; j++) {
                index0 = row * width_cut + i + j * width_cut;   //index0 is the element index in X-filtered image
                ker = k[j];
                r += monochrome_fil1[index0] * ker;

            }

            monochrome_fil2[index] = round(r);                  //Assign the result of every element to new pointer iteratively
            index++;
        }

    }

    return monochrome_fil2;

}


unsigned char* convolve_GPU(unsigned char* monochrome, float* k, int size_kernel, int width, int length) {

    int size_monochrome = width * length;                                       //Original size of the monochrome matrix
    int size_fil1 = (width - size_kernel + 1) * length;                         //size of X-filtered image matrix
    int size_fil2 = (width - size_kernel + 1) * (length - size_kernel + 1);     //size of Y-filtered image matrix
    unsigned char* monochrome_fil1 = (unsigned char*)malloc(size_fil1 * sizeof(unsigned char));
    unsigned char* monochrome_fil2 = (unsigned char*)malloc(size_fil2 * sizeof(unsigned char));

    unsigned char* dev_mono = 0;                    //monochrome matrix pointer in GPU device
    float* dev_kernel = 0;                          //Gaussian kernel pointer in GPU device
    unsigned char* dev_fil1 = 0;                    //X-filtered image pointer in GPU device
    unsigned char* dev_fil2 = 0;                    //Y-filtered image pointer in GPU device
    cudaError_t cudaStatus;                         //sentinel variable for observing potential errors


    //Allocate GPU monochrome matrix pointer 
    cudaStatus = cudaMalloc((void**)&dev_mono, size_monochrome * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {                //if failed, prompt error message and return
        fprintf(stderr, "cudaMalloc monochrome failed!");
        exit(1);
    }

    //Allocate GPU Gaussian kernel pointer 
    cudaStatus = cudaMalloc((void**)&dev_kernel, size_kernel * sizeof(float));
    if (cudaStatus != cudaSuccess) {                //if failed, prompt error message and return
        fprintf(stderr, "cudaMalloc kernel failed!");
        exit(1);
    }

    //Allocate GPU X-filtered image matrix pointer 
    cudaStatus = cudaMalloc((void**)&dev_fil1, size_fil1 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {                //if failed, prompt error message and return
        fprintf(stderr, "cudaMalloc dev_fil failed!");
        exit(1);
    }
    //Allocate GPU Y-filtered image matrix pointer 
    cudaStatus = cudaMalloc((void**)&dev_fil2, size_fil2 * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {                //if failed, prompt error message and return
        fprintf(stderr, "cudaMalloc failed!");
        exit(1);
    }

    //Copy monochrome matrix to GPU device
    cudaStatus = cudaMemcpy(dev_mono, monochrome, size_monochrome * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy monochrome failed!");
        exit(1);
    }

    //Copy Gaussian kernel to GPU device
    cudaStatus = cudaMemcpy(dev_kernel, k, size_kernel * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy kernel failed!");
        exit(1);
    }

    
    int BlkNum_fil1 = size_fil1 / 1024 + 1;                 //Calculate block number for x-filtering, using 1024(maximum) threads per block
    int BlkNum_fil2 = size_fil2 / 1024 + 1;                 //Calculate block number for y-filtering, using 1024(maximum) threads per block

    Conv_x <<< BlkNum_fil1, 1024 >>> (dev_mono, dev_kernel, dev_fil1, length, width, size_kernel);      //Run x-filtering kernel first
    Conv_y <<< BlkNum_fil2, 1024 >>> (dev_fil1, dev_kernel, dev_fil2, length, width, size_kernel);      //Run y-filtering kernel afterwards

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

    //Copy x-filtering results back to CPU
    cudaStatus = cudaMemcpy(monochrome_fil1, dev_fil1, size_fil1 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(1);
    }

    //Copy y-filtering results back to CPU
    cudaStatus = cudaMemcpy(monochrome_fil2, dev_fil2, size_fil2 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        exit(1);
    }


    return monochrome_fil2;


}


