/*
*
* NOTE: this was taken from the Udacity tutorials for CUDA
* keeping the RGBA stuff for now, in case we decide to work
* with that instead at some point
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat* image;

unsigned char *d_frame__; 
unsigned char *d_amean__;
unsigned char *d_avar__;
int *d_aage__;
unsigned char *d_cmean__;
unsigned char *d_cvar__;
int *d_cage__;
unsigned char *d_bin__;

size_t numRows() { return image->rows; }
size_t numCols() { return image->cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(cv::Mat *frame, unsigned char **a_bin, unsigned char **amean, unsigned char **avar,
                int **aage, unsigned char **cmean, unsigned char **cvar, int **cage, unsigned char **d_frame, 
                unsigned char **d_bin, unsigned char **d_amean, unsigned char **d_avar, int **d_aage,
                unsigned char **d_cmean, unsigned char **d_cvar, int **d_cage) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));
  image = frame;
  unsigned char *image_ptr = frame->ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  //unsigned char stuff
  checkCudaErrors(cudaMalloc(d_frame, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_amean, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_avar, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_cmean, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_cvar, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_bin, sizeof(unsigned char) * numPixels));

  //ages
  checkCudaErrors(cudaMalloc(d_aage, sizeof(int) * numPixels));
  checkCudaErrors(cudaMalloc(d_cage, sizeof(int) * numPixels));

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_frame, image_ptr, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_amean, *amean, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_avar, *avar, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_cmean, *cmean, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_cvar, *cvar, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_bin, *a_bin, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_aage, *aage, sizeof(int) * numPixels, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_cage, *cage, sizeof(int) * numPixels, cudaMemcpyHostToDevice));

  d_frame__ = *d_frame; 
  d_amean__ = *d_amean;
  d_avar__ = *d_avar;
  d_aage__ = *d_aage;
  d_cmean__ = *d_cmean;
  d_cvar__ = *d_cvar;
  d_cage__ = *d_cage;
  d_bin__ = *d_bin;
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
  printf("should show image here");
  //output the image
  // cv::imshow(output);
}

void cleanup()
{
  cudaFree(d_frame__);
  cudaFree(d_amean__);
  cudaFree(d_avar__);
  cudaFree(d_aage__);
  cudaFree(d_cmean__);
  cudaFree(d_cvar__);
  cudaFree(d_cage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}
