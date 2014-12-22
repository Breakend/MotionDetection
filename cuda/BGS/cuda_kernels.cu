#include "utils.h"

/*
* Specify the size of a THREAD BLOCK of sizze THREAD_SIZE x THREAD_SIZE
*/
#define THREAD_SIZE 11
/*
* Flag whether to use separable gaussian filter or 2D
*/
#define SEPARATED_GAUSSIAN_FILTER 1

/**
* CUDA Kernel for DSGM
*/
__global__
void gaussian_background_kernel(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                       int numRows, int numCols)
{
  const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t c = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t index = r * numCols + c;

  // Make sure you're in memory bounds
  if (index < numRows * numCols)
  {

    float alpha, V;
    int adiff;
    int cdiff;

    float pixel = d_frame[index];
    float ameanpixel = d_amean[index];
    float avarpixel = d_avar[index];
    float cmeanpixel = d_cmean[index];
    float cvarpixel = d_cvar[index];

    adiff = pixel - ameanpixel;
    cdiff = pixel - cmeanpixel;
    // If within some threshold of the absolute background, update
    if(adiff*adiff < 9 * avarpixel){
        alpha = 1.0f / (float)d_aage[index];
        d_amean[index] = (1.0f-alpha) * ameanpixel + (alpha) * pixel;
        adiff = d_amean[index] - pixel;
        V = adiff*adiff;
        d_avar[index] = (1.0f-alpha) * avarpixel + alpha * V;
        d_aage[index]++;
    }
    //otherwise if in some threshold of the candidate, update
    else if(cdiff*cdiff < 9 * cvarpixel){
        alpha = 1.0f / (float)d_cage[index];
        d_cmean[index] = (1.0f-alpha) * cmeanpixel + (alpha) * pixel;
        cdiff = d_cmean[index] - pixel;
        V = cdiff*cdiff;
        d_cvar[index] = (1.0f-alpha) * cvarpixel + alpha * V;
        d_cage[index]++;
    }
    //otherwise reset candidate
    else{      
        d_cmean[index] = pixel;
        d_cvar[index] = 255;
        d_cage[index] = 1;
    }

    //if candidate age is larger
    if(d_cage[index] > d_aage[index]){
      //swap the candidate to the absolute
      d_amean[index] = d_cmean[index];
      d_avar[index] = d_cvar[index];
      d_aage[index] = d_cage[index];

      //reset the candidate model
      d_cmean[index] = pixel;
      d_cvar[index] = 255;
      d_cage[index] = 1;
    }

    adiff = pixel - d_amean[index];

    // Update motion mask
    if (adiff*adiff <= 60) {
        //background
        d_bin[index]= 0;
    } else {
        //foreground
        d_bin[index] = 255;
    }
  }
}

/**
* A call to the CUDA kernel, specify the block size first
*/
void gaussian_background(unsigned char* const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                            size_t numRows, size_t numCols)
{
  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 
  gaussian_background_kernel<<<gridSize, blockSize>>>(d_frame, d_amean, d_cmean, 
                                              d_avar, d_cvar, d_bin, d_aage, d_cage,
                                                numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

/**
* 2D, non-separable Gaussian filter kernel
*/
__global__
void gaussian_filter_kernel(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols){

  const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t c = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t index = r * numCols + c; //the center pixel being blurred

  if (index >= numRows * numCols) return;

  int halfway_point = d_filter_width/2;
  float blurred_pixel = 0.0f;

  // Iterate over 2D Gaussian kernel
  for (int i = -halfway_point; i <= halfway_point; ++i){ 
    for (int j = -halfway_point; j <= halfway_point; ++j){ 
            // get the location of the desired pixel, clamped to borders of the image
            int h = fmin(fmax((float)(r + i), 0.f), (float)(numRows-1)); 
            int w = fmin(fmax((float)(c + j), 0.f), (float)(numCols-1)); 
            int current_pixel_id = w + numCols * h;
            float current_pixel = static_cast<float>(d_frame[current_pixel_id]); 

            // now, get the associated weight in the filter
            current_pixel_id = (i + halfway_point) * d_filter_width + j + halfway_point; 
            float weight = d_gfilter[current_pixel_id]; 
            blurred_pixel += current_pixel * weight; 
        } 
    } 
 
  d_blurred[index] = static_cast<int>(blurred_pixel); 
}

/**
* Median filter CUDA kernel
* NOTE: parts of this were taken from: http://stackoverflow.com/questions/19634328/2d-cuda-median-filter-optimization
*/
__global__
void median_filter_kernel(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols){

    const int size = 9;
    unsigned short surround[9];

    int iterator, i;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = x * numCols + y;   

    // if out of bounds return
    if( (x >= (numRows)) || (y >= numCols) || (x < 0) || (y < 0)) return;

    //if border, don't blur
    if( (x == (numRows - 1)) || (y == numCols - 1) || (x == 0) || (y == 0)){
      d_blurred[index] = d_frame[index];
    }

    // get the surrounding pixels and fill a local array
    iterator = 0;
    for (int r = x - 1; r <= x + 1; r++) {
        for (int c = y - 1; c <= y + 1; c++) {
            surround[iterator] = d_frame[r*numCols+c];
            iterator++;
        }
    }


    // simple sorting
    int middle = (size/2)+1;
    for (i=0; i<=middle; i++) {
        int minval=i;
        for (int l=i+1; l<size; l++){
          if (surround[l] < surround[minval]){
             minval=l;
          }
        } 
        unsigned short temp = surround[i];
        surround[i]=surround[minval];
        surround[minval]=temp;
    }

    // Set to the median value
    d_blurred[index] = surround[middle]; 
}

/**
* Call to the gaussian filter CUDA kernel to specify the blocksize
*/
void gaussian_filter(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 
  gaussian_filter_kernel<<<gridSize, blockSize>>>(d_frame, d_blurred, d_gfilter, 
                                                  d_filter_width, d_filter_height, 
                                                  numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* A separable gaussian filter kernel
*/
__global__
void gaussian_filter_kernel_separable(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols, bool x_direction){

  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  const int index = r * numCols + c; //the center pixel being blurred

  // bounds check
  if ((r >= numRows) || (c >= numCols))
  {
    return;
  }

  int halfway_point = d_filter_size/2;
  unsigned char blurred_pixel = 0;
  int h, w, temp;
  
  //iterate over 1 dimensional gaussian kernel for convolution
  for (int j = -halfway_point; j <= halfway_point; ++j){ 
    //get the desired direction and clamp to borders    
    if(x_direction){
      temp = r+j;
      if(temp > numRows-1) temp = numRows-1;
      else if(temp < 0) temp = 0;
      h = temp; 
      w = c;
    }
    else{
      temp = c+j;
      if(temp > numCols-1) temp = numCols-1;
      else if(temp < 0) temp = 0;

      w = temp; 
      h = r;
    }
    
    size_t current_pixel_id = w + numCols * h;
    float current_pixel = d_frame[current_pixel_id]; 

    // now, get the associated weight in the filter
    current_pixel_id = (j + halfway_point); 
    float weight = d_gfilter[current_pixel_id]; 
    unsigned char t = current_pixel * weight; 
    blurred_pixel += t;
  } 


  d_blurred[index] = blurred_pixel; 
}

/**
* Call to the separable filter
*/
void gaussian_filter_separable(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 
  // once in the x direction
  gaussian_filter_kernel_separable<<<gridSize, blockSize>>>(d_frame, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, true);
  //once in the y
  gaussian_filter_kernel_separable<<<gridSize, blockSize>>>(d_blurred_temp, d_blurred, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, false);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* Call to the median filter
*/
void median_filter(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 
  // once in the x direction
  //default median filter size 3
  median_filter_kernel<<<gridSize, blockSize>>>(d_frame, d_blurred, numRows, numCols);

  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* A sequenced call to either the separable gaussian filter or the 2d filter and a subsequent call
* to the median filter CUDA kernels to run on the GPU with the device memory pointers provided
*/
void gaussian_and_median_blur(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 

  #if SEPARATED_GAUSSIAN_FILTER == 1
  // once in the x direction
  gaussian_filter_kernel_separable<<<gridSize, blockSize>>>(d_frame, d_blurred, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, true);

  //once in the y direction
  gaussian_filter_kernel_separable<<<gridSize, blockSize>>>(d_blurred, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, false);
  #else
  // in this case, also need to make sure the filter is 2d
  gaussian_filter_kernel<<<gridSize, blockSize>>>(d_frame, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, d_filter_size, 
                                                  numRows, numCols);
  #endif

  median_filter_kernel<<<gridSize, blockSize>>>(d_blurred_temp, d_blurred, numRows, numCols);

  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


