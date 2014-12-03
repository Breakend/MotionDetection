// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

__global__
void rgba_to_greyscale(unsigned char * const d_frame,
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
    if(adiff*adiff < 9 * avarpixel){
        alpha = 1.0f / (float)d_aage[index];
        d_amean[index] = (1.0f-alpha) * ameanpixel + (alpha) * pixel;
        adiff = d_amean[index] - pixel;
        V = adiff*adiff;
        d_avar[index] = (1.0f-alpha) * avarpixel + alpha * V;
        d_aage[index]++;
    }
    else if(cdiff*cdiff < 9 * cvarpixel){
        alpha = 1.0f / (float)d_cage[index];
        d_cmean[index] = (1.0f-alpha) * cmeanpixel + (alpha) * pixel;
        cdiff = d_cmean[index] - pixel;
        V = cdiff*cdiff;
        d_cvar[index] = (1.0f-alpha) * cvarpixel + alpha * V;
        d_cage[index]++;
    }
    else{      
        d_cmean[index] = pixel;
        d_cvar[index] = 255;
        d_cage[index] = 1;
    }

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

    //this should be related to theta_d and variance theta_d * pixel_var.val[0]
    if (adiff*adiff <= 60) {
        //background
        d_bin[index]= 0;
    } else {
        //foreground
        d_bin[index] = 255;
    }
  }
}

void your_rgba_to_greyscale(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                            size_t numRows, size_t numCols)
{
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  int THREAD_SIZE = 1;
  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numRows / THREAD_SIZE + 1, numCols / THREAD_SIZE + 1, 1); 
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_frame, d_amean, d_cmean, 
                                              d_avar, d_cvar, d_bin, d_aage, d_cage,
                                                numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
