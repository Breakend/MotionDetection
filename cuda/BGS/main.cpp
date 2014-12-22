#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "compare.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "filter.h"

// Do we use OpenCV's Gaussian and median blurring
#define OPENCV_PROPROCESS 1
// Use separable 1D filter or 2D Gaussian filter
#define SEPARABLE_GAUSSIAN_FILTER 1
// Do we do any preprocessing (blurring) at all?
#define PREPROCESSING 1

/**
* function specifications for CUDA kernel calls
*/ 
void gaussian_background(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                            size_t numRows, size_t numCols);

/*
Two dimensional convolution
*/
void gaussian_filter(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols);

/*
One dimensional, separable convolution
*/
void gaussian_filter_separable(unsigned char* d_frame,
                     unsigned char* d_blurred,
                    unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols);

void median_filter(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols);

void gaussian_and_median_blur(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols);

void test_cuda();



//include the definitions of the above functions for the kernel calls
#include "cuda_mem_functions.cpp"

/*
* timer on the CPU used for measuring preformance
*/
double cpu_timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}


int main(int argc, char *argv[]) 
{

  test_cuda();

  return 0;
}

/**
* Read an image
*/
cv::Mat readImage(const std::string &filename){
    cv::Mat frame;
    /*
    *  READ IN IMAGE
    */
    frame = cv::imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    if (frame.empty()) {
      std::cerr << "Couldn't open file: " << filename << std::endl;
      exit(1);
    }

    //make sure its contiguous in memory
    if (!frame.isContinuous()) {
      std::cerr << "Images aren't continuous in mem! Exiting." << std::endl;
      exit(1);
    }
    return frame;
}

void test_cuda(){
  /*
  * Timing variables
  */
    double t_parallel_s, t_gpu, t_communication_s, t_total_s;
    double t_parallel_f, t_communication_f, t_total_f;
    double t_serial, t_parallel, t_communication, t_total;
    t_total_s = cpu_timer();

  /*
  * create window to display results
  */
  cv::namedWindow("origin", CV_WINDOW_AUTOSIZE);


  /*
  * Absolute background
  */ 
  unsigned char *a_mean, *a_variance;
  int *a_age;

  /*
  * Candidate background
  */
  unsigned char *c_mean, *c_variance;
  int *c_age;


  /*
  * The binary image
  */
  unsigned char *binary;

  /*
  * Current Frame
  */
  cv::Mat frame;

  /*
  * Device mem variables
  */

  unsigned char *d_frame, *d_bin, *d_amean, *d_avar, *d_blurred_temp, *d_cmean, *d_cvar, *d_frame_blurred, *d_frame_to_blur, *blurred_frame;
  int *d_cage, *d_aage;
  float *d_gaussian_filter, *gaussian_filter_vals;


  char buff[100];

  int i = 2;
  std::string input_file = "../../Videos/sofa/input/in000001.jpg";

  frame = readImage(input_file);
  if(!frame.isContinuous()){
    printf("Frame not continuous");
    exit(1);
  }

  /*
  * Initialize the arrays
  */
  cv::Mat am, av, cm, can_v, b, blurred;
  am = frame.clone();
  if(!am.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  a_mean  = am.ptr<unsigned char>(0);
  av.create(frame.rows, frame.cols, CV_8UC1);
  if(!av.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  a_variance  = av.ptr<unsigned char>(0);
  cm = frame.clone();
  if(!cm.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  c_mean  = cm.ptr<unsigned char>(0);
  can_v.create(frame.rows, frame.cols, CV_8UC1);
  if(!can_v.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  c_variance  = can_v.ptr<unsigned char>(0);
  b.create(frame.rows, frame.cols, CV_8UC1);
  if(!b.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  binary  = b.ptr<unsigned char>(0);

  blurred.create(frame.rows, frame.cols, CV_8UC1);
  if(!blurred.isContinuous()){
    printf("am not continuous");
    exit(1);
  }
  blurred_frame  = blurred.ptr<unsigned char>(0);

  a_age = (int *) std::calloc(frame.cols*frame.rows, sizeof(int));
  c_age = (int *) std::calloc(frame.cols*frame.rows, sizeof(int));
  
  for (int i = 0; i < frame.cols*frame.rows; i++) {
      a_age[i] = 1;
      c_age[i] = 1;
  }


  const int BLUR_SIZE = 9;
  const float SIGMA = 5.f;

  /*
  * PREPARE THE GAUSSIAN FILTER
  */
  #if SEPARABLE_GAUSSIAN_FILTER == 1
  cv::Mat temp_kernel = cv::getGaussianKernel(BLUR_SIZE, SIGMA, CV_32F);
  gaussian_filter_vals = temp_kernel.ptr<float>(0);
  #else
  gaussian_filter_vals = create2DGaussianFilter(BLUR_SIZE);
  #endif

  //put it into device memory once.
  #if SEPARABLE_GAUSSIAN_FILTER == 1
    checkCudaErrors(cudaMalloc(&d_gaussian_filter, sizeof(float) * BLUR_SIZE));
    checkCudaErrors(cudaMemcpy(d_gaussian_filter, gaussian_filter_vals, sizeof(float) * BLUR_SIZE, cudaMemcpyHostToDevice));
  #else
    checkCudaErrors(cudaMalloc(&d_gaussian_filter, sizeof(float) * BLUR_SIZE * BLUR_SIZE));
    checkCudaErrors(cudaMemcpy(d_gaussian_filter, gaussian_filter_vals, sizeof(float) * BLUR_SIZE * BLUR_SIZE, cudaMemcpyHostToDevice));
  #endif

  size_t numPixels;

  // Initialize a GPU timer based on events
  GpuTimer timer;

  while(i < 500){

    //BLURRING
    #if PREPROCESSING == 1
      //do we use opencv
      #if OPENCV_PROPROCESS == 1 
        cv::Mat dst, destination;
        cv::GaussianBlur( frame, destination, cv::Size(9,9), 0, 0 );
        cv::medianBlur ( destination, dst, 3 );

        //feed in the opencv result into device memory
        t_parallel_s = cpu_timer();
        //load the image and give us our input and output pointers
        preProcess(&dst, &binary, &a_mean,  &a_variance, &a_age, &c_mean, &c_variance, &c_age, &d_frame, 
                &d_bin, &d_amean, &d_avar, &d_aage, &d_cmean, &d_cvar, &d_cage);
      #else
        //our own GPU median and Gaussian blur
        t_parallel_s = cpu_timer();

        preprocessGaussianBlur(&frame, &d_frame_to_blur, &d_frame_blurred, &d_blurred_temp, BLUR_SIZE);
        timer.Start();
        gaussian_and_median_blur(d_frame_to_blur,
                        d_frame_blurred,
                        d_blurred_temp,
                        d_gaussian_filter,
                        BLUR_SIZE,
                        numRows(), numCols());

        timer.Stop();
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        t_gpu += (timer.Elapsed()/1000);
        size_t numPixels = numRows()*numCols();

        checkCudaErrors(cudaMemcpy(blurred_frame, d_frame_blurred, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

        cv::Mat dst(cv::Size(numCols(), numRows()),CV_8UC1,blurred_frame);

        cleanup_blur();

        t_parallel_f = cpu_timer();
        t_parallel += t_parallel_f - t_parallel_s;

        t_parallel_s = cpu_timer();
        //load the image and give us our input and output pointers
        preProcess(&dst, &binary, &a_mean,  &a_variance, &a_age, &c_mean, &c_variance, &c_age, &d_frame, 
                  &d_bin, &d_amean, &d_avar, &d_aage, &d_cmean, &d_cvar, &d_cage);

      #endif
    #else
      //no filtering
      t_parallel_s = cpu_timer();
      //load the image and give us our input and output pointers
      preProcess(&frame, &binary, &a_mean,  &a_variance, &a_age, &c_mean, &c_variance, &c_age, &d_frame, 
                  &d_bin, &d_amean, &d_avar, &d_aage, &d_cmean, &d_cvar, &d_cage);
    #endif

    /*
    * END PREPROCESSING AND BLURRING
    */

    /*
     See how much time is actually spent in the GPU
    */
    timer.Start();

    // CALL DSGM
    gaussian_background(d_frame,d_amean,d_cmean,d_avar,d_cvar, d_bin, d_aage, d_cage, numRows(), numCols());

    timer.Stop();
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    t_gpu += (timer.Elapsed()/1000);
    numPixels = numRows()*numCols();

    // Load all data back into CPU memory
    checkCudaErrors(cudaMemcpy(a_mean, d_amean, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(a_variance, d_avar, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(binary, d_bin, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(c_mean, d_cmean, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(c_variance, d_cvar, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(a_age, d_aage, sizeof(int) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(c_age, d_cage, sizeof(int) * numPixels, cudaMemcpyDeviceToHost));

    t_parallel_f = cpu_timer();

    t_parallel += t_parallel_f - t_parallel_s;
    cv::Mat temp = cv::Mat(numRows(), numCols(), CV_8UC1, binary);
    cv::imshow("origin", dst);
    cvWaitKey(1);

    //free up memory on the device
    cleanup();

    //get the next frame
    sprintf(buff, "../../Videos/sofa/input/in%06d.jpg", i++);
    std::string buffAsStdStr = buff;
    const char * c = buffAsStdStr.c_str();
    frame = readImage(c);
  }

  //free device memory for the filter
  cudaFree(d_gaussian_filter);

  //END LOOP and destroy the window
  cvDestroyWindow("origin");
  t_total_f = cpu_timer();
  t_total = t_total_f-t_total_s;
  t_serial = t_total-t_parallel;
  
  // print timing information
  printf("%f %f %f %f %f\n", t_total, t_serial, t_parallel, t_gpu, t_parallel-t_gpu);
  // printf("Total Execution time: %f\n", t_total);
  // printf("Serial Execution part: %f\n", t_serial);
  // printf("Parallel Execution part: %f\n", t_parallel);
  // printf("Computation time for GPU: %f\n", t_gpu);
  // printf("Communication time for GPU: %f\n", t_parallel - t_gpu);

}
