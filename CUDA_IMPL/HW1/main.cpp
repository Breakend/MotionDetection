//Udacity HW1 Solution

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include "reference_calc.h"
#include "compare.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void your_rgba_to_greyscale(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                            size_t numRows, size_t numCols);
void test_cuda();
//include the definitions of the above functions for this homework
#include "HW1.cpp"


int main(int argc, char *argv[]) 
{
  double start, finish;
  double ser_time, par_time, speedup; 

  // start = timer();
  // test_serial();
  // finish = timer();
  // ser_time = finish - start;
  // printf("Done! -- Serial execution time : %.10e\n", ser_time);

  // 
  // start = timer();
  // test_tbb();
  // finish = timer();
  // par_time = finish - start;
  // printf("Done! -- TBB execution time : %.10e\n", par_time);

  // speedup = ser_time / par_time;
  // printf("Speedup: %.10e\n", speedup);

  test_cuda();

  return 0;
}

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

    //This shouldn't ever happen given the way the images are created
    //at least based upon my limited understanding of OpenCV, but better to check
    if (!frame.isContinuous()) {
      std::cerr << "Images aren't continuous!! Exiting." << std::endl;
      exit(1);
    }
    return frame;
}

void test_cuda(){

  /*
  * 
  */
  cv::namedWindow("origin", CV_WINDOW_AUTOSIZE);


  /*
  * Absolute background
  */ 
  unsigned char *a_mean, *a_variance;
  int **a_age;

  /*
  * Candidate background
  */
  unsigned char *c_mean, *c_variance;
  int **c_age;


  /*
  * The binary image
  */
  unsigned char *binary;

  /*
  * Current Frame
  */
  cv::Mat frame;

  /*
  * Device mem;
  */

  unsigned char *d_frame, *d_bin, *d_amean, *d_avar, *d_cmean, *d_cvar;
  int *d_cage, *d_aage;
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
  cv::Mat am, av, cm, can_v, b;
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

  a_age = (int **) std::calloc(frame.cols, sizeof(int *));
  c_age = (int **) std::calloc(frame.cols, sizeof(int *));

  for (int i = 0; i < frame.cols; ++i) {
      a_age[i] = (int *) std::calloc(frame.rows, sizeof(int));
      c_age[i] = (int *) std::calloc(frame.rows, sizeof(int));
  }
  
  for (int i = 0; i < frame.cols; i++) {
      for (int j = 0; j< frame.rows; j++) {
          a_age[i][j] = 1;
          c_age[i][j] = 1;
          // c_variance[i*frame.cols + j] = 20;
          // a_variance[i*frame.cols + j] = 20;
          // a_mean[i*frame.cols + j] = frame.at<uchar>(i,j);
          // c_mean[i*frame.cols + j] = frame.at<uchar>(i,j);
      }
  }

  while(i < 500){

  //load the image and give us our input and output pointers
  preProcess(&frame, &binary, &a_mean,  &a_variance, a_age, &c_mean, &c_variance, c_age, &d_frame, 
                &d_bin, &d_amean, &d_avar, &d_aage, &d_cmean, &d_cvar, &d_cage);

    GpuTimer timer;
    timer.Start();
    //call the students' code
    your_rgba_to_greyscale(d_frame,d_amean,d_cmean,d_avar,d_cvar, d_bin, d_aage, d_cage,
                            numRows(), numCols());

    timer.Stop();
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
      //Couldn't print! Probably the student closed stdout - bad news
      std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
      exit(1);
    }

    size_t numPixels = numRows()*numCols();
    checkCudaErrors(cudaMemcpy(a_mean, d_amean, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(a_variance, d_avar, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(binary, d_bin, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

    cv::Mat temp = cv::Mat(numRows(), numCols(), CV_8UC1, binary);
    cv::imshow("origin", temp);
    cvWaitKey(1);


    // //check results and output the grey image
    // postProcess(output_file, h_greyImage);

    // referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());

    // postProcess(reference_file, h_greyImage);

    //generateReferenceImage(input_file, reference_file);
    // compareImages(reference_file, output_file, useEpsCheck, perPixelError, 
                  // globalError);

    cleanup();

    sprintf(buff, "../../Videos/sofa/input/in%06d.jpg", i++);
    std::string buffAsStdStr = buff;
    const char * c = buffAsStdStr.c_str();
    frame = readImage(c);
  }
  //END LOOP
  cvDestroyWindow("origin");

}
