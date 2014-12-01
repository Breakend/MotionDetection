
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "DualGaussianModel.h"

int main(int argc, char *argv[]) 
{

  int nthreads, tid;

  /* Fork a team of threads giving them their own copies of variables */
  #pragma omp parallel private(nthreads, tid) 
  {

    /* Obtain thread number */
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    /* Only master thread does this */
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }

  }  /* All threads join master thread and disband */

  int start = 01;
  int end = 500;
  char buff[100];

  Mat frame = imread("../Videos/sofa/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  DualGaussianModel gm(&frame, 10);
  
  std::cout << "Running OpenMP DualGaussianModel";

  for (int i = start + 1; i < end; i++) {
      sprintf(buff, "../Videos/sofa/input/in%06d.jpg", i);
      std::string buffAsStdStr = buff;
      const char * c = buffAsStdStr.c_str();
      frame = imread(c, CV_LOAD_IMAGE_GRAYSCALE);
      Mat destination;
      GaussianBlur( frame, destination, Size(9,9), 0, 0 );
      Mat dst;
      medianBlur ( destination, dst, 3 );
      cvWaitKey(1);
      gm.updateModel(&dst);
  }
    
  std::cout << "Done!";
  return 0;


}