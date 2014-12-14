
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "DualSGM.hpp"
#include "Blur.hpp"

// Must be odd blur size
#define BLUR_SIZE 5 


void test_serial();
void serialGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size);
void createGaussianFilter(float *gaussian_filter, int width);
double timer(void);

int main(int argc, char *argv[]) 
{
  //int num_threads = atoi(argv[1]);

  cv::Mat frame = imread("../Videos/badminton/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat destination;
  //serialGaussianBlur(frame, destination, Size(BLUR_SIZE,BLUR_SIZE));
  serialMedianBlur(frame, destination, 9);

  //test_serial();

  // badminton
  // boulevard
  // sofa
  // traffic

  return 0;
}

void test_serial()
{
  std::cout << "Running serial DualSGM \n";

  int start = 01;
  int end = 500;
  char buff[100];

  Mat frame = imread("../Videos/badminton/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);

  for (int i = start + 1; i < end; i++) {
      sprintf(buff, "../Videos/badminton/input/in%06d.jpg", i);
      std::string buffAsStdStr = buff;
      const char * c = buffAsStdStr.c_str();
      frame = imread(c, CV_LOAD_IMAGE_GRAYSCALE);
      Mat destination;
      GaussianBlur( frame, destination, Size(7,7), 0, 0 ); // Size(17,17)
      Mat dst;
      medianBlur( destination, dst, 3 );
      cvWaitKey(1);
      dsgm.serialUpdateModel(&dst);
  }

}

double timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}
