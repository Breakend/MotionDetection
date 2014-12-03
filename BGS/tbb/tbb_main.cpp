

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "DualSGM.hpp"

void test_ser_vs_par(int num_threads);
void test_serial();
void test_tbb(int num_threads);
double timer(void);

int main(int argc, char *argv[]) 
{
  int num_threads = atoi(argv[1]);

  //double start, finish;
  //double ser_time, par_time, t_exec; 

  test_ser_vs_par(num_threads);

  return 0;
}

void test_ser_vs_par(int num_threads)
{
  double start, finish;
  double ser_time, par_time, speedup; 

  printf("NUM_THREADS = %i \n", num_threads);

  start = timer();
  test_serial();
  finish = timer();
  ser_time = finish - start;
  printf("Done! -- Serial execution time : %.10e (%f) \n", ser_time, ser_time);

  start = timer();
  test_tbb(num_threads);
  finish = timer();
  par_time = finish - start;
  printf("Done! -- TBB execution time : %.10e (%f) \n", par_time, par_time);

  speedup = ser_time / par_time;
  printf("Speedup: %.10e\n", speedup);
}

void test_serial()
{
  std::cout << "Running serial DualSGM \n";

  int start = 01;
  int end = 500;
  char buff[100];

  Mat frame = imread("../Videos/sofa/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);

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
      dsgm.serialUpdateModel(&dst);
  }

}

void test_tbb(int num_threads) 
{
  std::cout << "Running tbb DualSGM \n";

  int start = 01;
  int end = 500;
  char buff[100];

  Mat frame = imread("../Videos/sofa/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);
  dsgm.NUM_THREADS = num_threads;

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
      dsgm.tbbUpdateModel(&dst);
  }
  printf("Time in parallel : %.10e (%f) \n", dsgm.parallel_time, dsgm.parallel_time);
}

double timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}
