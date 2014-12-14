
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "DualSGM.hpp"


// Path to images (badminton, boulevard, sofa, traffic)
const char* PATH = "../Videos/sofa/input/";

const int start = 01;
const int end = 500;


void get_speedup(int num_threads);
void run_serial();
double run_tbb(int num_threads);
void control();
const char* nextImagePathForIndex(int i);
double timer(void);

int main(int argc, char *argv[]) 
{  
  if (argc != 2) {
    printf("Missing 1 arg. Exiting\n");
    return 0;
  }

  int num_threads = atoi(argv[1]);

  //cv::Mat frame = imread("../Videos/badminton/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
  //cv::Mat destination;
  //serialGaussianBlur(frame, destination, Size(BLUR_SIZE,BLUR_SIZE));
  //serialMedianBlur(frame, destination, 9);

  //run_tbb(num_threads);
  get_speedup(num_threads); 

  //control();

  return 0;
}


void get_speedup(int num_threads)
{
  double ser_start, ser_end;
  ser_start = timer();
  run_serial();
  ser_end = timer();

  double par_start, par_end, par_proc;
  par_start = timer();
  par_proc = run_tbb(num_threads);
  par_end = timer();

  double ser_exec = ser_end - ser_start;
  double par_exec = par_end - par_start;

  //printf("Serial time = %f \n", ser_exec);
  //printf("Parallel time = %f \n", par_exec);
  //printf("Speedup = %f \n", (ser_exec / par_exec)); 
  printf("%f %f %f\n",ser_exec, par_exec, (ser_exec / par_exec));
}

void run_serial() 
{
  Mat frame = imread(nextImagePathForIndex(1),  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);
  for (int i = start + 1; i < end; i++) {
      frame = imread(nextImagePathForIndex(i), CV_LOAD_IMAGE_GRAYSCALE);
      dsgm.serialUpdateModel(&frame);
  }
}

double run_tbb(int num_threads)
{
  double par_proc = 0;
  Mat frame = imread(nextImagePathForIndex(1),  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);
  for (int i = start + 1; i < end; i++) {
      frame = imread(nextImagePathForIndex(i), CV_LOAD_IMAGE_GRAYSCALE);
      par_proc += dsgm.tbbUpdateModel(&frame, num_threads);
  }
  return par_proc;
} 


void control()
{
  std::cout << "Running serial DualSGM control \n";
  Mat frame = imread(nextImagePathForIndex(1),  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);
  for (int i = start + 1; i < end; i++) {
      frame = imread(nextImagePathForIndex(i), CV_LOAD_IMAGE_GRAYSCALE);
      dsgm.serialUpdateModel(&frame);
  }
}

const char* nextImagePathForIndex(int i) 
{
  char buff[100];
  sprintf(buff, "%sin%06d.jpg", PATH, i);
  std::string buffAsStdStr = buff;
  return buffAsStdStr.c_str();
} 

double timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}
