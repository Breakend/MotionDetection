
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/video/background_segm.hpp>

#include "DualSGM.hpp"

// Specify path to image set 
// (ex: badminton, boulevard, sofa, traffic)
const char* PATH = "../Videos/sofa/input/";

/* Start and end frames */
const int start = 01;
const int end = 500;

void generate_speedup(int num_threads, int use_opencv_blur, int do_motion_comp);
void generate_histogram(int num_threads, int use_opencv_blur, int do_motion_comp);
void run_test(int num_threads, int use_opencv_blur, int do_motion_comp, DualSGM::Timing *run_times);
const char* nextImagePathForIndex(int i);
double timer(void);

int main(int argc, char *argv[]) 
{  
  if (argc != 4) {
    printf("Missing args. Exiting.\n");
    return 0;
  }

  // Parce inputs
  int num_threads = atoi(argv[1]);
  int use_opencv_blur = atoi(argv[2]);
  int do_motion_comp = atoi(argv[3]);

  //generate_histogram(num_threads, use_opencv_blur, do_motion_comp);
  generate_speedup(num_threads, use_opencv_blur, do_motion_comp);

  return 0;
}

/**
 *  Used by run.sh to generate speedup comparisons
 */
void generate_speedup(int num_threads, int use_opencv_blur, int do_motion_comp) 
{

  DualSGM::Timing par_times;
  par_times.t_exec = 0;
  par_times.t_blur = 0;
  par_times.t_mtnc = 0;
  par_times.t_dsgm = 0;
  par_times.t_serl = 0;

  run_test(num_threads, use_opencv_blur, do_motion_comp, &par_times);
  printf("%i %f\n", num_threads, par_times.t_exec);

}

/**
 *  Used by run.sh to generate histograms
 */
void generate_histogram(int num_threads, int use_opencv_blur, int do_motion_comp) 
{
  DualSGM::Timing run_times;
  run_times.t_exec = 0;
  run_times.t_blur = 0;
  run_times.t_mtnc = 0;
  run_times.t_dsgm = 0;
  run_times.t_serl = 0;

  run_test(num_threads, use_opencv_blur, do_motion_comp, &run_times);

  run_times.t_serl = run_times.t_exec - run_times.t_blur - run_times.t_mtnc - run_times.t_dsgm;

  printf("%i %f %f %f %f %f\n", num_threads, 
    run_times.t_exec, run_times.t_blur, run_times.t_mtnc, run_times.t_dsgm, run_times.t_serl);
}

/**
 *  Runs a single DSGM test, completes the Timing struct
 */
void run_test(int num_threads, int use_opencv_blur, int do_motion_comp, DualSGM::Timing *run_times)
{
  double t_exec_start = timer();
  Mat frame = imread(nextImagePathForIndex(1),  CV_LOAD_IMAGE_GRAYSCALE);
  DualSGM dsgm(&frame, 10);
  for (int i = start + 1; i < end; i++) {
    frame = imread(nextImagePathForIndex(i), CV_LOAD_IMAGE_GRAYSCALE);
    dsgm.updateModel(&frame, num_threads, use_opencv_blur, do_motion_comp, run_times);
  }
  run_times->t_exec = timer() - t_exec_start;
}

/**
 *  Retrieves the test image path for an index
 */
const char* nextImagePathForIndex(int i) 
{
  char buff[100];
  sprintf(buff, "%sin%06d.jpg", PATH, i);
  std::string buffAsStdStr = buff;
  return buffAsStdStr.c_str();
} 

/**
 *  Returns the time of day in seconds as a double
 */
double timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}
