
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/video/background_segm.hpp>

#include "DualSGM.hpp"

// Path to images (badminton, boulevard, sofa, traffic)
const char* PATH = "../Videos/sofa/input/";

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
    printf("Missing args. Exiting\n");
    return 0;
  }

  int num_threads = atoi(argv[1]);
  int use_opencv_blur = atoi(argv[2]);
  int do_motion_comp = atoi(argv[3]);

  //generate_histogram(num_threads, use_opencv_blur, do_motion_comp);

  generate_speedup(num_threads, use_opencv_blur, do_motion_comp);

  return 0;
}

void generate_speedup(int num_threads, int use_opencv_blur, int do_motion_comp) 
{
  /*DualSGM::Timing ser_times;
  ser_times.t_exec = 0;
  ser_times.t_blur = 0;
  ser_times.t_mtnc = 0;
  ser_times.t_dsgm = 0;
  ser_times.t_serl = 0;*/

  DualSGM::Timing par_times;
  par_times.t_exec = 0;
  par_times.t_blur = 0;
  par_times.t_mtnc = 0;
  par_times.t_dsgm = 0;
  par_times.t_serl = 0;

  //run_test(0, use_opencv_blur, do_motion_comp, &ser_times);
  run_test(num_threads, use_opencv_blur, do_motion_comp, &par_times);

  //printf("num_threads ser_exec par_exec speedup\n");
  printf("%i %f\n", num_threads, par_times.t_exec);
  //printf("%f %f %f\n",ser_times.t_exec, par_times.t_exec, (ser_times.t_exec / par_times.t_exec));

}

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

  //printf("num_threads t_exec t_blur t_mtnc t_dsgm t_serl\n");
  printf("%i %f %f %f %f %f\n", num_threads, 
    run_times.t_exec, run_times.t_blur, run_times.t_mtnc, run_times.t_dsgm, run_times.t_serl);
}

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

// double start, end, dsgm_proc;
// start = timer();
// dsgm_proc = run_test(num_threads, use_opencv_blur, do_motion_comp);
// end = timer();

// double t_exec = end - start;

// printf("num_threads t_exec t_dsgm\n");
// printf("%i %f %f\n", num_threads, t_exec, dsgm_proc);

/*
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
} */

//cv::Mat frame = imread("../Videos/sofa/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
//cv::Mat destination = frame.clone();

//namedWindow("original", CV_WINDOW_AUTOSIZE);
//namedWindow("blurred", CV_WINDOW_AUTOSIZE);

//serialGaussianBlur(frame, destination, Size(7,7));
//serialMedianBlur(frame, destination, 3);

//imshow("original", frame);
//cvWaitKey(1);
//imshow("blurred", destination);
//cvWaitKey(0);

//cvDestroyWindow("original");
//cvDestroyWindow("blurred");

//run_serial();
//run_tbb(num_threads);
//get_speedup(num_threads); 

//control();

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
