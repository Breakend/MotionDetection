
#ifndef __Blur__
#define __Blur__

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void serialMedianBlur(cv::Mat frame, cv::Mat destination, int size);
void insertionSort(int *window, int length);

void serialGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size);
void createGaussianFilter(float *gaussian_filter, int width);

void tbbMedianBlur(cv::Mat frame, cv::Mat destination, int size, int num_threads);
void tbbGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size, int num_threads);

#endif /* defined(__Blur__) */