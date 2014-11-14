//
//  GuassianModel.h
//  BGS
//
//  Created by Peter Henderson on 2014-11-02.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#ifndef __BGS__GaussianModel__
#define __BGS__GaussianModel__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <stdlib>
#include <ctime>

using namespace cv;
//using namespace std;
/*!
 @class Gaussian Model
 @discussion This keeps track of the gaussian model of a video, note, some of the implementation may have been taken from 
              http://iss.bu.edu/data/jkonrad/reports/HWYY10-06buece.pdf for now, but will probably change later for optimization
 */
class GaussianModel{
public:
    /*!
     Generates a new guassian model which can be updated with values
     \param first_image the first frame 
     \param N the size of the
     */
    GaussianModel(Mat* first_image, int N);
    void init(Mat * next_frame);
    ~GaussianModel();

    void updatePixel(Mat * next_frame, int y, int x);
    void updateBinary(Mat * next_frame);
    void setPixel(Mat *frame, int y, int x);
    
    /**
     For greyscale
     */
    //original images of video
    Mat* frame_mat = NULL;
    //expected images of video
    Mat* frame_u_mat = NULL;
    //variance images
    Mat* frame_var_mat = NULL;
    //deviation images
//    Mat* frame_std_mat = NULL;
    //binary images
    Mat* frame_bin_mat = NULL;
    //Mat images
//    Mat* frame_diff_mat = NULL;
    
    double age;
    int **ages;

private:
    /*
     Pixel scalars for calculations
     */
    //original values of pixels
    cv::Scalar pixel = {0};
    //expected values of pixels
    cv::Scalar pixel_u = {0};
    //variance images
    cv::Scalar pixel_var = {0};
    //standard deviation images
//    cv::Scalar pixel_std = {0};
//    cv::Scalar pixel_for = {255,0,0,0};
//    cv::Scalar pixel_back = {0};
    
    /*
     Constants
     TODO: capitalize them
     */

    const double theta_d = 15.0;
    
    /*
     Other params
     */
    // learning factor?
    double alpha;
    // initial std?
    double std_init;
    //initialized var
    double var_init;
    double lamda;
};

#endif /* defined(__BGS__GuassianModel__) */
