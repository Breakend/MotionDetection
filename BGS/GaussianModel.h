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
    GaussianModel(IplImage* first_image, int N);
    ~GaussianModel();
    void updateModel(IplImage * next_frame);

private:
    /*
     Pixel scalars for calculations
     */
    //original values of pixels
    CvScalar pixel = {0};
    //expected values of pixels
    CvScalar pixel_u = {0};
    //variance images
    CvScalar pixel_var = {0};
    //standard deviation images
    CvScalar pixel_std = {0};
    CvScalar pixel_for = {255, 0, 0,0};
    CvScalar pixel_back = {0};
    
    /*
     Frames
     */
    //original images of video
    IplImage* frame = NULL;
    //expected images of video
    IplImage* frame_u = NULL;
    //variance images
    IplImage* frame_var = NULL;
    //deviation images
    IplImage* frame_std = NULL;
    //binary images
    IplImage* frame_bin = NULL;
    //difference images
    IplImage* frame_diff = NULL;
    
    /*
     Constants
     TODO: capitalize them
     */
    const double e = 2.7183;
    const double segama = 30; //is this supposed to be sigma?
    const double pi = 3.14;
    const int numFrame = 10;
    const int numInitial = 1;
    const int numMRF = 4;
    const int num_write = 500;
    const double threshold = 0.65;
    
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
