//
//  DualGaussianModel.h
//  BGS
//
//  Created by Peter Henderson on 2014-11-04.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#ifndef __BGS__DualGaussianModel__
#define __BGS__DualGaussianModel__

#include <stdio.h>
//#include <opencv2/video/tracking.hpp>
#include "GaussianModel.h"

//TODO: memory cleanup in destructor
class DualGaussianModel{
public:
    ~DualGaussianModel();

    
    DualGaussianModel(Mat* first_image, int N);
    void updateModel(Mat* next_frame);
    
    float meanThreshold = 9.0; //This constant was taken from the paper, but could probably be varied

    //cv::Mat *last_frame;

private:
    //TODO: change these to auto_ptr or something
    GaussianModel* candidateBackgroundModel;
    void swapPixelsMat(int y, int x);

    GaussianModel* apparentBackgroundModel;

};
#endif /* defined(__BGS__DualGaussianModel__) */
