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
#include "GaussianModel.h"

//TODO: memory cleanup in destructor
class DualGaussianModel{
public:
    DualGaussianModel(IplImage* first_image, int N);
    ~DualGaussianModel();
    void updateModel(IplImage * next_frame);
    
    
    int meanThreshold = 8; //This constant was taken from the paper, but could probably be varied

private:
    //TODO: change these to auto_ptr or something
    GaussianModel* candidateBackgroundModel;
    void swapPixels(int y, int x);
    GaussianModel* apparentBackgroundModel;

};
#endif /* defined(__BGS__DualGaussianModel__) */
