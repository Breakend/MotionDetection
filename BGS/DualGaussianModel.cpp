//
//  DualGaussianModel.cpp
//  BGS
//
//  Created by Peter Henderson on 2014-11-04.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#include "DualGaussianModel.h"
#include <cmath>

DualGaussianModel::DualGaussianModel(IplImage* first_image, int N){
    candidateBackgroundModel = new GaussianModel(first_image, N);
    apparentBackgroundModel = new GaussianModel(first_image, N); 
    //set up windows
    cvNamedWindow("origin", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("processing", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);

}

void DualGaussianModel::swapPixels(int y, int x){
    CvScalar pixel = cvGet2D(apparentBackgroundModel->frame_u, y, x);
    CvScalar pixel2 = cvGet2D(candidateBackgroundModel->frame_u, y, x);
    cvSet2D(apparentBackgroundModel->frame_u, y, x, pixel2);
    cvSet2D(candidateBackgroundModel->frame_u, y, x, pixel);
    
    pixel = cvGet2D(apparentBackgroundModel->frame_std, y, x);
    pixel2 = cvGet2D(candidateBackgroundModel->frame_std, y, x);
    cvSet2D(apparentBackgroundModel->frame_std, y, x, pixel2);
    cvSet2D(candidateBackgroundModel->frame_std, y, x, pixel);
    
    pixel = cvGet2D(apparentBackgroundModel->frame_var, y, x);
    pixel2 = cvGet2D(candidateBackgroundModel->frame_var, y, x);
    cvSet2D(apparentBackgroundModel->frame_var, y, x, pixel2);
    cvSet2D(candidateBackgroundModel->frame_var, y, x, pixel);
}

void DualGaussianModel::updateModel(IplImage *next_frame){
    //TODO: this should be gridwise instead of pixel wise
    for(int y = 0; y < next_frame->height; ++y)
    {
//        std::cout<< "y: " << y << std::endl;
        for(int x = 0; x < next_frame->width; ++x)
        {
//            std::cout<< "x: " << x << std::endl;
            //TODO: this is wrong, because in updateModel I update the whole frame, should create new thing, update pixel or something.
            for (int i = 0; i<3; i++){
                if (pow(cvGet2D(apparentBackgroundModel->frame, y, x).val[i] - cvGet2D(apparentBackgroundModel->frame_u, y, x).val[i], 2) < meanThreshold * cvGet2D(apparentBackgroundModel->frame_std, y, x).val[i]){
                    apparentBackgroundModel->updatePixel(next_frame, y, x, i);
                }
                else if (pow(cvGet2D(candidateBackgroundModel->frame, y, x).val[i] - cvGet2D(candidateBackgroundModel->frame_u, y, x).val[i], 2) < meanThreshold * cvGet2D(candidateBackgroundModel->frame_std, y, x).val[i]){
                    candidateBackgroundModel->updatePixel(next_frame, y, x, i);
                    
                }else{
                    candidateBackgroundModel->setPixel(next_frame, y, x);
//                    //TODO: replace this with N when not per pixel
//                    GaussianModel * oldCandidateModel = candidateBackgroundModel;
//                    candidateBackgroundModel = new GaussianModel(next_frame, 1);
//                    delete oldCandidateModel;
                }
            }
            if (candidateBackgroundModel->ages[x][y] > apparentBackgroundModel->ages[x][y] ){
                //Swap the models
                swapPixels(y,x);
                candidateBackgroundModel->setPixel(next_frame, y, x);
            }
            
        }
    }
    
    apparentBackgroundModel->updateBinary(next_frame);

    cvShowImage("origin", next_frame);
    cvWaitKey(1);
    cvShowImage("processing", apparentBackgroundModel->frame_u);
    cvWaitKey(1);
    cvShowImage("result", apparentBackgroundModel->frame_bin);
    cvWaitKey(1);
}

DualGaussianModel::~DualGaussianModel(){
    delete apparentBackgroundModel;
    delete candidateBackgroundModel;
    
    cvDestroyWindow("origin");
    cvDestroyWindow("processing");
    cvDestroyWindow("result");
}