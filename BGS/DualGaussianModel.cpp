//
//  DualGaussianModel.cpp
//  BGS
//
//  Created by Peter Henderson on 2014-11-04.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#include "DualGaussianModel.h"
#include <cmath>


DualGaussianModel::DualGaussianModel(Mat* first_image, int N){
    candidateBackgroundModel = new GaussianModel(first_image, N);
    apparentBackgroundModel = new GaussianModel(first_image, N);
    //set up windows
    namedWindow("origin", CV_WINDOW_AUTOSIZE);
    namedWindow("processing", CV_WINDOW_AUTOSIZE);
    namedWindow("result", CV_WINDOW_AUTOSIZE);
    
}

void DualGaussianModel::swapPixelsMat(int y, int x){
    float pixel = apparentBackgroundModel->frame_u_mat->at<uchar>(y, x);
    float pixel2 = candidateBackgroundModel->frame_u_mat->at<uchar>(y, x);
    apparentBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
    candidateBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
    
    pixel = apparentBackgroundModel->frame_var_mat->at<uchar>(y, x);
    pixel2 = candidateBackgroundModel->frame_var_mat->at<uchar>(y, x);
    apparentBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;
    candidateBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;
}

void DualGaussianModel::updateModel(Mat *next_frame){
    for(int y = 0; y < next_frame->rows; ++y)
    {
        for(int x = 0; x < next_frame->cols; ++x)
        {
            
            // Get the differences for the candidate and apparent background models
            float adiff = next_frame->at<uchar>(y, x) - apparentBackgroundModel->frame_u_mat->at<uchar>(y, x);
            float cdiff = next_frame->at<uchar>(y, x) - candidateBackgroundModel->frame_u_mat->at<uchar>(y, x);
            
            if (pow(adiff, 2) < meanThreshold * apparentBackgroundModel->frame_var_mat->at<uchar>(y, x)){
                apparentBackgroundModel->updatePixel(next_frame, y, x);
            }
            else if(pow(cdiff, 2) < meanThreshold * candidateBackgroundModel->frame_var_mat->at<uchar>(y, x)){
                candidateBackgroundModel->updatePixel(next_frame, y, x);
            }
            else{
//                candidateBackgroundModel->ages[x][y] = 1;
                candidateBackgroundModel->setPixel(next_frame, y, x);
            }
        
            if (candidateBackgroundModel->ages[x][y] > apparentBackgroundModel->ages[x][y] ){
                //Swap the models
                swapPixelsMat(y,x);
//                float temp =apparentBackgroundModel->ages[x][y];
                apparentBackgroundModel->ages[x][y] = candidateBackgroundModel->ages[x][y];
//                candidateBackgroundModel->ages[x][y] = temp;

                candidateBackgroundModel->setPixel(next_frame, y, x);
//                candidateBackgroundModel->ages[x][y] = 1;
            }
            
            
        }
    }
    
    apparentBackgroundModel->updateBinary(next_frame);
    
    cv::imshow("origin", *next_frame);
    cvWaitKey(1);
    imshow("processing", *apparentBackgroundModel->frame_u_mat);
    cvWaitKey(1);
    imshow("result", *apparentBackgroundModel->frame_bin_mat);
    cvWaitKey(1);
}

DualGaussianModel::~DualGaussianModel(){
    delete apparentBackgroundModel;
    delete candidateBackgroundModel;
    
    cvDestroyWindow("origin");
    cvDestroyWindow("processing");
    cvDestroyWindow("result");
}