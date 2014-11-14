//
//  GuassianModel.cpp
//  BGS
//
//  Created by Peter Henderson on 2014-11-02.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#include "GaussianModel.h"
//#include <stdlib.h>
// Here we're going to define the Gaussian model for the background of an image
// There will be 1 gaussian model per a small N X N grid subsection of the overall image
// In the paper, this is G_i^t

//using namespace std;

GaussianModel::GaussianModel(Mat* first_image, int N){
    
    
    //Initialize params
    frame_mat = first_image;
    frame_u_mat = new cv::Mat(first_image->size(), CV_8U);
    frame_var_mat = new cv::Mat(first_image->size(), CV_8U);
//    frame_std_mat = new cv::Mat(first_image->size(), CV_8U);
//    frame_diff_mat = new cv::Mat(first_image->size(), CV_8U);
    frame_bin_mat = new cv::Mat(first_image->size(), CV_8U);
    
    age = 1;
    ages = (int **) std::calloc(frame_mat->cols, sizeof(int *));
    
    for(int i = 0; i< frame_mat->cols; ++i) {
        ages[i] =  (int *) std::calloc(frame_mat->rows, sizeof(int));
    }
    
    for(int i = 0; i < frame_mat->cols;i++){
        for(int j = 0; j<frame_mat->rows;j++){
            ages[i][j] = 1;
        }
    }
    
    alpha = 0.05;
    // initial std?
    std_init = 20.0;
    //initialized var
    var_init = std_init * std_init;
    
    // TODO: this should actually be y = min(y+N, frame->height) and should update according to the sum as in the paper,
    // want to test per pixel Gaussian first
    for( int y = 0; y < frame_mat->rows; ++y)
    {
        for( int x = 0; x < frame_mat->cols; ++x)
        {
            setPixel(frame_mat, y, x);
        }
    }
    
}

void GaussianModel::init(Mat * next_frame){

    // TODO: this should actually be y = min(y+N, frame->height) and should update according to the sum as in the paper,
    // want to test per pixel Gaussian first
    for( int y = 0; y < next_frame->rows; ++y)
    {
        for( int x = 0; x < next_frame->cols; ++x)
        {
            setPixel(next_frame, y, x);
        }
    }

}

/**
 Trying with greyscale
 */

void GaussianModel::updatePixel(Mat * next_frame, int y, int x){
    pixel = next_frame->at<uchar>(y, x);
    pixel_u = frame_u_mat->at<uchar>(y, x);
    pixel_var = frame_var_mat->at<uchar>(y, x);
    
    int i = 0;
    alpha = 1.0/(double)ages[x][y];
//    alpha = .07;
    pixel_u.val[i] = (1.0-alpha) * pixel_u.val[i] + (alpha) * pixel.val[i];
    pixel_var.val[i] = (1.0-alpha) * pixel_var.val[i] + (alpha) * pow((pixel_u.val[i] - pixel.val[i]),2);
    
    //write into matrix
    frame_u_mat->at<uchar>(y, x) = pixel_u.val[i];
    frame_var_mat->at<uchar>(y, x) = pixel_var.val[i];
    
    ages[x][y]++;
}

void GaussianModel::updateBinary(Mat * next_frame){
    for( int y = 0; y < next_frame->rows; ++y)
    {
        for(int x = 0; x < next_frame->cols; ++x)
        {
            cv::Scalar  pixel_diff = frame_u_mat->at<uchar>(y, x) - next_frame->at<uchar>(y, x);
            cv::Scalar pixel_var = frame_var_mat->at<uchar>(y, x);

            if(pow(pixel_diff.val[0], 2) <= theta_d * pixel_var.val[0]){
                //background
                frame_bin_mat->at<uchar>(y, x) = 0;
            }
            else{
                //foreground
                frame_bin_mat->at<uchar>(y, x) = 255;
            }
        }
    }
}
void GaussianModel::setPixel(Mat *frame, int y, int x){
        pixel = frame->at<uchar>(y, x);
        pixel_u.val[0] = pixel.val[0];
        pixel_var.val[0] = var_init;
//    frame_var_mat->at<uchar>(y, x) = var_init;

    frame_u_mat->at<uchar>(y, x) = pixel_u.val[0];
    frame_var_mat->at<uchar>(y, x) = pixel_var.val[0];
//    frame_std_mat->at<uchar>(y, x) = pixel_std.val[0];
//    ages[x][y]++;
}

GaussianModel::~GaussianModel(){


}
