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

//TODO: include other params as another constructor (initial std, etc.)
GaussianModel::GaussianModel(IplImage* first_image, int N){
    
    //set up windows
    cvNamedWindow("origin", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("processing", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
    
    //Initialize params
    frame = first_image;
    frame_u = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
    frame_var = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
    frame_std = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
    frame_diff = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
    frame_bin = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    
    alpha = 0.05;
    // initial std?
    std_init = 20;
    //initialized var
    var_init = std_init * std_init;
    lamda = 2.5 * 1.2;
    
    // TODO: this should actually be y = min(y+N, frame->height) and should update according to the sum as in the paper,
    // want to test per pixel Gaussian first
    for( int y = 0; y < frame->height; ++y)
    {
        for( int x = 0; x < frame->width; ++x)
        {
            pixel = cvGet2D(frame, y, x);
            pixel_u.val[0] = pixel.val[0];
            pixel_u.val[1] = pixel.val[1];
            pixel_u.val[2] = pixel.val[2];
            pixel_std.val[0] = std_init;
            pixel_std.val[1] = std_init;
            pixel_std.val[2] = std_init;
            pixel_var.val[0] = var_init;
            pixel_var.val[1] = var_init;
            pixel_var.val[2] = var_init;
            cvSet2D(frame_u, y, x, pixel_u);
            cvSet2D(frame_var, y, x, pixel_var);
            cvSet2D(frame_std, y, x, pixel_std);
        }
    }
    
}

void GaussianModel::updateModel(IplImage * next_frame){
    //TODO: learning rate alpha should be replaced here by age.
    time_t TimeStart, TimeEnd, TimeUsed;
    cvShowImage("origin", next_frame);

    TimeStart = clock();
    //Guassian model
    for(int y = 0; y < next_frame->height; ++y)
    {
        for(int x = 0; x < next_frame->width; ++x)
        {
            pixel = cvGet2D(next_frame, y, x);
            pixel_u = cvGet2D(frame_u, y, x);
            pixel_std = cvGet2D(frame_std, y, x);
            pixel_var = cvGet2D(frame_var, y, x);
            //if|I-u| < lamda*std, background, update
            if(fabs(pixel.val[0]-pixel_u.val[0]) < lamda * pixel_std.val[0] &&
                 fabs(pixel.val[1]-pixel_u.val[1]) < lamda * pixel_std.val[1] &&
                 fabs(pixel.val[2]-pixel_u.val[2]) < lamda * pixel_std.val[2])
            {
                //update u = (1-alpha)*u + alpha*I
                pixel_u.val[0] = (1-alpha) * pixel_u.val[0] + alpha * pixel.val[0];
                pixel_u.val[1] = (1-alpha) * pixel_u.val[1] + alpha * pixel.val[1];
                pixel_u.val[2] = (1-alpha) * pixel_u.val[2] + alpha * pixel.val[2];
                //update var = (1-alpha)*var + alpha*(I-u)^2
                pixel_var.val[0] = (1-alpha) * pixel_var.val[0]
                                                + (pixel.val[0]-pixel_u.val[0]) * (pixel.val[0]-pixel_u.val[0]);
                pixel_var.val[1] = (1-alpha) * pixel_var.val[1] + (pixel.val[1]-pixel_u.val[1]) * (pixel.val[1]-pixel_u.val[1]);
                pixel_var.val[2] = (1-alpha) * pixel_var.val[2] + (pixel.val[2]-pixel_u.val[2]) * (pixel.val[2]-pixel_u.val[2]);
                //update deviation
                pixel_std.val[0] = sqrt(pixel_var.val[0]);
                pixel_std.val[1] = sqrt(pixel_var.val[1]);
                pixel_std.val[2] = sqrt(pixel_var.val[2]);
                
                //write into matrix
                cvSet2D(frame_u, y, x, pixel_u);
                cvSet2D(frame_var, y, x, pixel_var);
                cvSet2D(frame_std, y, x, pixel_std);
            }
        }
    }
    
    /*
     Don't reall get what this is for?
     */
    cvAbsDiff(frame_u, next_frame, frame_diff);
    for( int y = 0; y < next_frame->height; ++y)
    {
        for(int x = 0; x < next_frame->width; ++x)
        {
            if ((frame_diff->imageData + y * frame_diff->widthStep)[3 * x] > 20 &&
                 (frame_diff->imageData + y * frame_diff->widthStep)[3 * x + 1] > 20 &&
                 (frame_diff->imageData + y * frame_diff->widthStep)[3 * x + 2] > 20 )
                cvSet2D(frame_bin, y, x, pixel_for);
            else
                cvSet2D(frame_bin, y, x, pixel_back);
        }
    }
    
    TimeEnd = clock();
    TimeUsed = TimeEnd-TimeStart;
    std::cout<<"Time To Process Frame: " << TimeUsed << std::endl; 
    
    cvShowImage("processing", frame_u);
    cvShowImage("result", frame_bin);
}

GaussianModel::~GaussianModel(){
    cvReleaseImage(&frame_u);
    cvReleaseImage(&frame_var);
    cvReleaseImage(&frame_bin);
    cvReleaseImage(&frame_diff);
    cvReleaseImage(&frame_std);
    cvDestroyWindow("origin");
    cvDestroyWindow("processing");
    cvDestroyWindow("result");
}
