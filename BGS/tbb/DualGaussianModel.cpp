//
//  DualGaussianModel.cpp
//  BGS
//
//  Created by Peter Henderson on 2014-11-04.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#include "DualGaussianModel.h"
#include <cmath>

/*class Parallel_process : public cv::ParallelLoopBody 
{
    private:
        int size;
        float meanThreshold;
        cv::Mat *next_frame;
        cv::Mat& retVal;
        GaussianModel* candidateBackgroundModel;
        GaussianModel* apparentBackgroundModel;

    public:
        Parallel_process(int num_threads, 
            float meanThresh,
            cv::Mat *inputImgage, 
            cv::Mat& outImage, 
            GaussianModel* candidateBM, 
            GaussianModel* apparentBM)
            : size(num_threads), 
            meanThreshold(meanThresh),
            next_frame(inputImgage), 
            retVal(outImage),
            candidateBackgroundModel(candidateBM), 
            apparentBackgroundModel(apparentBM) {}

        virtual void operator()(const cv::Range& range) const
        {
            for (int i = range.start; i < range.end; i++) {
                // Your code here
                //printf("[%i] with size %i \n", i, size);
            
                int blocking_factor = next_frame->rows / size; 
                int offset = blocking_factor * i;
                int row_limit = offset + blocking_factor; 
                if (i == size - 1) row_limit = next_frame->rows;

                Mat *app_frame_u_mat = apparentBackgroundModel->frame_u_mat;
                Mat *can_frame_u_mat = candidateBackgroundModel->frame_u_mat;
                Mat *app_frame_var_mat = apparentBackgroundModel->frame_var_mat;
                Mat *can_frame_var_mat = candidateBackgroundModel->frame_var_mat;


                for(int y = offset; y < row_limit; ++y)
                {
                    for(int x = 0; x < next_frame->cols; ++x)
                    {
                        
                        // Get the differences for the candidate and apparent background models
                        float adiff = next_frame->at<uchar>(y,x) - app_frame_u_mat->at<uchar>(y,x);
                        float cdiff = next_frame->at<uchar>(y,x) - can_frame_u_mat->at<uchar>(y,x);
                        
                        if (pow(adiff, 2) < meanThreshold * app_frame_var_mat->at<uchar>(y,x)) {
                            apparentBackgroundModel->updatePixel(next_frame, y, x);
                        }
                        else if(pow(cdiff, 2) < meanThreshold * candidateBackgroundModel->frame_var_mat->at<uchar>(y, x)) {
                            candidateBackgroundModel->updatePixel(next_frame, y, x);
                        }
                        else {
                            candidateBackgroundModel->ages[x][y] = 1;
                            candidateBackgroundModel->setPixel(next_frame, y, x);
                        }
                    
                        if (candidateBackgroundModel->ages[x][y] > apparentBackgroundModel->ages[x][y]) {
                            //Swap the models
                            //swapPixelsMat(y,x);

                            float pixel = apparentBackgroundModel->frame_u_mat->at<uchar>(y, x);
                            float pixel2 = candidateBackgroundModel->frame_u_mat->at<uchar>(y, x);
                            apparentBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
                            candidateBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
                            
                            pixel = apparentBackgroundModel->frame_var_mat->at<uchar>(y, x);
                            pixel2 = candidateBackgroundModel->frame_var_mat->at<uchar>(y, x);
                            apparentBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;
                            candidateBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;

                            float temp = apparentBackgroundModel->ages[x][y];
                            apparentBackgroundModel->ages[x][y] = candidateBackgroundModel->ages[x][y];
                            candidateBackgroundModel->ages[x][y] = temp;

                            candidateBackgroundModel->setPixel(next_frame, y, x);
                            candidateBackgroundModel->ages[x][y] = 1;
                        }
                    }
                }
                
                //cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, 
                //           img.cols, img.rows/diff));
                //cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, 
                //                    retVal.cols, retVal.rows/diff));

            }
        }



};*/



DualGaussianModel::DualGaussianModel(Mat* first_image, int N) {
    candidateBackgroundModel = new GaussianModel(first_image, N);
    apparentBackgroundModel = new GaussianModel(first_image, N);
    //set up windows
    namedWindow("origin", CV_WINDOW_AUTOSIZE);
    namedWindow("processing", CV_WINDOW_AUTOSIZE);
    namedWindow("result", CV_WINDOW_AUTOSIZE);
    
    //last_frame = cvCloneMat(first_image);

    /*
    int NUM_THREADS = 4;
    cv::Mat img, out;
    img = cv::imread("~/Desktop/omp/lenna.png"); //argv[1]
    out = cv::Mat::zeros(img.size(), CV_8UC3);
    cv::parallel_for_(cv::Range(0,NUM_THREADS), Parallel_process(NUM_THREADS, img,out,candidateBackgroundModel,apparentBackgroundModel));
    */
}

void DualGaussianModel::swapPixelsMat(int y, int x) {
    float pixel = apparentBackgroundModel->frame_u_mat->at<uchar>(y, x);
    float pixel2 = candidateBackgroundModel->frame_u_mat->at<uchar>(y, x);
    apparentBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
    candidateBackgroundModel->frame_u_mat->at<uchar>(y, x) = pixel2;
    
    pixel = apparentBackgroundModel->frame_var_mat->at<uchar>(y, x);
    pixel2 = candidateBackgroundModel->frame_var_mat->at<uchar>(y, x);
    apparentBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;
    candidateBackgroundModel->frame_var_mat->at<uchar>(y, x) = pixel2;
}

void DualGaussianModel::updateModel(Mat *next_frame) {

    /*cv::Mat img, out;
    img = cv::imread("~/Desktop/omp/lenna.png");
    out = cv::Mat::zeros(img.size(), CV_8UC3);;
    int NUM_THREADS = 4;
    cv::parallel_for_(cv::Range(0,NUM_THREADS), Parallel_process(NUM_THREADS,meanThreshold,next_frame,out,
        candidateBackgroundModel,apparentBackgroundModel));
    */

    // 2.3. Motion Compensation by Mixing Models    
    // 1. Divide into grids 32 × 24, KLT on every corner of the grid
    // 2. RANSAC to obtain a homography matrix Ht:t−1 
    // 3. 

    /*
    cv::Mat m_prevImg;
    cv::Mat m_nextImg;
    std::vector<cv::Point2f>   m_prevPts;
    std::vector<cv::Point2f>   m_nextPts;
    std::vector<unsigned char> m_status;
    std::vector<float>         m_error;
    cv::calcOpticalFlowPyrLK(m_prevImg, m_nextImg, m_prevPts, m_nextPts, m_status, m_error);

    //last_frame = cvCloneMat(next_frame);

    double ransacThreshold = 3;
    // compute homography using RANSAC
    cv::Mat mask;
    cv::Mat H = cv::findHomography(m_prevPts, m_nextPts, CV_RANSAC, ransacThreshold, mask);
    */

    // set the next frame... 
    // next_frame

    // initial std?
    double std_init = 20.0;
    //initialized var
    double var_init = std_init * std_init;

    int i;
    float alpha;
    float V;

    int row_lim = next_frame->rows;
    int col_lim = next_frame->cols;

    /* Mats */
    Mat *bin_mat = apparentBackgroundModel->frame_bin_mat;
    Mat *app_u_mat = apparentBackgroundModel->frame_u_mat;
    Mat *app_var_mat = apparentBackgroundModel->frame_var_mat;
    Mat *can_u_mat = candidateBackgroundModel->frame_u_mat;
    Mat *can_var_mat = candidateBackgroundModel->frame_var_mat;
    int **app_ages = apparentBackgroundModel->ages;
    int **can_ages = candidateBackgroundModel->ages;

    /* Core update loop */
    for (int y = 0; y < row_lim; ++y) {
        for (int x = 0; x < col_lim; ++x) {
            
            cv::Scalar i_sclr       = next_frame->at<uchar>(y,x);
            cv::Scalar app_u_sclr   = app_u_mat->at<uchar>(y,x);
            cv::Scalar app_var_sclr = app_var_mat->at<uchar>(y,x);
            cv::Scalar can_u_sclr   = can_u_mat->at<uchar>(y,x);
            cv::Scalar can_var_sclr = can_var_mat->at<uchar>(y,x);

            // Get the differences for the candidate and apparent background models
            float adiff = i_sclr.val[0] - app_u_sclr.val[0];
            float cdiff = i_sclr.val[0] - can_u_sclr.val[0];
            
            if (pow(adiff, 2) < meanThreshold * app_var_sclr.val[0]){
                //apparentBackgroundModel->updatePixel(next_frame, y, x);
                i = 0;
                alpha = 1.0 / (double)app_ages[x][y];
                app_u_sclr.val[i] = (1.0-alpha) * app_u_sclr.val[i] + (alpha) * i_sclr.val[i];
                V =  pow((app_u_sclr.val[i] - i_sclr.val[i]),2);
                app_var_sclr.val[i] = (1.0-alpha) * app_var_sclr.val[i] + alpha * V;
                
                //write into matrix
                app_u_mat->at<uchar>(y,x) = app_u_sclr.val[i];
                app_var_mat->at<uchar>(y,x) = app_var_sclr.val[i];
                app_ages[x][y]++;

            } else if (pow(cdiff, 2) < meanThreshold * can_var_sclr.val[0]){
                //candidateBackgroundModel->updatePixel(next_frame, y, x);
                i = 0;
                alpha = 1.0 / (double)can_ages[x][y];
                can_u_sclr.val[i] = (1.0-alpha) * can_u_sclr.val[i] + (alpha) * i_sclr.val[i];
                V = pow((can_u_sclr.val[i] - i_sclr.val[i]),2);
                can_var_sclr.val[i] = (1.0-alpha) * can_var_sclr.val[i] + alpha * V;
                
                //write into matrix
                can_u_mat->at<uchar>(y,x) = can_u_sclr.val[i];
                can_var_mat->at<uchar>(y,x) = can_var_sclr.val[i];
                can_ages[x][y]++;

            } else {
                //can_ages[x][y] = 1;
                //candidateBackgroundModel->setPixel(next_frame, y, x);
                
                can_u_mat->at<uchar>(y, x) = i_sclr.val[0];
                can_var_mat->at<uchar>(y, x) = var_init;
                can_ages[x][y] = 1;
            }
        
            if (can_ages[x][y] > app_ages[x][y]) {
                //Swap the models

                app_u_mat->at<uchar>(y,x) = can_u_sclr.val[0];
                app_var_mat->at<uchar>(y,x) = can_var_sclr.val[0];

                float temp = app_ages[x][y];
                app_ages[x][y] = can_ages[x][y];
                can_ages[x][y] = temp;

                //candidateBackgroundModel->setPixel(next_frame, y, x);
                can_u_mat->at<uchar>(y, x) = i_sclr.val[0];
                can_var_mat->at<uchar>(y, x) = var_init;
                can_ages[x][y] = 1;
            }
            

            cv::Scalar app_diff = app_u_mat->at<uchar>(y,x) - next_frame->at<uchar>(y,x);
            //cv::Scalar pixel_var = frame_var_mat->at<uchar>(y, x);

            //this should be related to theta_d and variance theta_d * pixel_var.val[0]
            if (pow(app_diff.val[0], 2) <= 60) {
                //background
                bin_mat->at<uchar>(y, x) = 0;
            } else {
                //foreground
                bin_mat->at<uchar>(y, x) = 255;
            }
            
        }
    }
    
    //apparentBackgroundModel->updateBinary(next_frame);
    
    /*
    for (int y = 0; y < row_lim; ++y) {
        for (int x = 0; x < col_lim; ++x) {
            cv::Scalar pixel_diff = frame_u_mat->at<uchar>(y, x) - next_frame->at<uchar>(y, x);
            cv::Scalar pixel_var = frame_var_mat->at<uchar>(y, x);

            //this should be related to theta_d and variance theta_d * pixel_var.val[0]
            if (pow(pixel_diff.val[0], 2) <= 60){
                //background
                frame_bin_mat->at<uchar>(y, x) = 0;
            } else {
                //foreground
                frame_bin_mat->at<uchar>(y, x) = 255;
            }
        }
    }*/

    imshow("origin", *next_frame);
    cvWaitKey(1);
    imshow("processing", *apparentBackgroundModel->frame_u_mat);
    cvWaitKey(1);
    imshow("result", *apparentBackgroundModel->frame_bin_mat);
    cvWaitKey(1);
}

DualGaussianModel::~DualGaussianModel() {
    delete apparentBackgroundModel;
    delete candidateBackgroundModel;
    
    cvDestroyWindow("origin");
    cvDestroyWindow("processing");
    cvDestroyWindow("result");
}

