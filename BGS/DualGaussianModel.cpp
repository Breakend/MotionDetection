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
    prevFrame = first_image->clone();
    //set up windows
    namedWindow("origin", CV_WINDOW_AUTOSIZE);
    namedWindow("processing", CV_WINDOW_AUTOSIZE);
    namedWindow("result", CV_WINDOW_AUTOSIZE);
    namedWindow("homography", CV_WINDOW_AUTOSIZE);
    
    //last_frame = cvCloneMat(first_image);
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

    // 2.3. Motion Compensation by Mixing Models    
    // 1. Divide into grids 32 × 24, KLT on every corner of the grid
    // 2. RANSAC to obtain a homography matrix Ht:t−1 
    // 3. 

    
    cv::Mat m_prevImg = prevFrame.clone();
    cv::Mat m_nextImg= next_frame->clone();

    cv::Mat m_outImg;
    std::vector<cv::Point2f>   m_prevPts;
    std::vector<cv::Point2f>   m_nextPts;
    std::vector<unsigned char> m_status;
    std::vector<float>         m_error;
    
    // maxCorners – The maximum number of corners to return. If there are more corners
    // than that will be found, the strongest of them will be returned
    int maxCorners = 10000;
    // qualityLevel – Characterizes the minimal accepted quality of image corners;
    // the value of the parameter is multiplied by the by the best corner quality
    // measure (which is the min eigenvalue, see cornerMinEigenVal() ,
    // or the Harris function response, see cornerHarris() ).
    // The corners, which quality measure is less than the product, will be rejected.
    // For example, if the best corner has the quality measure = 1500,
    // and the qualityLevel=0.01 , then all the corners which quality measure is
    // less than 15 will be rejected.
    double qualityLevel = 0.01;
    // minDistance – The minimum possible Euclidean distance between the returned corners
    double minDistance = 5;
    // mask – The optional region of interest. If the image is not empty (then it
    // needs to have the type CV_8UC1 and the same size as image ), it will specify
    // the region in which the corners are detected
    cv::Mat mask;
    // blockSize – Size of the averaging block for computing derivative covariation
    // matrix over each pixel neighborhood, see cornerEigenValsAndVecs()
    int blockSize = 7;
    // useHarrisDetector – Indicates, whether to use operator or cornerMinEigenVal()
    bool useHarrisDetector = false;
    // k – Free parameter of Harris detector
    double k = 0.04;
    cv::Mat n;
    cv::goodFeaturesToTrack( m_prevImg, m_prevPts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );

    if(m_prevPts.size() >= 1) {
        
        cv::calcOpticalFlowPyrLK(m_prevImg, m_nextImg, m_prevPts, m_nextPts, m_status, m_error, Size(20,20), 5);
     
        //last_frame = cvCloneMat(next_frame);

        double ransacThreshold = 3;
        // compute homography using RANSAC
        cv::Mat mask;
        vector <Point2f> prev_corner2, cur_corner2;
        n = next_frame->clone();

        // weed out bad matches
        for(size_t i=0; i < m_status.size(); i++) {
            if(m_status[i]) {
                prev_corner2.push_back(m_prevPts[i]);
                cur_corner2.push_back(m_nextPts[i]);
//                cv::circle(n, m_prevPts[i], 100, cv::Scalar(255, 255, 255), -1);
//                cv::line(n, m_prevPts[i], m_nextPts[i], cv::Scalar(255,250,255));
//                cv::circle(n, m_nextPts[i], 100, cv::Scalar(255,255,255), -1);
            }
        }
        
        
        cv::Mat H = cv::findHomography(prev_corner2,cur_corner2, CV_RANSAC);
//        Mat M = estimateRigidTransform(prev_corner2,cur_corner2,0);
//        warpAffine(m_nextImg,m_outImg,M,m_nextImg.size(),INTER_NEAREST|WARP_INVERSE_MAP);
        warpPerspective(m_nextImg, n, H, m_prevImg.size(), INTER_LINEAR | WARP_INVERSE_MAP);
//        perspectiveTransform( m_prevImg, m_outImg, H);
        Mat meanCopy = apparentBackgroundModel->frame_u_mat->clone();
        Mat varCopy = apparentBackgroundModel->frame_var_mat->clone();
        Mat cmeanCopy = candidateBackgroundModel->frame_u_mat->clone();
        Mat cvarCopy = candidateBackgroundModel->frame_var_mat->clone();
        
        //warpPerspective(meanCopy, *apparentBackgroundModel->frame_u_mat,H,m_nextImg.size(), WARP_INVERSE_MAP, BORDER_CONSTANT);
        //warpPerspective(varCopy, *apparentBackgroundModel->frame_var_mat,H,m_nextImg.size(), WARP_INVERSE_MAP, BORDER_CONSTANT);
        //warpPerspective(meanCopy, *candidateBackgroundModel->frame_u_mat,H,m_nextImg.size(), WARP_INVERSE_MAP, BORDER_CONSTANT);
        //warpPerspective(varCopy, *candidateBackgroundModel->frame_var_mat,H,m_nextImg.size(), WARP_INVERSE_MAP, BORDER_CONSTANT);
//
//        for(int j=0; j<m_status.size(); j++){
//            if(m_status[j]){
//                
//                line(m_outImg,m_prevPts[j],m_nextPts[j],CV_RGB(64, 64, 255));
//            }
//        }
        // Update each mean of the apparent background model and candidate background model?
        
        // upd_mean.chatAt(y, x) = previous mapped mu
        
//         for(size_t i=0; i < m_status.size(); i++) {
//             if(m_status[i]) {
//                 cv::circle(n, m_prevPts[i], 1, cv::Scalar(255, 255, 255), -1);
//                 cv::line(n, m_prevPts[i], m_nextPts[i], cv::Scalar(40,250,255));
// //                cv::circle(n, m_nextPts[i], 1, cv::Scalar(0,255,255), -1);
//             }
//         }
        
        // variance.charAt(y,x) = previous mapped var
        imshow("homography", n);
            cv::Mat next_temp = next_frame->clone();
    prevFrame = next_temp;
        next_frame = &n;
        cvWaitKey(0);
    }
    else{
         cv::Mat next_temp = next_frame->clone();
    prevFrame = next_temp;
        printf("NO matching points");
    }

    // set the next frame...
    // next_frame



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
                candidateBackgroundModel->ages[x][y] = 1;
                candidateBackgroundModel->setPixel(next_frame, y, x);
            }
        
            if (candidateBackgroundModel->ages[x][y] > apparentBackgroundModel->ages[x][y] ){
                //Swap the models
                swapPixelsMat(y,x);
                float temp =apparentBackgroundModel->ages[x][y];
                apparentBackgroundModel->ages[x][y] = candidateBackgroundModel->ages[x][y];
                candidateBackgroundModel->ages[x][y] = temp;

                candidateBackgroundModel->setPixel(next_frame, y, x);
                candidateBackgroundModel->ages[x][y] = 1;
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
    cvDestroyWindow("homography");
}