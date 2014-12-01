
#include "DualSGM.hpp"

DualSGM::DualSGM(Mat* first_image, int N) {

    // Set up windows
    namedWindow("origin", CV_WINDOW_AUTOSIZE);
    namedWindow("processing", CV_WINDOW_AUTOSIZE);
    namedWindow("result", CV_WINDOW_AUTOSIZE);

    bin_mat     = new cv::Mat(first_image->size(), CV_8U);
    app_u_mat   = new cv::Mat(first_image->size(), CV_8U);
    app_var_mat = new cv::Mat(first_image->size(), CV_8U);
    can_u_mat   = new cv::Mat(first_image->size(), CV_8U);
    can_var_mat = new cv::Mat(first_image->size(), CV_8U);

    app_ages = (int **) std::calloc(first_image->cols, sizeof(int *));
    can_ages = (int **) std::calloc(first_image->cols, sizeof(int *));

    for (int i = 0; i < first_image->cols; ++i) {
        app_ages[i] = (int *) std::calloc(first_image->rows, sizeof(int));
        can_ages[i] = (int *) std::calloc(first_image->rows, sizeof(int));
    }
    
    for (int i = 0; i < first_image->cols; i++) {
        for (int j = 0; j< first_image->rows; j++) {
            app_ages[i][j] = 1;
            can_ages[i][j] = 1;
        }
    }

}

void DualSGM::serialUpdateModel(Mat *next_frame) {

    int i;
    float alpha;
    float V;

    int row_lim = next_frame->rows;
    int col_lim = next_frame->cols;

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

            } else if (pow(cdiff, 2) < meanThreshold * can_var_sclr.val[0]) {
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
    
    // Show update
    imshow("origin", *next_frame);
    cvWaitKey(1);
    imshow("processing", *app_u_mat);
    cvWaitKey(1);
    imshow("result", *bin_mat);
    cvWaitKey(1);
}

DualSGM::~DualSGM() {
    delete bin_mat;
    delete app_u_mat;
    delete app_var_mat;
    delete can_u_mat;
    delete can_var_mat;
    free(app_ages);
    free(can_ages);

    cvDestroyWindow("origin");
    cvDestroyWindow("processing");
    cvDestroyWindow("result");
}

