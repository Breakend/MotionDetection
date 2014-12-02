
#include "DualSGM.hpp"

DualSGM::DualSGM(Mat* first_image, int N) {

    // Set up windows
    namedWindow("origin", CV_WINDOW_AUTOSIZE);
    namedWindow("processing", CV_WINDOW_AUTOSIZE);
    namedWindow("result", CV_WINDOW_AUTOSIZE);

    /* Init mem, canidate, apparent */
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

    //DualSGM::num_rows = first_image->rows;
    //DualSGM::num_cols = first_image->cols;

}

void DualSGM::serialUpdateModel(Mat *next_frame) {

    int i;
    float alpha;
    float V;
    float meanThreshold = MEAN_THRESH;

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

class Parallel_process : public cv::ParallelLoopBody 
{
    private:
        cv::Mat *next_frame;
        cv::Mat *bin_mat;
        cv::Mat *app_u_mat;
        cv::Mat *app_var_mat;
        cv::Mat *can_u_mat;
        cv::Mat *can_var_mat;
        int **app_ages;
        int **can_ages;

    public:
        Parallel_process(cv::Mat *frame, cv::Mat *bmat, cv::Mat *aumat, cv::Mat *avmat, 
            cv::Mat *cumat, cv::Mat *cvmat, int **aamat, int **camat) :
            next_frame(frame), bin_mat(bmat), app_u_mat(aumat), 
            app_var_mat(avmat), can_u_mat(cumat), can_var_mat(cvmat),
            app_ages(aamat), can_ages(camat) {}

        virtual void operator()(const cv::Range& range) const
        {
            for (int rank = range.start; rank < range.end; rank++) {
                int size = DualSGM::NUM_THREADS;
                int blocking_factor = next_frame->rows / size; 
                int offset = blocking_factor * rank;
                int row_limit = offset + blocking_factor; 
                if (rank == size - 1) row_limit = next_frame->rows;

                int i;
                float alpha;
                float V;
                float meanThreshold = DualSGM::MEAN_THRESH;
                float var_init = DualSGM::VAR_INIT;

                //printf("[%i] with size %i, owns rows %i to %i of %i total \n", rank, size, offset, row_limit, next_frame->rows);

                //cv::in(next_frame, cv::Rect(0, offset, next_frame->cols, blocking_factor));

                //cv::Mat part_frame = (*next_frame)(cv::Rect(0, offset, next_frame->cols, blocking_factor));

                for (int y = offset; y < row_limit; y++) { // ++y
                    for(int x = 0; x < next_frame->cols; x++) {

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
                        

                        cv::Scalar app_diff = app_u_mat->at<uchar>(y,x) - i_sclr.val[0];
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

                // Your code here
                //printf("[%i] with size %i \n", rank, DualSGM::NUM_THREADS);
                
                //cv::Mat B = cv::Mat(2,2,CV_8U);
                //cv::Mat A = cv::Mat(3,3,CV_8U);
                //cv::Mat tmp = A(cv::Rect(0,0,2,2));
                //B.copyTo(tmp);
                //cv::in(img, cv::Rect(0, (img.rows/diff)*i, img.cols, img.rows/diff));

                //int blocking_factor = next_frame->rows / size; 
                //int offset = blocking_factor * i;
                //int row_limit = offset + blocking_factor; 
                //if (i == size - 1) row_limit = next_frame->rows;

                //cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i, img.cols, img.rows/diff));
                //cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i, retVal.cols, retVal.rows/diff));
                //cv::GaussianBlur(in, out, cv::Size(size, size), 0);

            }
        }

};

void DualSGM::tbbUpdateModel(Mat *next_frame) {

    cv::parallel_for_(cv::Range(0,NUM_THREADS), Parallel_process(next_frame, bin_mat, app_u_mat, 
        app_var_mat, can_u_mat, can_var_mat, app_ages, can_ages));

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

