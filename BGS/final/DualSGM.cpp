
#include "DualSGM.hpp"

#define SHOW_IMAGES 0
#define MOTION_COMP 1
#define OPENCV_BLUR 1
#define GAUSSIAN_SIZE 7 // Must be odd
#define MEDIAN_SIZE 3 // Must be odd

#define AGE_THRESH 10
#define MEAN_THRESH 9.0 //9.0
#define THETA_D 5
#define VAR_INIT 255 // max unsigned char

DualSGM::DualSGM(Mat* first_image, int N) {
    if (SHOW_IMAGES) {
        // Set up windows
        namedWindow("origin", CV_WINDOW_AUTOSIZE);
        namedWindow("blurred", CV_WINDOW_AUTOSIZE);
        namedWindow("processing", CV_WINDOW_AUTOSIZE);
        namedWindow("result", CV_WINDOW_AUTOSIZE);
        if (MOTION_COMP) {
            namedWindow("homography", CV_WINDOW_AUTOSIZE);
        }
    }
    
    prev_frame = first_image->clone();
    
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
            // app_var_mat->at<uchar>(i,j) = var_init; INIT VAR
        }
    }

    //DualSGM::num_rows = first_image->rows;
    //DualSGM::num_cols = first_image->cols;
}

void DualSGM::serialUpdateModel(Mat *next_frame) {
    Mat origin;
    if (SHOW_IMAGES) {
        origin = next_frame->clone(); 
    }
    /* Pre processing */
    Mat destination;
    Size gb_size = Size(GAUSSIAN_SIZE,GAUSSIAN_SIZE);
    if (OPENCV_BLUR) {
        GaussianBlur(*next_frame, destination, gb_size, 0, 0); 
        medianBlur(destination, *next_frame, MEDIAN_SIZE);
        cvWaitKey(1);
    } else {
        serialGaussianBlur(*next_frame, destination, gb_size);
        serialMedianBlur(destination, *next_frame, MEDIAN_SIZE);
    }

    /* Motion Compensation */
    if (MOTION_COMP) {
        motionCompensation(next_frame);
    }

    /* Duel Gaussian Model */
    core_dsgm_update(next_frame, bin_mat, 
        app_u_mat, app_var_mat, 
        can_u_mat, can_var_mat, 
        app_ages, can_ages, 
        0, next_frame->rows);
    
    if (SHOW_IMAGES) {
        // Show update
        imshow("origin", origin);
        cvWaitKey(1);
        imshow("blurred", *next_frame);
        cvWaitKey(1);
        imshow("processing", *app_u_mat);
        cvWaitKey(1);
        imshow("result", *bin_mat);
        cvWaitKey(1);
    }

}

void DualSGM::motionCompensation(Mat* next_frame)
{
    cv::Mat n;

    cv::Mat m_prevImg = prev_frame.clone();
    cv::Mat m_nextImg = next_frame->clone();

    cv::Mat m_outImg;
    std::vector<cv::Point2f>   m_prevPts;
    std::vector<cv::Point2f>   m_nextPts;
    std::vector<unsigned char> m_status;
    std::vector<float>         m_error;
    
    // maxCorners – The maximum number of corners to return. If there are more corners
    // than that will be found, the strongest of them will be returned
    int maxCorners = 100;
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

    cv::goodFeaturesToTrack(m_prevImg, m_prevPts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

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
            }
        }
        
        cv::Mat H = cv::findHomography(prev_corner2,cur_corner2, CV_RANSAC);
        warpPerspective(m_nextImg, n, H, m_prevImg.size(), INTER_LINEAR | WARP_INVERSE_MAP);
        
        // variance.charAt(y,x) = previous mapped var
        if (SHOW_IMAGES) {
            cvWaitKey(1);
            imshow("homography", n);
        }

        cv::Mat next_temp = next_frame->clone();
        prev_frame = next_temp;
        next_frame = &n;
    } else {
        cv::Mat next_temp = next_frame->clone();
        prev_frame = next_temp;
        printf("NO matching points");
    }
}

class Dsgm_process : public cv::ParallelLoopBody 
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
        int num_threads;

    public:
        Dsgm_process(cv::Mat *frame, cv::Mat *bmat, cv::Mat *aumat, cv::Mat *avmat, 
            cv::Mat *cumat, cv::Mat *cvmat, int **aamat, int **camat, int nt) :
            next_frame(frame), bin_mat(bmat), app_u_mat(aumat), 
            app_var_mat(avmat), can_u_mat(cumat), can_var_mat(cvmat),
            app_ages(aamat), can_ages(camat), num_threads(nt) {}

        virtual void operator()(const cv::Range& range) const
        {
            for (int rank = range.start; rank < range.end; rank++) {
                int size = num_threads;
                int blocking_factor = next_frame->rows / size; 
                int offset = blocking_factor * rank;
                int row_limit = offset + blocking_factor; 
                if (rank == size - 1) row_limit = next_frame->rows;

                core_dsgm_update(next_frame, bin_mat, 
                    app_u_mat, app_var_mat, 
                    can_u_mat, can_var_mat, 
                    app_ages, can_ages, 
                    offset, row_limit);

            }
        }

};

double DualSGM::tbbUpdateModel(Mat *next_frame, int num_threads) {
    Mat origin;
    if (SHOW_IMAGES) {
        origin = next_frame->clone(); 
    }

    /* Pre processing */
    Mat destination;
    Size gb_size = Size(GAUSSIAN_SIZE,GAUSSIAN_SIZE);
    if (OPENCV_BLUR) {
        GaussianBlur(*next_frame, destination, gb_size, 0, 0); 
        medianBlur(destination, *next_frame, MEDIAN_SIZE);
        cvWaitKey(1);
    } else {
        tbbGaussianBlur(*next_frame, destination, gb_size, num_threads);
        tbbMedianBlur(destination, *next_frame, MEDIAN_SIZE, num_threads);
    }

    /* Motion Compensation */
    if (MOTION_COMP) {
        motionCompensation(next_frame);
    }

    /* Duel Gaussian Model */
    double start_time = timer();
    cv::parallel_for_(cv::Range(0,num_threads), Dsgm_process(next_frame, bin_mat, app_u_mat, 
        app_var_mat, can_u_mat, can_var_mat, app_ages, can_ages, num_threads));
    double end_time = timer();

    if (SHOW_IMAGES) {
        // Show update
        imshow("origin", origin);
        cvWaitKey(1);
        imshow("blurred", *next_frame);
        cvWaitKey(1);
        imshow("processing", *app_u_mat);
        cvWaitKey(1);
        imshow("result", *bin_mat);
        cvWaitKey(1);
    }

    return end_time - start_time;
}

double DualSGM::timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}

DualSGM::~DualSGM() {
    //delete prev_frame;
    delete bin_mat;
    delete app_u_mat;
    delete app_var_mat;
    delete can_u_mat;
    delete can_var_mat;
    free(app_ages);
    free(can_ages);

    if (SHOW_IMAGES) {
        cvDestroyWindow("origin");
        cvDestroyWindow("blurred");
        cvDestroyWindow("processing");
        cvDestroyWindow("result");
    }

}

inline void core_dsgm_update(cv::Mat *next_frame, cv::Mat *bin_mat, 
    cv::Mat *app_u_mat, cv::Mat *app_var_mat, 
    cv::Mat *can_u_mat, cv::Mat *can_var_mat, 
    int **app_ages, int **can_ages, 
    int offset, int row_lim) 
{
    int i;
    float alpha;
    float V;

    int col_lim = next_frame->cols;

    /* Core update loop */
    for (int y = offset; y < row_lim; ++y) {
        for (int x = 0; x < col_lim; ++x) {
            
            cv::Scalar i_sclr       = next_frame->at<uchar>(y,x);
            cv::Scalar app_u_sclr   = app_u_mat->at<uchar>(y,x);
            cv::Scalar app_var_sclr = app_var_mat->at<uchar>(y,x);
            cv::Scalar can_u_sclr   = can_u_mat->at<uchar>(y,x);
            cv::Scalar can_var_sclr = can_var_mat->at<uchar>(y,x);

            // Get the differences for the candidate and apparent background models
            float adiff = i_sclr.val[0] - app_u_sclr.val[0];
            float cdiff = i_sclr.val[0] - can_u_sclr.val[0];
            
            if (pow(adiff, 2) < MEAN_THRESH * max(app_var_sclr.val[0], .1)){
                //apparentBackgroundModel->updatePixel(next_frame, y, x);
                i = 0;
                alpha = 1.0 / (double)app_ages[x][y];
                app_u_sclr.val[i] = (1.0-alpha) * app_u_sclr.val[i] + (alpha) * i_sclr.val[i];
                V =  pow((app_u_sclr.val[i] - i_sclr.val[i]),2);
                app_var_sclr.val[i] = (1.0-alpha) * app_var_sclr.val[i] + alpha * V;
                
                //write into matrix
                app_u_mat->at<uchar>(y,x) = app_u_sclr.val[i];
                app_var_mat->at<uchar>(y,x) = app_var_sclr.val[i];
                
                if (app_ages[x][y] < AGE_THRESH) {
                    app_ages[x][y]++;
                }

            } else if (pow(cdiff, 2) < MEAN_THRESH * max(can_var_sclr.val[0], .1)) {
                //candidateBackgroundModel->updatePixel(next_frame, y, x);
                i = 0;
                alpha = 1.0 / (double)can_ages[x][y];
                can_u_sclr.val[i] = (1.0-alpha) * can_u_sclr.val[i] + (alpha) * i_sclr.val[i];
                V = pow((can_u_sclr.val[i] - i_sclr.val[i]),2);
                can_var_sclr.val[i] = (1.0-alpha) * can_var_sclr.val[i] + alpha * V;
                
                //write into matrix
                can_u_mat->at<uchar>(y,x) = can_u_sclr.val[i];
                can_var_mat->at<uchar>(y,x) = can_var_sclr.val[i];
                
                // Cap ages
                if (can_ages[x][y] < AGE_THRESH) {
                    can_ages[x][y]++;
                }

            } else {
                //can_ages[x][y] = 1;
                //candidateBackgroundModel->setPixel(next_frame, y, x);
                
                can_u_mat->at<uchar>(y, x) = i_sclr.val[0];
                can_var_mat->at<uchar>(y, x) = VAR_INIT;
                can_ages[x][y] = 1;
            }
        
            if (can_ages[x][y] > app_ages[x][y]) {
                // Swap the models
                app_u_mat->at<uchar>(y,x) = can_u_sclr.val[0];
                app_var_mat->at<uchar>(y,x) = can_var_sclr.val[0];
                app_ages[x][y] = can_ages[x][y];

                //candidateBackgroundModel->setPixel(next_frame, y, x);
                can_u_mat->at<uchar>(y, x) = i_sclr.val[0];
                can_var_mat->at<uchar>(y, x) = VAR_INIT;
                can_ages[x][y] = 1;
            }
            

            cv::Scalar app_diff = app_u_mat->at<uchar>(y,x) - next_frame->at<uchar>(y,x);
            //cv::Scalar pixel_var = frame_var_mat->at<uchar>(y, x);

            //this should be related to theta_d and variance theta_d * i_sclr.val[0] //pixel_var.val[0]
            if (pow(app_diff.val[0], 2) <= THETA_D*max(0.25, i_sclr.val[0])) { // 60
                //background
                bin_mat->at<uchar>(y, x) = 0;
            } else {
                //foreground
                bin_mat->at<uchar>(y, x) = 255;
            }
            
        }
    }
}

