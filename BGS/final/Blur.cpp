
#include "Blur.hpp"

#define SHOW_BLUR_IMAGES 0

void serialMedianBlur(cv::Mat frame, cv::Mat destination, int size)
{
  if (SHOW_BLUR_IMAGES) {
    namedWindow("original", CV_WINDOW_AUTOSIZE);
    namedWindow("MedianBlur", CV_WINDOW_AUTOSIZE);
  }

  destination = frame.clone();

  // Filter width should be odd as we are calculating average blur 
  // for a pixel plus some offset in all directions
  assert(size % 2 == 1);

  int window_len = size * size;
  int window[window_len];
  
  int rows = frame.rows;
  int cols = frame.cols;

  const int half   = size / 2;
  const int width  = cols - 1;
  const int height = rows - 1;

  // Compute blur
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
    	int idx = 0;
      //Average pixel color summing up adjacent pixels.
      for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
          // Clamp filter to the image border
       		int h = min(max(r + i, 0), height); 
       		int w = min(max(c + j, 0), width);
       		window[idx] = frame.at<uchar>(h,w); // (row,col)
       		idx++;
       	}
    	}

    	// sort the window to find median
      insertionSort(window, window_len);

      // assign the median to centered element of the matrix
      destination.at<uchar>(r,c) = window[idx / 2];
    }
  }
  
  if (SHOW_BLUR_IMAGES) {
    imshow("original", frame);
    cvWaitKey(1);
    imshow("MedianBlur", destination);
    cvWaitKey(0);
  }
} 

//sort the window using insertion sort
//insertion sort is best for this sorting
void insertionSort(int *window, int length)
{
  int temp, i , j;
  for (i = 0; i < length; i++) {
      temp = window[i];
      for (j = i-1; j >= 0 && temp < window[j]; j--) {
          window[j+1] = window[j];
      }
      window[j+1] = temp;
  }
}

void serialGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size)
{
  if (SHOW_BLUR_IMAGES) {
    namedWindow("original", CV_WINDOW_AUTOSIZE);
    namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);
  }

  destination = frame.clone();

  // Filter width should be odd as we are calculating average blur for a pixel plus some offset in all directions
  assert(size.width % 2 == 1);

  float gaussian_filter[size.width * size.width];
  createGaussianFilter(gaussian_filter, size.width);

  
  int rows = frame.rows;
  int cols = frame.cols;

  const int half   = size.width / 2;
  const int width  = cols - 1;
  const int height = rows - 1;

  // Compute blur
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float blur = 0.f;

      //Average pixel color summing up adjacent pixels.
      for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
          // Clamp filter to the image border
          int h = min(max(r + i, 0), height); 
          int w = min(max(c + j, 0), width);

          // Blur is a product of current pixel value and weight of that pixel.
          // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
          float pixel = frame.at<uchar>(h,w); // (row,col)
          int idx = (i + half) * size.width + (j + half); // width
          float weight = gaussian_filter[idx];
          blur += pixel * weight;
        }
      }
      destination.at<uchar>(r,c) = static_cast<unsigned char>(blur);
    }
  } 
  
  if (SHOW_BLUR_IMAGES) {
    imshow("original", frame);
    cvWaitKey(1);
    imshow("GaussianBlur", destination);
    cvWaitKey(0);
  }
} 

// Creates gaussian filter based on G(x,y) formula: http://en.wikipedia.org/wiki/Gaussian_blur.
void createGaussianFilter(float *gaussian_filter, int width)
{
    const float sigma   = 2.f;              // Standard deviation of the Gaussian distribution.
    const int   half    = width / 2;
    float       sum     = 0.f;
  
    // Create convolution matrix
    //m_filter.weight.resize(width * width);
 
    // Calculate filter sum first
    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            // e (natural logarithm base) to the power x, where x is what's in the brackets
            float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * sigma * sigma));
            int idx = (r + half) * width + (c + half);
 
            gaussian_filter[idx] = weight;
            sum += weight;
        }
    }
 
    // Normalize weight: sum of weights must equal 1
    float normal = 1.f / sum;
 
    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            int idx = (r + half) * width + c + half;
            gaussian_filter[idx] *= normal;
        }
    }
}

class Median_blur_process : public cv::ParallelLoopBody 
{
    private:
        cv::Mat *frame;
        cv::Mat *destination;
        int blur_size;
        int num_threads;

    public:
        Median_blur_process(cv::Mat *frm, cv::Mat *dest, int bsz, int nt) :
            frame(frm), destination(dest), blur_size(bsz), num_threads(nt) {}

        virtual void operator()(const cv::Range& range) const
        {
          for (int rank = range.start; rank < range.end; rank++) {
            int size = num_threads;
            int blocking_factor = frame->rows / size; 
            int offset = blocking_factor * rank;
            int row_limit = offset + blocking_factor; 
            if (rank == size - 1) row_limit = frame->rows;

            int window_len = blur_size * blur_size;
            int window[window_len];
          
            const int half   = blur_size / 2;
            const int width  = frame->cols - 1;
            const int height = frame->rows - 1;

            // Compute blur
            for (int r = offset; r < row_limit; ++r) {
              for (int c = 0; c < frame->cols; ++c) {
                int idx = 0;
                //Average pixel color summing up adjacent pixels.
                for (int i = -half; i <= half; ++i) {
                  for (int j = -half; j <= half; ++j) {
                    // Clamp filter to the image border
                    int h = min(max(r + i, 0), height); 
                    int w = min(max(c + j, 0), width);
                    window[idx] = frame->at<uchar>(h,w); // (row,col)
                    idx++;
                  }
                }

                // sort the window to find median
                insertionSort(window, window_len);

                // assign the median to centered element of the matrix
                destination->at<uchar>(r,c) = window[idx / 2];
              }
            }
          }
        }
};

void tbbMedianBlur(cv::Mat frame, cv::Mat destination, int size, int num_threads)
{
  if (SHOW_BLUR_IMAGES) {
    namedWindow("original", CV_WINDOW_AUTOSIZE);
    namedWindow("MedianBlur", CV_WINDOW_AUTOSIZE);
  }

  destination = frame.clone();

  // Filter width should be odd as we are calculating average blur 
  // for a pixel plus some offset in all directions
  assert(size % 2 == 1);

  cv::parallel_for_(cv::Range(0,num_threads), Median_blur_process(&frame, &destination, size, num_threads));

  if (SHOW_BLUR_IMAGES) {
    imshow("original", frame);
    cvWaitKey(1);
    imshow("MedianBlur", destination);
    cvWaitKey(0);
  }
}

class Gaussian_blur_process : public cv::ParallelLoopBody 
{
    private:
        cv::Mat *frame;
        cv::Mat *destination;
        cv::Size blur_size;
        int num_threads;

    public:
        Gaussian_blur_process(cv::Mat *frm, cv::Mat *dest, cv::Size bsz, int nt) :
            frame(frm), destination(dest), blur_size(bsz), num_threads(nt) {}

        virtual void operator()(const cv::Range& range) const
        {
          for (int rank = range.start; rank < range.end; rank++) {
            int size = num_threads;
            int blocking_factor = frame->rows / size; 
            int offset = blocking_factor * rank;
            int row_limit = offset + blocking_factor; 
            if (rank == size - 1) row_limit = frame->rows;
       
            float gaussian_filter[blur_size.width * blur_size.width];
            createGaussianFilter(gaussian_filter, blur_size.width);

            const int half   = blur_size.width / 2;
            const int width  = frame->cols - 1;
            const int height = frame->rows - 1;

            // Compute blur
            for (int r = offset; r < row_limit; ++r) {
              for (int c = 0; c < frame->cols; ++c) {
                float blur = 0.f;

                //Average pixel color summing up adjacent pixels.
                for (int i = -half; i <= half; ++i) {
                  for (int j = -half; j <= half; ++j) {
                    // Clamp filter to the image border
                    int h = min(max(r + i, 0), height); 
                    int w = min(max(c + j, 0), width);

                    // Blur is a product of current pixel value and weight of that pixel.
                    // Remember that sum of all weights equals to 1, so we are averaging sum of all pixels by their weight.
                    float pixel = frame->at<uchar>(h,w); // (row,col)
                    int idx = (i + half) * blur_size.width + (j + half); // width
                    float weight = gaussian_filter[idx];
                    blur += pixel * weight;
                  }
                }
                destination->at<uchar>(r,c) = static_cast<unsigned char>(blur);
              }
            } 
          }
        }

};

void tbbGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size, int num_threads)
{
  if (SHOW_BLUR_IMAGES) {
    namedWindow("original", CV_WINDOW_AUTOSIZE);
    namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);
  }

  destination = frame.clone();

  // Filter width should be odd as we are calculating average blur for a pixel plus some offset in all directions
  assert(size.width % 2 == 1);

  cv::parallel_for_(cv::Range(0,num_threads), Gaussian_blur_process(&frame, &destination, size, num_threads));

  if (SHOW_BLUR_IMAGES) {
    imshow("original", frame);
    cvWaitKey(1);
    imshow("GaussianBlur", destination);
    cvWaitKey(0);
  }
}

