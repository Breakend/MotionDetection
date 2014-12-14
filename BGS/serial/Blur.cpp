
#include "Blur.hpp"

void serialMedianBlur(cv::Mat frame, cv::Mat destination, int size)
{
  namedWindow("original", CV_WINDOW_AUTOSIZE);
  namedWindow("MedianBlur", CV_WINDOW_AUTOSIZE);

  destination = frame.clone();

  // Filter width should be odd as we are calculating average blur 
  // for a pixel plus some offset in all directions
  assert(size % 2 == 1);

  int window[size * size];
  
  int rows = frame.rows;
  int cols = frame.cols;

  const int half   = size / 2;
  const int width  = cols - 1;
  const int height = rows - 1;

  // Compute blur
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      //float blur = 0.f;

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
      insertionSort(window);

      // assign the median to centered element of the matrix
      destination.at<uchar>(r,c) = window[idx / 2];
    }
  } 

  imshow("original", frame);
  cvWaitKey(1);
  imshow("MedianBlur", destination);
  cvWaitKey(0);
} 

//sort the window using insertion sort
//insertion sort is best for this sorting
void insertionSort(int *window)
{
  int temp, i , j;
  for (i = 0; i < 9; i++) {
      temp = window[i];
      for (j = i-1; j >= 0 && temp < window[j]; j--) {
          window[j+1] = window[j];
      }
      window[j+1] = temp;
  }
}

void serialGaussianBlur(cv::Mat frame, cv::Mat destination, cv::Size size)
{
  namedWindow("original", CV_WINDOW_AUTOSIZE);
  namedWindow("GaussianBlur", CV_WINDOW_AUTOSIZE);

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

  imshow("original", frame);
  cvWaitKey(1);
  imshow("GaussianBlur", destination);
  cvWaitKey(0);
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
 
    printf("Half way!\n");

    // Normalize weight: sum of weights must equal 1
    float normal = 1.f / sum;
 
    for (int r = -half; r <= half; ++r) {
        for (int c = -half; c <= half; ++c) {
            int idx = (r + half) * width + c + half;
            gaussian_filter[idx] *= normal;
        }
    }
}
