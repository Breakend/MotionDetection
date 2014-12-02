############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# collect source files
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -stdlib=libstdc++" )
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -ccbin /usr/bin/clang )

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(HW1 ${OpenCV_LIBS})

