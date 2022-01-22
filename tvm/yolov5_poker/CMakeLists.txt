cmake_minimum_required(VERSION 3.7)
project(vastai VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(TVM_PATH /home/pi/tvm)
set(OPENCV_PATH /usr/local/include)
set(OPENCV_LIBS /usr/local/lib)

include_directories(include)
include_directories(${OPENCV_PATH})
include_directories(${TVM_PATH}/include/)
include_directories(${TVM_PATH}/include/tvm/runtime/)
include_directories(${TVM_PATH}/3rdparty/dlpack/include)
include_directories(${TVM_PATH}/3rdparty/dmlc-core/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

file(GLOB SOURCE_FILES "*.cc")
SET(CMAKE_BUILD_TYPE Release)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/exe)
add_executable(delopy ${SOURCE_FILES} ${SRC_LIST})

target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_core.so)
target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_highgui.so)
target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_imgproc.so)
target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_imgcodecs.so)
target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_videoio.so)
target_link_libraries(delopy ${OPENCV_LIBS}/libopencv_video.so)

target_link_libraries(delopy stdc++fs dl pthread wiringPi) #wiringPi

