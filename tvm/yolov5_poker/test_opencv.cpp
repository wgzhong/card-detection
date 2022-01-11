#include<iostream>
#include<opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
	Mat src=imread("./cam_image38.jpg");
	
	return 0;
}
//g++ test_opencv.cpp -o imshow -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs