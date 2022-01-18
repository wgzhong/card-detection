#include <iostream>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/logging.h>

using namespace std;
using namespace cv;

class CaperaCapture{
public:
	CaperaCapture(int video_id = 0, int defatlt_w = 480, int defatlt_h = 480){
		cap.open(video_id);
		cap.set(CAP_PROP_FRAME_HEIGHT, defatlt_h);
		cap.set(CAP_PROP_FRAME_WIDTH, defatlt_w);
		if (!cap.isOpened()){
			LOG(INFO)<<"camera open error.........";
		}
	}
	int getimage(Mat &read_image){
		if (!cap.read(read_image)) {
			return -2;
		}
		return 0;
	}
private:
	VideoCapture cap;
};
//g++ test_opencv.cpp -o imshow -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs