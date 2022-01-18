#include <iostream>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <time.h>
#include <sys/time.h>    
#include <unistd.h> 
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui.hpp>

#include "tvm_runtime_pack.h"
// #include "test_uart.h"
#include "test_opencv.h"

#define IMAGE_WIDTH 640
#define IMAGE_HIGHT 640
#define IMAGE_DEPTH 3

#define CONF_THRES 0.025
#define IOU_THRES 0.6
#define MAX_NUM 3000
 struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    bbox(){
        x1=0.0;
        x2=0.0;
        y1=0.0;
        y2=0.0;
    }
};
struct ground_truth{
    bbox box;
    float class_prob;
    int label_idx;
    ground_truth(){
        class_prob=0.0;
        label_idx=0;
    }
};

bbox xywh2xyxy(float x, float y, float w, float h, int shift){
    bbox output_box;
    output_box.x1 = x - (w / 2.0) + shift;
    output_box.x2 = x + (w / 2.0) + shift;
    output_box.y1 = y - (h / 2.0) + shift;
    output_box.y2 = y + (h / 2.0) + shift;
    return output_box;
}

template <typename T>
void dump_data(T *data, std::string save_path, int size){
    std::ofstream outfile(save_path, std::ifstream::binary);
    outfile.write((char *)(data), size);
    outfile.close();
}

void mat2data(cv::Mat input_frame, char *data, int h, int w){
    assert(data && !input_frame.empty());
    unsigned int volChl = w * h;
    for (unsigned j = 0; j < volChl *3 ; ++j){
        data[j] = input_frame.data[j];
    }
}

void letterbox(cv::Mat image, cv::Mat &output, int new_w, int new_h){
    int w = image.cols;
    int h = image.rows;
    float ratio = std::min(float((new_w*1.0)/(w*1.0)), float((new_h*1.0)/(h*1.0)));
    int new_uppad_w = int(std::round(w*ratio));
    int new_uppad_h = int(std::round(h*ratio));
    int dw = (new_w - new_uppad_w) / 2;
    int dh = (new_h - new_uppad_h) / 2;
    cv::Mat resize_image;
    if((w != new_uppad_w) || (h != new_uppad_h)){
        cv::resize(image, resize_image, cv::Size(new_uppad_w, new_uppad_h));
    }
    if(resize_image.empty()){
        resize_image = image;
    }
    int top = std::round(dh - 0.1);
    int bottom = std::round(dh + 0.1);
    int left = std::round(dw - 0.1);
    int right = std::round(dw + 0.1);
    cv::copyMakeBorder(resize_image, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
}

void Mat_to_CHW(cv::Mat &input_frame, float *data, int w, int h)
{
    assert(data && !input_frame.empty());
    unsigned int volChl = w * h;
    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(float(input_frame.data[j * 3 + c]) / 255.0);
    }
}

void process_image(cv::Mat src, float *data){
    cv::Mat dst, input;
    letterbox(src, dst, IMAGE_WIDTH, IMAGE_HIGHT);
    cv::cvtColor(dst, input, cv::COLOR_BGR2RGB);
    Mat_to_CHW(input, data, IMAGE_WIDTH , IMAGE_HIGHT);
}

int load_input_hex(std::string path, float *data){
    std::ifstream is(path , std::ifstream::in | std::ios::binary);
	// 2. 计算图片长度
	is.seekg(0, is.end);
	int length = is.tellg();
	is.seekg(0, is.beg);
    LOG(INFO) << "length = "<<length;
    if(length < 1){
        LOG(INFO) <<"not data, please input image or text or bin";
        return -1;
    }
	char * buffer = new char[length];
	is.read(buffer, length);
	is.close();
    memcpy(data, buffer, length);
    delete [] buffer;
    return 0;
}

std::vector<std::string> read_label(std::string path){
    std::vector<std::string> label;
    std::ifstream is(path);
    std::string  line; 
    while(getline(is, line))
    {
        label.push_back(line); 
    }
    return label;
} 
float soft_iou(bbox b1, bbox b2){
    float maxx1 = std::max(b1.x1, b2.x1);
    float minx2 = std::min(b1.x2, b2.x2);
    float maxy1 = std::max(b1.y1, b2.y1);
    float miny2 = std::min(b1.y2, b2.y2);
    float h = miny2 - maxy1 + 1;
    float w = minx2 - maxx1 + 1;
    if(h < 0 || w < 0){
        return 0;
    }
    float inarea = w*h;
    float area1 = (b1.x2 - b1.x1 + 1)*(b1.y2 - b1.y1 + 1);
    float area2 = (b2.x2 - b2.x1 + 1)*(b2.y2 - b2.y1 + 1);
    float iou = inarea / (area1 + area1 - inarea);
    // LOG(INFO)<<" b1.x1= "<<b1.x1<<" b1.x2= "<<b1.x2<<" b1.y1= "<<b1.y1<<" b1.y2= "<<b1.y2;
    // LOG(INFO)<<" b2.x1= "<<b2.x1<<" b2.x2= "<<b2.x2<<" b2.y1= "<<b2.y1<<" b2.y2= "<<b2.y2;
    // LOG(INFO)<<"area1= "<<area1<<" area2= "<<area2<<" inarea= "<<inarea<<" w= "<<w<<" h= "<<h;
    return iou;
}

std::vector<int> argsort(const std::vector<ground_truth>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;
	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (array[pos1].class_prob > array[pos2].class_prob);});
	return array_index;
}

void non_max_suppression(float *data, std::vector<ground_truth> &output,  float conf_thres, float iou_thres, int max_nms, int class_num, int output_num){
    int stride = class_num+4+1;
    std::vector<ground_truth> gt_v;
    for(int n=0; n<output_num*stride; n=n+stride){
        float center_x = data[n];
        float center_y = data[n+1];
        float width = data[n+2];
        float hight = data[n+3];
        float conf = data[n+4];
        if(conf < conf_thres){
            continue;
        }
        float max_class_conf=0.0;
        int idx=0;
        for (int i = n+5; i < n+class_num+5; i++){
            if(data[i] > max_class_conf){
                max_class_conf = data[i];
                idx = (i-5) % 29;
            }
        }
        max_class_conf = max_class_conf*conf;
        if(max_class_conf < conf_thres){
            continue;
        }
        int shift = 0;
        bbox box = xywh2xyxy(center_x, center_y, width, hight, shift);
        ground_truth gt;
        gt.box = box;
        gt.class_prob = max_class_conf;
        gt.label_idx = idx;
        gt_v.push_back(gt);
    }
    std::vector<int> idx = argsort(gt_v);
    while(idx.size() > 0){
        int k = idx[0];
        if(gt_v[k].class_prob > conf_thres){
            output.push_back(gt_v[k]);
        }
        std::vector<int> idx_tmp;
        for(int i = 1; i < idx.size(); i++){
            float iou = soft_iou(gt_v[k].box, gt_v[idx[i]].box);
            if(iou <= iou_thres){
                idx_tmp.push_back(i-1);
            }
        }
        std::vector<int> i_t = idx;
        idx.clear();
        for(int i=0;i<idx_tmp.size();i++){
            idx.push_back(i_t[idx_tmp[i]+1]);
        }
    }
}

long getTimeUsec()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (long)((long)t.tv_sec * 1000 * 1000 + t.tv_usec);
}

int main(){
    DLDevice dev{kDLCPU, 0};
    float data[IMAGE_WIDTH * IMAGE_HIGHT * IMAGE_DEPTH];
    std::vector<ground_truth> output;
    int class_num=54;
    int dims = 4 + 1 + class_num;//x,y,h,w,confidence,class
    int output_shape=(20*20+40*40+80*80)*3;//三路输出的特征图大小，每个cell有3个检测框
    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_HIGHT}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, output_shape, dims}, DLDataType{kDLFloat, 32, 1}, dev);
    LOG(INFO) << "load model...";
    std::vector<std::string> label = read_label("../../data/classes.txt");
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("../python/relay_yolov5s.so");
    // create the graph executor module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    CaperaCapture cam(0, 640, 640);
    // uart ser;
    cv::Mat src;
    // bool flag = ser.uartInit("/dev/ttyAMA0", 115200);
    // assert(!flag);
    char *rcv_buf = new char[16];             
    char send_buf[16]={0x12, 0x13, 0x14, 0x12, 0x13, 0x14, 0x12, 0x13, 0x14, 0x12, 0x13, 0x14};
    while (1) //循环读取数据    
    {   
        // int len = ser.uartRecv(rcv_buf);     
        // if(len>0 && rcv_buf[0]==0x01){   
            // src=cv::imread("./src2.jpg");
            cam.getimage(src);
            assert(!src.empty());
            cv::imwrite("src.jpg", src);
            process_image(src, data); 
            //load_input_hex("./cam_image.bin", data);
            // dump_data<float>(data, "./input_cpp.bin", IMAGE_WIDTH * IMAGE_HIGHT * IMAGE_DEPTH*4);//float
            memcpy(x->data, data, IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_HIGHT*4);
            // set the right input
            long start_time = getTimeUsec();
            set_input("images", x);
            run();
            get_output(0, y);
            float* result = static_cast<float*>(y->data);
            non_max_suppression(result, output, CONF_THRES, IOU_THRES, MAX_NUM, class_num, output_shape);
            for(int i=0;i<output.size();i++){
                LOG(INFO)<<output[i].class_prob<<" label= "<<output[i].label_idx << " classes= " << label[output[i].label_idx];
            }
            LOG(INFO)<<"time is: "<<(getTimeUsec() - start_time) / 1000<<" ms";
            // ser.uartSend(send_buf, 16);
            memset(rcv_buf, 0, 16); 
        // }          
    }   
    return 0;
}
