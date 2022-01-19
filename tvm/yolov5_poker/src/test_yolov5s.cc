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
#include "util.h"
#include "tool.h"
#include "data_process.h"
#include "cal_time.h"
class entrance{
float CONF_THRES = 0.025;
float IOU_THRES = 0.6;
float MAX_NUM = 3000;

public:
    entrance(int w, int h, int d){
        m_width = w;
        m_hight = h;
        m_depth = d;
        m_class_num=54;
        m_dims = 4 + 1 + m_class_num;//x,y,h,w,confidence,class
        m_output_shape=(20*20+40*40+80*80)*3;//三路输出的特征图大小，每个cell有3个检测框
        m_dev = {kDLCPU, 0};
        // m_cam = new CaperaCapture(0, m_width, m_hight);
        m_data = new float[m_width * m_hight * m_depth];
        m_x = tvm::runtime::NDArray::Empty({1, m_depth, m_width, m_hight}, DLDataType{kDLFloat, 32, 1}, m_dev);
        m_y = tvm::runtime::NDArray::Empty({1, m_output_shape, m_dims}, DLDataType{kDLFloat, 32, 1}, m_dev);
    }

    void init(std::string class_path, std::string lib_path){
        LOG(INFO) << "init model..................";
        m_label = read_label(class_path);
        m_mod_factory = tvm::runtime::Module::LoadFromFile(lib_path);
        m_gmod = m_mod_factory.GetFunction("default")(m_dev);
        m_set_input = m_gmod.GetFunction("set_input");
        m_get_output = m_gmod.GetFunction("get_output");
        m_run = m_gmod.GetFunction("run");
    }

    void run(){
        LOG(INFO) << "run model..................";
        std::vector<ground_truth> output;
        cv::Mat input_image;
        // m_cam->getimage(input_image);
        input_image=cv::imread("../../data/cam_image38.jpg");//test
        assert(!m_input_image.empty());
        process_image(input_image, m_data, m_width, m_hight); 
        //load_input_hex("./cam_image.bin", data);
        // dump_data<float>(data, "./input_cpp.bin", m_width * m_hight * m_depth*4);//float
        memcpy(m_x->data, m_data, m_depth * m_width * m_hight*4);
        long start_time = getTimeUsec();
        m_set_input("images", m_x);
        m_run();
        m_get_output(0, m_y);
        float* result = static_cast<float*>(m_y->data);
        non_max_suppression(result, output, CONF_THRES, IOU_THRES, MAX_NUM, m_class_num, m_output_shape);
        m_output = output;
        LOG(INFO)<<"time is: "<<(getTimeUsec() - start_time) / 1000<<" ms";
    }

    void print_output(){
        for(int i=0;i<m_output.size();i++){
            LOG(INFO)<<m_output[i].class_prob<<" label= "<<m_output[i].label_idx << " classes= " << m_label[m_output[i].label_idx];
        }
    }

private:
    DLDevice m_dev;
    float *m_data;
    int m_class_num;
    int m_dims;
    int m_output_shape;
    int m_width;
    int m_hight;
    int m_depth;

    std::vector<ground_truth> m_output;
    tvm::runtime::NDArray m_x;
    tvm::runtime::NDArray m_y;
    std::vector<std::string> m_label;
    tvm::runtime::Module m_mod_factory;
    tvm::runtime::Module m_gmod;
    tvm::runtime::PackedFunc m_set_input;
    tvm::runtime::PackedFunc m_get_output;
    tvm::runtime::PackedFunc m_run;
    // CaperaCapture *m_cam;
};

int main(int argc, char **argv){
    if(argc != 4){
        LOG(ERROR)<<"please image size and dims";
        return -1;
    }
    int width = atoi(argv[1]);
    int hight = atoi(argv[2]);
    int depth = atoi(argv[3]);
    entrance enter(width, hight, depth);
    enter.init("../../data/classes.txt", "../python/relay_yolov5s.so");
    enter.run();
    enter.print_output();

    // uart ser;
    // bool flag = ser.uartInit("/dev/ttyAMA0", 115200);
    // assert(!flag);
    // char *rcv_buf = new char[16];             
    // char send_buf[16]={0x12, 0x13, 0x14, 0x12, 0x13, 0x14, 0x12, 0x13, 0x14, 0x12, 0x13, 0x14};
    // while (1) //循环读取数据    
    // {   
    //     int len = ser.uartRecv(rcv_buf);     
    //     if(len>0 && rcv_buf[0]==0x01){   
    //         enter.run();
    //         enter.print_output();
    //         ser.uartSend(send_buf, 16);
    //         memset(rcv_buf, 0, 16); 
    //     }          
    // }   
    return 0;
}
