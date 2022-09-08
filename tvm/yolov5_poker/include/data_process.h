#include <iostream>
#include <vector>
#include <opencv2/core.hpp> 
#include <fstream>

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

void process_image(cv::Mat src, float *data, int w, int h){
    cv::Mat dst, input;
    letterbox(src, dst, w, h);
    cv::cvtColor(dst, input, cv::COLOR_BGR2RGB);
    Mat_to_CHW(input, data, w , h);
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
        if(line.empty()){
            continue;
        }
        if(line[line.size()-1] == '\r'){
            line = line.substr(0, line.size()-1);
        }
        label.push_back(line); 
    }
    return label;
} 