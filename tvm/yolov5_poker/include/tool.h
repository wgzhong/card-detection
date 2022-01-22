#include <iostream>
#include <opencv2/core.hpp> 

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
