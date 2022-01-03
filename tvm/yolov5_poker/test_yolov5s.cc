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

bool sort_score_cmp(ground_truth gt1, ground_truth gt2)
{
    return (gt1.class_prob > gt2.class_prob);
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

void non_max_suppression(float *data, float conf_thres, float iou_thres, int max_nms, int class_num, int output_num){
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
    std::vector<ground_truth> output;
    std::vector<int> idx = argsort(gt_v);
    while(idx.size() > 0){
        int k = idx[0];
        output.push_back(gt_v[k]);
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

    for(int i=0;i<output.size();i++){
        LOG(INFO)<<output[i].class_prob<<" label= "<<output[i].label_idx;
    }
}

long getTimeUsec()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (long)((long)t.tv_sec * 1000 * 1000 + t.tv_usec);
}

int main(){
    LOG(INFO) << "Running poker...";
    // load in the library
    DLDevice dev{kDLCPU, 0};
    LOG(INFO) << "load model...";
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./relay_yolov5s.so");
    // create the graph executor module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    // Use the C++ API
    LOG(INFO) << "load data...";
    std::ifstream is("./cam_image.bin", std::ifstream::in | std::ios::binary);
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
    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1,3,640,640}, DLDataType{kDLFloat, 32, 1}, dev);
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1,25200,29}, DLDataType{kDLFloat, 32, 1}, dev);

    memcpy(x->data, buffer, 3 * 640 * 640 * sizeof(float));
    // set the right input
    LOG(INFO) << "set input...";
    
    long start_time = getTimeUsec();
    set_input("images", x);
    run();
    get_output(0, y);
    float* result = static_cast<float*>(y->data);
    non_max_suppression(result, 0.001, 0.6, 3000, 24, 25200);
    printf("time is: %d ms\n", (getTimeUsec() - start_time) / 1000);
    delete [] buffer;
    return 0;
}