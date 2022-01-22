#include <iostream>
#include <vector>
#include <opencv2/core.hpp> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/highgui.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>



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


bbox xywh2xyxy(float x, float y, float w, float h, int shift){
    bbox output_box;
    output_box.x1 = x - (w / 2.0) + shift;
    output_box.x2 = x + (w / 2.0) + shift;
    output_box.y1 = y - (h / 2.0) + shift;
    output_box.y2 = y + (h / 2.0) + shift;
    return output_box;
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
    return iou;
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
        // LOG(INFO)<<center_x<<" "<<center_y<<" "<<width<<" "<<hight<<" "<<conf;
        if(conf < conf_thres){
            continue;
        }
        float max_class_conf=0.0;
        int idx=0;
        for (int i = n+5; i < n+class_num+5; i++){
            if(data[i] > max_class_conf){
                max_class_conf = data[i];
                idx = (i-5) % stride;
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
    // for(int i=0;i<idx.size();i++){
    //     LOG(INFO)<<gt_v[idx[i]].class_prob<<" "<<gt_v[idx[i]].label_idx<<" "<<gt_v[idx[i]].box.x1<<" "<<gt_v[idx[i]].box.y1<<" "<<gt_v[idx[i]].box.x2;
    // }
    while(idx.size() > 0){
        int k = idx[0];
        if(gt_v[k].class_prob > conf_thres && gt_v[k].class_prob > 0.1){
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
