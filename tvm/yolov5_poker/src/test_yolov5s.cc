#include "tvm_runtime_pack.h"
#include "test_uart.h"
#include "test_opencv.h"
#include "tool.h"
#include "data_process.h"
#include "cal_time.h"
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
class entrance{
float CONF_THRES = 0.025;
float IOU_THRES = 0.6;
float MAX_NUM = 3000;

public:
    entrance(int w, int h, int d){
        m_output.clear();
        m_width = w;
        m_hight = h;
        m_depth = d;
        m_class_num=54;
        m_data = new float[m_width * m_hight * m_depth];
        m_dims = 4 + 1 + m_class_num;//x,y,h,w,confidence,class
        // m_output_shape=(20*20+40*40+80*80)*3;//三路输出的特征图大小，每个cell有3个检测框
        m_output_shape = 1008;
        m_dev = {kDLCPU, 0};
        // m_cam = new CaperaCapture(0, m_width, m_hight);
        m_x = tvm::runtime::NDArray::Empty({1, m_depth, m_width, m_hight}, DLDataType{kDLFloat, 32, 1}, m_dev);
        m_y = tvm::runtime::NDArray::Empty({1, m_output_shape, m_dims}, DLDataType{kDLFloat, 32, 1}, m_dev);
    }

    void init(std::string class_path, std::string lib_path){
        LOG(INFO) << "init model..................";
        m_label = read_label(class_path);
        LOG(INFO) << "read_label..................";
        m_mod_factory = tvm::runtime::Module::LoadFromFile(lib_path);
        m_gmod = m_mod_factory.GetFunction("default")(m_dev);
        m_set_input = m_gmod.GetFunction("set_input");
        m_get_output = m_gmod.GetFunction("get_output");
        m_run = m_gmod.GetFunction("run");
    }

    void init(std::string class_path, std::string graph_path, 
                std::string lib_path, std::string params_path){
        LOG(INFO) << "init model..................";
        m_label = read_label(class_path);
        LOG(INFO) << "read_label..................";
        tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path) ;
        std::ifstream json_in(graph_path);
        if(json_in.fail()){
            throw std::runtime_error("could not open json file");
        }
        std::ifstream params_in(params_path, std::ios::binary);
        if(params_in.fail()){
            throw std::runtime_error("could not open json file");
        }
        const std::string json_data((std::istreambuf_iterator<char>(json_in)),
                               std::istreambuf_iterator<char>());
        json_in.close();
        const std::string params_data((std::istreambuf_iterator<char>(params_in)),
                                        std::istreambuf_iterator<char>());
        params_in.close();
        TVMByteArray params_arr;
        params_arr.data = params_data.c_str();
        params_arr.size = params_data.length();
        int device_type = kDLCPU;
        int device_id = 0;
        tvm::runtime::Module mod = 
           (*tvm::runtime::Registry::Get("tvm.graph_executor.create"))(json_data,
                                                                mod_dylib, device_type, device_id);
        tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
        load_params(params_arr);
        m_set_input = mod.GetFunction("set_input");
        m_get_output = mod.GetFunction("get_output");
        m_run = mod.GetFunction("run");    
        LOG(INFO) << "init done..................";                                                    
    }
    void run_cam(){
        cv::Mat input_image;
        // m_cam->getimage(input_image);
        input_image=cv::imread("/home/wgzhong/datasets/poker_new/train/images/poker_0.jpg");//test
        assert(!m_input_image.empty());
        process_image(input_image, m_data, m_width, m_hight); 
        //load_input_hex("/home/wgzhong/card-detection/tvm/yolov5_poker/python/cam_image.bin", data);
        // dump_data<float>(m_data, "./input_cpp.bin", m_width * m_hight * m_depth*4);//float
        memcpy(m_x->data, m_data, m_depth * m_width * m_hight*2);
    }
    void run(tvm::runtime::NDArray x){
        LOG(INFO) << "run model..................";
        long start_time = getTimeUsec();
        std::vector<ground_truth> output;
        m_set_input("images", x);
        m_run();
        m_get_output(0, m_y);
        float* result = static_cast<float*>(m_y->data);
        non_max_suppression(result, output, CONF_THRES, IOU_THRES, MAX_NUM, m_class_num, m_output_shape);
        m_output = output;
        LOG(INFO)<<"time is: "<<(getTimeUsec() - start_time) / 1000<<" ms";
    }

    void print_output(){
        if(m_output.size()>0){
            for(int i=0;i<1;i++){
                LOG(INFO)<<m_output[i].class_prob<<" label= "<<m_output[i].label_idx << " classes= " << m_label[m_output[i].label_idx];
            }
        }
    }
    
    tvm::runtime::NDArray get_x(){
        return m_x;
    }

    std::vector<std::string> get_label(){
        return m_label;
    }

    std::vector<ground_truth> get_output(){
        return m_output;
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
    std::vector<std::string> m_label;
    tvm::runtime::Module m_mod_factory;
    tvm::runtime::Module m_gmod;
    tvm::runtime::PackedFunc m_set_input;
    tvm::runtime::PackedFunc m_get_output;
    tvm::runtime::PackedFunc m_run;
    tvm::runtime::NDArray m_x;
    tvm::runtime::NDArray m_y;

    // CaperaCapture *m_cam;
};

int main(int argc, char **argv){
    if(argc < 4){
        LOG(ERROR)<<"please image size and dims";
        LOG(INFO)<<"such as ./delopy 128 128 3 test, test can remove";
        return -1;
    }
    bool test = false;
    int width = atoi(argv[1]);
    int hight = atoi(argv[2]);
    int depth = atoi(argv[3]);
    if(argc == 5){
        test = true;
    }
    entrance enter(width, hight, depth);
    
    enter.run_cam();
    // enter.init("./classes.txt", "./relay_yolov5s.so");
    enter.init("./classes.txt", "./3p/yolov5s_poker_llvm.json", "./3p/yolov5s_poker_llvm.so", "./3p/yolov5s_poker_llvm.params");
    if(test){
        enter.run(enter.get_x());
        enter.print_output();
        return 0;
    }
    // enter.run(enter.get_x());
    // enter.print_output();  
    // uart *ser = new uart(8, 32);
    // bool flag = ser->uartInit((char*)"/dev/ttyAMA0", 115200);
    // assert(!flag); 
    // LOG(INFO)<<"init ok";   
    
    // while (1)    
    // {   
    //     enter.run_cam();
    //     bool uart_flag = ser->uartRecv();    
    //     if(uart_flag && ser->isCorrect()){  
    //         enter.run(enter.get_x());
    //         std::vector<ground_truth> output = enter.get_output();
    //         std::vector<std::string> labels = enter.get_label();
    //         ser->uartSend(output, labels);
    //         enter.print_output();
    //     }           
    // }   
    return 0;
}


