这是一个检测扑克牌的代码，基于yolov5进行预训练，主要识别9, 10, J, Q, K, A，总共24类 

使用首先需要对anchor用kmeans算法得出适合本数据集最佳的宽与高 本代码后续会提供在芯片上部署的方法 
![image](./readme.jpg) ![image](./out_pred.jpg)

1.提供pt转pth工具 

2.提供tvm加载onnx模型的脚本以及tvm的配置与改进的方法 

3.提供c++代码加载模型的方法