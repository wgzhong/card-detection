import os
import sys
ROOT_PATH = "/home/wgzhong/card-detection/"
sys.path.append(ROOT_PATH)
sys.path.append(ROOT_PATH+"models")
import torch
from models.experimental import attempt_load
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
from models.yolo import Model, Detect
from nms import non_max_suppression
from util import *
from utils import *
import cv2
 
def load_image(path, size):
    img0 = cv2.imread(path) # BGR
    assert img0 is not None, f'Image Not Found'
    # Padded resize
    img = letterbox(img0, size)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: 
        img = img.unsqueeze(0)
    # img.tofile("letterbox.bin")
    img = np.array(img)
    return img

def from_pytorch_yolov5s(model_path, img, input_node, quant=False, dtype="float32"):
    model = attempt_load(model_path, device = "cpu", inplace = True, fuse = True)
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True          
    input_data = torch.randn(img.shape)
    for _ in range(2):
        y = model(input_data) 
    scripted_model = torch.jit.trace(model, input_data, strict=False).eval()   
    shape_dict = [(input_node, img.shape)]
    print(shape_dict)
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)
    
    if quant:
        dataset = [{input_node: tvm.nd.array(img)}]
        qmod, lparams = load_model_from_json("")
        if qmod == None:
            mod = run_auto_quantize_pass(mod, lparams, dataset)
        else:
            mod = qmod
    return mod, params

def run_cpu(mod, params, input_node, data, dtype="float32"):
    target = "llvm"
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(mod, target, params=params)
    ##########################################
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input(input_node, tvm.nd.array(data.astype(dtype)))
    module.run()
    out_deploy = module.get_output(0).asnumpy()
    out_deploy = torch.from_numpy(out_deploy)
    print(out_deploy.shape)
    return out_deploy

def reauslt(out_deploy):
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold
    out = non_max_suppression(out_deploy, conf_thres, iou_thres)
    print(out)
    return out

def run():
    target = "rasp3b"
    # target = "llvm"
    dtype = "float32"
    name = "yolov5s_poker"
    input_node = "images"
    img_size = 128
    quant = False
    only_export = True
    img = load_image(ROOT_PATH + 'tvm/data/poker_452.jpg', img_size)
    mod, params = from_pytorch_yolov5s(ROOT_PATH + "runs/exp/weights/best.pt", img, input_node, quant, dtype)
    if only_export:
        # export_so(mod, params, name, target)
        export_three_part(mod, params, name, target)
    else:
        output = run_cpu(mod, params, input_node, img, dtype)
        reauslt(output)

run()
