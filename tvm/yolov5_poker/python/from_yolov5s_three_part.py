import cv2
import torch
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
from tvm.contrib import utils
from general import non_max_suppression
import time
from tvm.contrib import graph_executor

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def load_image(path, size):
    img0 = cv2.imread(path) # BGR
    assert img0 is not None, f'Image Not Found'
    # Padded resize
    img = letterbox(img0, size)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    # img = np.float32(np.array(img))
    img = img/255.0  # 0 - 255 to 0.0 - 1.0
    # print(img)
    # img.tofile("cam_image.bin")
    if img.ndimension() == 3: 
        img = img.unsqueeze(0)
    return img

def load_three_part(graph_json_path, libpath, param_path):
    loaded_json = open(graph_json_path).read()
    loaded_lib = tvm.runtime.load_module(libpath)
    loaded_params = bytearray(open(param_path, "rb").read())
    return loaded_json, loaded_lib, loaded_params

def load_graph_lib(libpath):
    loaded_lib = tvm.runtime.load_module(libpath)
    return loaded_lib

def run_cpu(graph, lib, params, data):
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.load_params(params)
    module.set_input("images", data)
    module.run()
    out_deploy = module.get_output(0).asnumpy()
    out_deploy = torch.from_numpy(out_deploy)
    print(out_deploy.shape)
    return out_deploy

def run_cpu_lib(graph_lib, name, data):
    ctx = tvm.cpu()
    m = graph_executor.GraphModule(graph_lib["default"](ctx))
    m.set_input("images", tvm.nd.array(data))
    m.run()
    out_deploy = m.get_output(0).asnumpy()
    out_deploy = torch.from_numpy(out_deploy)
    print(out_deploy)
    return out_deploy

def reauslt(out_deploy):
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold
    out = non_max_suppression(out_deploy, conf_thres, iou_thres)
    print(out)
    return out

def run():
    img_size = 640
    img = load_image('./src2.jpg', img_size)
    # graph, lib, params = load_three_part("./yolo5s_poker.json", "./yolo5s_poker.tar", "./yolo5s_poker.params")
    mod_lib = load_graph_lib("./relay_yolov5s.so")
    # mod_lib = tvm.runtime.load_module("relay_yolov5s.tar")

    start = time.time()
    output = run_cpu_lib(mod_lib, "YOLOV5S", img)
    # output = run_cpu(graph, lib, params, img)
    reauslt(output)
    end = time.time()
    print(end-start)

run()