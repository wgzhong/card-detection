import torch
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
import onnx
from general import non_max_suppression
import cv2
from utils import *

# def get_calibration_dataset(mod, input_name):
#     dataset = []
#     input_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
#     for i in range(0):
#         data = np.random.uniform(size=input_shape)
#         dataset.append({input_name: data})
#     return dataset

def run_auto_quantize_pass(mod, params, dataset):
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    # dataset = get_calibration_dataset(mod, "images")
    with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                skip_conv_layers=[],
                                weight_scale="max",
                                calibrate_chunk_by=num_cpu):
        mod = relay.quantize.quantize(mod, params, dataset)
    return mod

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
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: 
        img = img.unsqueeze(0)
    # img.tofile("letterbox.bin")
    return img

def from_onnx_yolov5s(model_path, img, quant=False):
    model = onnx.load(model_path)
    input_name = 'images' 
    shape_dict = {input_name: img.shape}
    print(shape_dict)
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    net = mod['main']
    data = tvm.nd.array(img)
    dataset = [{input_name: data}]
    # mod = run_auto_quantize_pass(mod, params, dataset) 
    # if quant:
    #     data = tvm.nd.array(img)
    #     dataset = [{input_name: data}]
    #     qmod, lparams = load_model_from_json("")
    #     if qmod == None:
    #         mod = run_auto_quantize_pass(mod, lparams, dataset)
    #     else:
    #         mod = qmod
    print(mod)
    return mod, params

def export_so(mod, params, tar):
    if tar == "llvm":
        target = tvm.target.Target("llvm")
    else:
        target = tar
    with tvm.transform.PassContext(opt_level=2):
        compiled_lib = relay.build(mod, target, params=params)
    if tar == "llvm":
        compiled_lib.export_library("relay_yolov5s.so")
    else:
        compiled_lib.export_library("relay_yolov5s.so", cc="/usr/bin/arm-linux-gnueabihf-g++")

def export_three_part(mod, params, tar):
    if tar == "llvm":
        target = tvm.target.Target("llvm")
    else:
        target = tar
    # with relay.build_config(opt_level=2):
    #     graph, lib, params = relay.build_module.build(mod, target, params=params)
    with tvm.transform.PassContext(opt_level=2):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod["main"], params=params)
            graph, lib, params = relay.build_module.build(mod, target, params=params)
    libpath = "./yolo5s_poker.so"
    if tar == "llvm":
        lib.export_library(libpath)
    else:
        lib.export_library(libpath, cc="/usr/bin/arm-linux-gnueabihf-g++")
    graph_json_path = "./yolo5s_poker.json"
    with open(graph_json_path, 'w') as fo:
        fo.write(graph)
    param_path = "./yolo5s_poker.params"
    with open(param_path, 'wb') as fo:
        fo.write(relay.save_param_dict(params))
    exit(0)

def run_cpu(mod, params, data):
    target = "llvm"
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(mod, target, params=params)

    ##########################################
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("images", data)
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
    # target = tvm.target.arm_cpu("rasp3b")
    target = "llvm"
    img_size = 128
    quant = False
    img = load_image('/home/wgzhong/card-detection/tvm/data/a.jpg', img_size)
    mod, params = from_onnx_yolov5s("/home/wgzhong/card-detection/runs/exp/weights/best.onnx", img, quant)
    export_so(mod, params, target)
    # export_three_part(mod, params, target)
    output = run_cpu(mod, params, img)
    reauslt(output)

run()