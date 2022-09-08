import torch
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
import onnx
from general import non_max_suppression
import cv2
from util import *
from utils import *
 
ROOT_PATH = "/home/wgzhong/card-detection/"

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

def from_onnx_yolov5s(model_path, img, quant=False, dtype="float32"):
    model = onnx.load(model_path)
    input_name = 'images' 
    shape_dict = {input_name: img.shape}
    print(shape_dict)
    mod, params = relay.frontend.from_onnx(model, shape_dict)
    mod = mod['main']
    if dtype=="float16":
        mod = update_fp16(mod)
        for k, v in params.items():
            if v.dtype=='float32':
                v = v.asnumpy().astype(np.float16)
            params[k] = tvm.nd.array(v)
    if quant:
        dataset = [{input_name: tvm.nd.array(img)}]
        qmod, lparams = load_model_from_json("")
        if qmod == None:
            mod = run_auto_quantize_pass(mod, lparams, dataset)
        else:
            mod = qmod
    return mod, params

def run_cpu(mod, params, data):
    target = "llvm"
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(mod, target, params=params)

    ##########################################
    ctx = tvm.cpu()
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("images", tvm.nd.array(data.astype('float32')))
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
    mod, params = from_onnx_yolov5s(ROOT_PATH + "runs/exp/weights/best.onnx", img, quant, dtype)
    if only_export:
        # export_so(mod, params, name, target)
        export_three_part(mod, params, name, target)
    else:
        output = run_cpu(mod, params, input_node, img, dtype)
        reauslt(output)

run()