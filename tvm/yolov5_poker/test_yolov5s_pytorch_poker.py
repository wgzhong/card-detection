import torch
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import numpy as np
from models.yolo import Model
from nms import non_max_suppression
from util import letterbox
import cv2
 
def load_image(path, size):
    img0 = cv2.imread(path) # BGR
    assert img0 is not None, f'Image Not Found'
    # Padded resize
    img = letterbox(img0, size, False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: 
        img = img.unsqueeze(0)
    return img

img_size = 640
img = load_image('./test.jpg', img_size)
model_weights = torch.load("../runs/exp/weights/best.pt", map_location='cpu')

# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in model_weights.items():
#     name = "model." + k
#     new_state_dict[name] = v
                
model = Model()
model.eval()
model.load_state_dict(model_weights, False)


input_data = torch.ones(img.shape)
scripted_model = torch.jit.trace(model, input_data).eval()       
# model = onnx.load('./best.onnx')

target = "llvm"
input_name = 'images' 
shape_dict = [(input_name, img.shape)]
print(shape_dict)
mod, params = relay.frontend.from_pytorch(scripted_model, shape_dict)

# shape_dict = {input_name: img.shape}
# print(shape_dict)
# mod, params = relay.frontend.from_onnx(scripted_model, shape_dict)
print(mod)

# build
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build_module.build(mod, target, params=params)
 
dtype = 'float32'
 
# save three-parts
libpath = "./yolo5s_poker.so"
lib.export_library(libpath)
#graph
graph_json_path = "./yolo5s_poker.json"
with open(graph_json_path, 'w') as fo:
    fo.write(graph)
#weight
param_path = "./yolo5s_poker.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
 
# load threee-parts
# loaded_json = open(graph_json_path).read()
# loaded_lib = tvm.module.load(libpath)
# loaded_params = bytearray(open(param_path, "rb").read())
 
# cpu run
ctx = tvm.cpu()
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input("images", img)
module.run()
out_deploy = module.get_output(0).asnumpy()
out_deploy = torch.from_numpy(out_deploy)
print(out_deploy.shape)

#nms
conf_thres = 0.001  # confidence threshold
iou_thres = 0.6  # NMS IoU threshold
out = non_max_suppression(out_deploy, conf_thres, iou_thres)

print(out)
