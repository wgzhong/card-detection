from models.experimental import attempt_load
from nms import non_max_suppression
from datasets import letterbox
import torch
import time
from plots import output_to_target, plot_images
import cv2
import numpy as np

def load_weight(model_path, device):
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    model.eval()
    return model

def load_image(path, size):
    img0 = cv2.imread(path)  # BGR
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
    return img

def run():
    weight_path = "./runs/exp/weights/best.pt"
    device = "cpu"
    image_path="../datasets/poker/test/cam_image45.jpg"
    img_size = 640
    conf_thres = 0.001  # confidence threshold
    iou_thres = 0.6  # NMS IoU threshold

    model = load_weight(weight_path, device)

    start = time.time()
    img = load_image(image_path, img_size)
    out, train_out = model(img, augment=False)  # inference and training outputs
    out = non_max_suppression(out, conf_thres, iou_thres)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    end = time.time()
    print(end-start)
    # Plot images
    f = 'out_pred.jpg'  # predictions
    plot_images(img, output_to_target(out), image_path, f, names)

run()

