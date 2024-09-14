import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class Detector:
    img_size = 640
    threshold = 0.3
    stride = 1    
    weights = './weights/yolov5m.pt'
    device = '0' if torch.cuda.is_available() else 'cpu' # cpu or gpu
    device = select_device(device)   # set to cpu or gpu
    model = attempt_load(weights, map_location=device)
    model.to(device).eval()
    model.float()
    m = model
    names = model.module.names if hasattr(model, 'module') else model.names

    
    def preprocess(img):
        if img is None:
            raise ValueError("img参数不能为空")
        img0 = img.copy()
        img = letterbox(img, new_shape=Detector.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(Detector.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(im): 
        im0, img = Detector.preprocess(im)
        pred = Detector.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, Detector.threshold, 0.4)

        boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = Detector.names[int(cls_id)]
                    if lbl not in ['person', 'bicycle']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes