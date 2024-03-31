from models.models import convert, Darknet, load_darknet_weights
import torch
import numpy as np

import torch
import numpy as np


if __name__ == "__main__":
    path_to_yolo_cfg = "/home/c3po/Documents/project/learning/amar-works/PyTorch_YOLOv4_meamarp/cfg/yolov3-spp.cfg"
    path_to_yolo_weights = "/home/c3po/Documents/project/learning/amar-works/PyTorch_YOLOv4_meamarp/weights/yolov3-spp.weights"

    
    
    model = Darknet(path_to_yolo_cfg)
    load_darknet_weights(model, path_to_yolo_weights)
    torch.save(model.state_dict(), "yolov3_converted.pth")
    # Save the model
    # torch.save(model.state_dict(), "yolov3_converted.pth")

    # convert(cfg=path_to_yolo_cfg, weights=path_to_yolo_weights)
