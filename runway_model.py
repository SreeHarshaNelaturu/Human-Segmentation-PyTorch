#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import torch, argparse
from models import UNet
import runway
import cv2, torch
import numpy as np
from time import time
from torch.nn import functional as F
from PIL import Image
from dataloaders import transforms
from utils import utils

cat = runway.category(choices=["mobilenetv2", "resnet18"], default="mobilenetv2")

@runway.setup(options={"backbone" : cat, "checkpoint" : runway.file(extension=".pth")})
def setup(opts):
    model = UNet(backbone=opts["backbone"], num_classes=2)

    if torch.cuda.is_available():
        print("Using CUDA")
        trained_dict = torch.load(opts["checkpoint"])['state_dict']
        model.load_state_dict(trained_dict, strict=False)
        model.cuda()
    else:
        print("Using CPU")
        trained_dict = torch.load(opts["checkpoint"], map_location="cpu")['state_dict']
        model.load_state_dict(trained_dict, strict=False)

    return model


inputs = {"input_image" : runway.image}
outputs = {"output_image" : runway.image}


@runway.command("segment_humans", inputs=inputs, outputs=outputs, description="Segments Humans")
def segment_humans(model, inputs):
    frame = inputs["input_image"]

    #image = frame[...,::-1]
    h, w = frame.height, frame.width

    # Predict mask
    X, pad_up, pad_left, h_new, w_new = utils.preprocessing(frame, expected_size=320, pad_value=0)

    with torch.no_grad():
        if torch.cuda.is_available():
            mask = model(X.cuda())
            mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0,1,...].cpu().numpy()
        else:
            mask = model(X)
            mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0,1,...].numpy()

    mask = 255*mask
    mask = np.expand_dims(mask, axis=2)
    image_alpha = np.concatenate((frame, mask), axis=2)
    return image_alpha.astype(np.uint8)

if __name__ == "__main__":
    runway.run(model_options={"backbone": "mobilenetv2", "checkpoint" : "UNet_MobileNetV2.pth"})
