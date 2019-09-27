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
    trained_dict = torch.load(opts["checkpoint"], map_location="cpu")['state_dict']
    model.load_state_dict(trained_dict, strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model



inputs = {"input_file" : runway.image}
outputs = {"output_file" : runway.image}


@runway.command("Segment Humans", inputs=inputs, outputs=outputs, description="Segments Humans")
def segment_humans(model, inputs):
	if torch.cuda.is_available() == True:
		use_cuda = True
	else:
		use_cuda = False

	frame = inputs["input_file"]
	#image = frame[...,::-1]
	h, w = frame.height, frame.width

	# Predict mask
	X, pad_up, pad_left, h_new, w_new = utils.preprocessing(frame, expected_size=320, pad_value=0)

	with torch.no_grad():
		if use_cuda:
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

	image_alpha = utils.draw_matting(frame, mask)
	final_image = Image.fromarray(image_alpha)

	#img = Image.open('img.png')
	#img = final_image.convert("RGBA")

	return final_image

if __name__ == "__main__":
    runway.run()
