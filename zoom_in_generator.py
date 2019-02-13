from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np
import os


# config parameters
nb_zoom_level = 10
original_image_path = ''
model_path = ''
output_dir = ''
use_cuda = False
size_of_zoom_region = (256, 256)
scale_factor = 4

def open_and_crop_image(in_path):

    img = Image.open(in_path).convert('YCbCr')

    left = (img.size[0] - size_of_zoom_region[0]) // 2
    right = left + size_of_zoom_region[0]
    top = (img.size[1] - size_of_zoom_region[1]) // 2
    bottom = top + size_of_zoom_region[1]
    bbox = (left, top, right, bottom)
    img = img.crop(bbox)
    # y, cb, cr = img.split()
    # img_to_tensor = ToTensor()
    # input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])


def super_resolute_image(img, model):

    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    if use_cuda:
        input = input.cuda()

    out = model(input)

    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

# # load initial image
# img = Image.open(input_image_path).convert('YCbCr')
# y, cb, cr = img.split()

# load model
model = torch.load(model_path)


# img_to_tensor = ToTensor()
# input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

if use_cuda:
    model = model.cuda()
    input = input.cuda()


input_filename = original_image_path

for step in range(nb_zoom_level):

    img = open_and_crop_image(input_filename)
    img = super_resolute_image(img, model)
    output_filename = os.path.join(output_dir, os.path.basename(original_image_path) + "_" + str(step))
    img.save(output_filename)
    print('output image saved to ', output_filename)
    input_filename = output_filename
