#! /usr/bin/python
# -*- coding:UTF-8 -*-
# Convert the images from the format png to jpg using OpenCV
# Author: Jun Lou

import os, sys
import glob
from PIL import Image

png_dir = "/home/robin/Documents/ImagesPNG"
jpg_dir = "/home/robin/Documents/ImagesJPG"

if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir) 

img_lists = glob.glob(png_dir + '/*.png')

img_basenames = []
for item in img_lists:
    img_basenames.append(os.path.basename(item))

img_names = []
for item in img_basenames:
    doc_name, _ = os.path.splitext(item)
    img_names.append(doc_name)

for img in img_names:
	im = Image.open((png_dir + '/' + img + '.png'))
	rgb_im = im.convert('RGB')
	rgb_im.save((jpg_dir + '/' + img + '.jpg'))
	print('Converted image ' + img + '.png to ' + img + '.jpg')

print('Well, all done!')