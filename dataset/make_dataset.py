import os
import numpy as np
from pycocotools.coco import COCO
from skimage import io, transform


# Crop center part of image
def crop_center(img):
    m = min(img.shape[0], img.shape[1])
    x = abs(img.shape[0] - img.shape[1])/2
    if img.shape[0] >= img.shape[1]:
        return img[x : m+x, :, :]
    else:
        return img[:, x : m+x, :]


pic_dir = './pic/train/'
dataset_nm = './imgdata.dat'

f = open(dataset_nm, 'w+')
f.truncate()
for file in os.listdir(pic_dir):
	I = io.imread(pic_dir + file)
	# Transformed version is already normalized
	I = transform.resize(crop_center(I), (256, 256))
	f.write(I.tobytes())

f.close()