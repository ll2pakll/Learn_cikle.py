import random
import cv2
from matplotlib import pyplot as plt
import DFLIMG
import albumentations as A
import time
from PIL import Image
import numpy as np

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=8):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        # print(f'x = {x}, y = {y}')
    print('---------------------------------------------------------------------')

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)

img_path = 'd:\Work Area\Xseg_exstract\\frames\\0001.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dflimg = DFLIMG.DFLJPG.load(img_path)

keypoints = dflimg.get_dict()['target']

transform = A.Compose([
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), shear=(-10, 10), rotate=(-30, 30), p=0.8),
        A.Perspective(scale=(0.05, 0.1), p=1),
        A.Blur(blur_limit=(3, 20), p=0.5),
        A.CLAHE(p=0.1),
        A.Downscale(scale_min=0.1, scale_max=0.9, p=1),
        A.Equalize(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.7),
            A.ChannelDropout(p=0.05),
            A.ChannelShuffle(p=0.05)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomShadow(p=0.5)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
    p=0.8
)

while 1:
    start_time = time.time()
    transformed = transform(image=image, keypoints=keypoints)
    print((time.time() - start_time))
    print(type(transformed['image']))
    print(transformed['keypoints'])
    vis_keypoints(transformed['image'], transformed['keypoints'])
    plt.show()
