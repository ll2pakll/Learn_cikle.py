import os
import DFLIMG
import pickle
from torchvision.transforms.functional import to_tensor
import torch
from PIL import Image
import numpy as np
import albumentations as A
import cv2
from matplotlib import pyplot as plt


def dict_marked_filse(dir_path):
    file_list = os.listdir(path=dir_path)
    formats_tuple = ('jpg', 'png')
    for i, n in enumerate(file_list):
        if len(n.split('.')) != 2:
            file_list.pop(i)
            break
        if n.split(('.'))[1] not in formats_tuple:
            file_list.pop(i)
    len_file_list = len(file_list)
    marked_list = []
    print(f'find {len_file_list} files')
    print('looking for labelled images')
    for i, n in enumerate(file_list):
        img_path = dir_path + n
        if i % 50 == 0 and i:
            print(f'scanned {i}/{len_file_list}')
        try:
            DFLIMG.DFLJPG.load(img_path).get_dict()['target']
            marked_list.append(i)
        except:
            pass
    print(f'find {len(marked_list)}/{len_file_list} marked fils' )
    return file_list, len_file_list, marked_list

def list_marked_filse_names(dir_path, all_files=None):
    file_list = os.listdir(path=dir_path)
    formats_tuple = ('jpg', 'png')
    for i, n in enumerate(file_list):
        if len(n.split('.')) != 2:
            file_list.pop(i)
            break
        if n.split(('.'))[1] not in formats_tuple:
            file_list.pop(i)
    len_file_list = len(file_list)
    file_names = []
    targets = []

    print(f'find {len_file_list} files')
    if not all_files:
        print('looking for labelled images')
    for i, n in enumerate(file_list):
        img_path = dir_path + n
        if i % 50 == 0 and i:
            print(f'scanned {i}/{len_file_list}')
        try:
            targets.append(DFLIMG.DFLJPG.load(img_path).get_dict()['target'])
            file_names.append(n)
        except:
            if all_files:
                targets.append(np.array([[0, 0]]*4, np.int32))
                file_names.append(n)
    if all_files:
        print(f'find {len_file_list}')
    else:
        print(f'find {len(file_names)}/{len_file_list} marked fils' )
    return (file_names, targets)

def next_index(index, list, next = None, previous=None):
    if previous:
        for ind in reversed(list):
            if index > ind:
                return ind
    elif next:
        for ind in list:
            if index < ind:
                return ind
    else:
        for i, ind in enumerate(list):
            if index < ind:
                return i
    return index

def save_picle_file(obgect, path):
    with open(path, 'wb') as f:
        pickle.dump(obgect, f)

def load_picle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def ImagetoTensor(imgpath, batch_size, todevice=True):
    imgpath = imgpath
    batch_size = batch_size
    todevice = todevice
    img = Image.open(imgpath)
    matrix = to_tensor(img)
    matrix = torch.unsqueeze(matrix, 0)
    matrix_tuple = tuple(matrix for n in range(batch_size))
    batch_torch = torch.vstack(matrix_tuple)
    if todevice:
        batch_torch = batch_torch.to('cuda' if torch.cuda.is_available() else 'cpu')
    return batch_torch

def predict_to_markers(predict, size_img=1920):
    pred = (predict * size_img).type(torch.int32).to("cpu").numpy()
    pred = np.ravel(pred).reshape(4, 2)
    return pred

def save_preduct_in_metadata(img_path, pred):
    dflimg = DFLIMG.DFLJPG.load(img_path)
    meta_data_dict = dflimg.get_dict()
    try:
        meta_data_dict['predict'] = pred
    except:
        meta_data_dict = {'predict': pred}
    dflimg.set_dict(dict_data=meta_data_dict)
    dflimg.save()

def augumentator(image, keypoints):
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
        # A.RandomShadow(p=0.5)
    ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=None),
        p=0.8
    )
    transformed = transform(image=image, keypoints=keypoints)
    # vis_keypoints(transformed['image'], transformed['keypoints'], (0, 255, 0))
    # plt.show()
    keypoints = []
    for i in transformed['keypoints']:
        for j in i:
            keypoints.append(j)
    keypoints = np.array(keypoints, np.float32)
    return transformed['image'], keypoints

def vis_keypoints(image, keypoints, color, diameter=8):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        print(f'x = {x}, y = {y}')
    print('---------------------------------------------------------------------')

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(image)