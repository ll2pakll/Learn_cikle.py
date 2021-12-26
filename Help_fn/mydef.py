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

class Lists_manager:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.image_list = image_list(self.dir_path)
        self.len_image_list = len(self.image_list)
        self.make_target_list()

    def make_target_list(self):
        self.marked_list = self.image_list.copy()
        self.len_target_list = 0
        for i, n in enumerate(self.image_list):
            img_path = self.dir_path + n
            if i % 50 == 0 and i != 0:
                print(f'{i}/{self.len_image_list}')
            try:
                DFLIMG.DFLJPG.load(img_path).get_dict()['target']
                self.marked_list[i] = [n, True, i]
                self.len_target_list += 1
            except:
                self.marked_list[i] = [n, None, i]

    def get_image_list(self):
        return self.image_list

    def get_target_list(self):
        return self.marked_list

    def get_len_image_list(self):
        return self.len_image_list

    def get_len_target_list(self):
        return self.len_target_list

    def get_next_marked_idx(self, idx):
        for i, n in enumerate(self.marked_list[idx:]):
            if n[1] == True and n[2] != idx:
                return n[2]
        return idx

    def get_previous_marked_idx(self, idx):
        for i, n in enumerate(reversed(self.marked_list[:idx])):
            if n[1] == True:
                return n[2]
        return idx

    def get_path_file(self, x):
        if x is int:
            return self.dir_path + self.image_list[x]
        if x is str:
            return self.dir_path + x

    def set_marked_fale_True(self, idx):
        self.marked_list[idx][1] = True

    def set_marked_fale_None(self, idx):
        self.marked_list[idx][1] = None


class Inter_points:
    def __init__(self, idx, previous_idx, next_idx, marker, dir_path, file_list):
        self.dir_path = dir_path
        self.file_list = file_list
        self.set_data(idx, previous_idx, next_idx, marker)


    def set_data(self, idx, previous_idx, next_idx, marker):
        self.previous_idx = previous_idx
        self.next_idx = next_idx
        self.idx = idx
        self.set_previous_points(marker)
        self.set_next_points()


    def get_points_from_image(self, idx):
        dflimg = DFLIMG.DFLJPG.load(self.dir_path+self.file_list[idx])
        return dflimg.get_dict()['target']

    def get_points(self):
        range = self.next_idx - self.previous_idx
        step = (self.next_points - self.previous_points)/range
        number_of_steps = self.idx - self.previous_idx
        self.inter_points = self.previous_points + step*number_of_steps
        return np.int16(self.inter_points)

    def set_previous_points(self, marker):
        if marker.any():
            self.previous_points = marker
            self.previous_idx = self.idx
        else:
            try:
                self.previous_points = self.get_points_from_image(self.previous_idx)
            except:
                try:
                    self.previous_points = self.get_points_from_image(self.next_idx)
                except:
                    self.previous_points = np.array([[0, 0]]*4, np.int32)

    def set_next_points(self):
        try:
            self.next_points = self.get_points_from_image(self.next_idx)
        except:
            try:
                self.next_points = self.get_points_from_image(self.previous_idx)
            except:
                self.next_points = np.array([[0, 0]]*4, np.int32)

def image_list(dir_path):
    file_list = os.listdir(path=dir_path)
    formats_tuple = ('jpg', 'png')
    for i, n in enumerate(file_list):
        if len(n.split('.')) != 2:
            file_list.pop(i)
            break
        if n.split(('.'))[1] not in formats_tuple:
            file_list.pop(i)
    return file_list

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
    vis_keypoints(transformed['image'], transformed['keypoints'], (0, 255, 0))
    plt.show()
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