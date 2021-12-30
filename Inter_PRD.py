from Help_fn.mydef import *

file_list = image_list()

class Inter_points_prd(Inter_points):
    def __init__(self, file_list=file_list, dir_path=dir_path):
        self.dir_path = dir_path
        self.file_list = file_list
        self.file_list_len = len(file_list)
        self.mod = 'predict'

    def set_data(self, idx):
        if idx == 0 or idx >= self.file_list_len - 1:
            self.previous_idx = idx
            self.next_idx = idx
        else:
            self.previous_idx = idx - 1
            self.next_idx = idx + 1

        self.idx = idx
        self.previous_points = self.get_points_from_image(self.previous_idx)
        self.next_points = self.get_points_from_image(self.next_idx)


inter_points_prd = Inter_points_prd()

for i in range(inter_points_prd.file_list_len):
    if i % 50 == 0 and i:
        print(f"{i}/{inter_points_prd.file_list_len}")
    inter_points_prd.set_data(i)
    inter_points_prd.save_point_in_image()