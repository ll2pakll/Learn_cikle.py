from Help_fn.mydef import *

file_list = image_list()

class Inter_points_prd(Inter_points):
    def __init__(self, file_list=file_list, dir_path=dir_path, step=1, shift_factor=0.4):
        self.dir_path = dir_path
        self.file_list = file_list
        self.file_list_len = len(file_list)
        self.step = step
        self.mod = 'predict'
        self.target_shift_factor = shift_factor
        self.save_mod = self.mod + '_inter'

    def set_data(self, idx):
        if self.step <= idx < (self.file_list_len - self.step):
            self.previous_idx = idx - self.step
            self.next_idx = idx + self.step
        else:
            self.previous_idx = idx
            self.next_idx = idx

        self.idx = idx
        if self.previous_idx:
            self.previous_points = self.get_points_from_image(idx=self.previous_idx, try_get_inter=True)
        else:
            self.previous_points = self.get_points_from_image(idx=self.previous_idx)
        self.next_points = self.get_points_from_image(idx=self.next_idx)

    def get_points_from_image(self, idx=None, try_get_inter=None):
        if idx == None:
            idx = self.idx
        dflimg = DFLIMG.DFLJPG.load(self.dir_path+self.file_list[idx])
        if try_get_inter:
            try:
                return dflimg.get_dict()[self.save_mod]
            except:
                pass
        return dflimg.get_dict()[self.mod]

    def shift_factor(self):
        height_prev = int(max([(self.previous_points[3][1] - self.previous_points[0][1]),
                               (self.previous_points[3][0] - self.previous_points[0][0])]))
        height_next = int(max([(self.next_points[3][1] - self.next_points[0][1]),
                               (self.next_points[3][0] - self.next_points[0][0])]))
        height = min([height_prev, height_next])
        shift = self.previous_points - self.next_points
        shift_factor = abs(np.sum(shift)/height)
        return shift_factor

    def save_point_in_image(self, idx=None):
        if idx == None:
            idx = self.idx
        dflimg = DFLIMG.DFLJPG.load(self.dir_path + self.file_list[idx])
        meta = dflimg.get_dict()
        if self.shift_factor() < self.target_shift_factor:
            try:
                meta[self.save_mod] = self.get_points()
            except:
                meta = {self.save_mod: self.get_points()}
        else:
            try:
                meta[self.save_mod] = meta[self.mod].copy()
            except:
                meta = {self.save_mod: meta[self.mod].copy()}
        dflimg.set_dict(dict_data=meta)
        dflimg.save()


inter_points_prd = Inter_points_prd(step=1)

for i in range(inter_points_prd.file_list_len):
    if i % 50 == 0 and i:
        print(f"{i}/{inter_points_prd.file_list_len}")
    inter_points_prd.set_data(i)
    inter_points_prd.save_point_in_image()
    if inter_points_prd.shift_factor() >= 0.4:
        print(f'{file_list[i]} - {inter_points_prd.shift_factor()}')

# inter_points_prd.set_data(145)
# print(inter_points_prd.shift_factor())