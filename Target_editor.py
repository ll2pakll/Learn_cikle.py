import cv2
import DFLIMG
import numpy as np
import os
from Help_fn.torch_model import *
import time

class PointsCreator():
    def __init__(self, dir_path, alpha=0.5, marker_radius=10):
        self.alpha = alpha
        self.marker_radius = marker_radius
        self.dir_path = dir_path
        #-----------------------------
        self.list_manager = Lists_manager(self.dir_path)
        # -----------------------------
        self.file_list = self.list_manager.get_image_list()
        self.len_file_list = self.list_manager.get_len_image_list()
        self.marked_file_list = self.list_manager.get_target_list()
        self.default_marker = np.array([[0, 0]]*4, np.int32)
        self.wnd_name = 'Main'
        #-----------------------------
        self.file_index = 0
        self.is_render = True
        self.show_inf_bar = True
        self.show_prd = 0
        #-----------------------------
        cv2.namedWindow(self.wnd_name)
        cv2.setMouseCallback(self.wnd_name, self.mouse_callback)

    def start(self):
        self.file_data_crate()
        self.make_inf_bar()
        while self.is_render:
            self.keyboard_handler()
        cv2.destroyAllWindows()
        self.write_metadata()
        print(f'{len(self.marked_file_list)} images were labeled')

    def file_data_crate(self):
        self.file_name = self.file_list[self.file_index]
        self.img_path = dir_path + self.file_name
        self.img = cv2.imread(self.img_path)
        self.img_view = self.img.copy()
        self.read_metadata()
        self.make_inf_bar()

    def draw_pts(self):
        img = self.img.copy()
        overlay = img.copy()
        colors = [[0, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0]]
        if self.show_prd:
            render_points = self.prd
        else:
            render_points = self.markers
        for i, point in enumerate(render_points):
            if point.any():
                overlay = cv2.circle(overlay, tuple(point), self.marker_radius, tuple(colors[i]), thickness=-1)
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        self.img_view = img
        self.show_window()

    def mouse_callback(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONUP:
            self.markers[self.marker_nmr] = [x, y]
            self.draw_pts()
            self.marker_nmr = (self.marker_nmr + 1) % 4

    def keyboard_handler(self):
        self.key = cv2.waitKey(1)
        if 48 < self.key < 53:
            self.marker_nmr = (self.key-49)
        elif self.key in {1, 4, 97, 100}: # 1 = 'Ctrl+a', 4 = 'Ctrl+d'
            self.cheng_nmb_img()
        elif self.key == 9:
            self.make_inf_bar()
        elif self.key == 8:
            self.del_metadata()
        elif self.key == 114:
            self.show_prd = (self.show_prd + 1) % 2
            self.make_inf_bar()
        elif self.key == 27:
            self.is_render = False

    def read_metadata(self):
        self.marker_nmr = 0
        dflimg = DFLIMG.DFLJPG.load(self.img_path)
        self.markers = self.prd = self.default_marker.copy()
        try:
            self.markers = dflimg.get_dict()['target']
        except:
            pass
        try:
            self.prd = dflimg.get_dict()['predict']
        except:
            pass
        self.draw_pts()

    def write_metadata(self):
        if self.markers.any():
            dflimg = DFLIMG.DFLJPG.load(self.img_path)
            meta = dflimg.get_dict()
            try:
                meta['target'] = self.markers
            except:
                meta = {'target': self.markers}
            dflimg.set_dict(dict_data=meta)
            dflimg.save()
            if self.marked_file_list[self.file_index][1] == None:
                self.marked_file_list[self.file_index][1] == True
            print(f'metadata write in {self.file_list[self.file_index]}:\n{self.markers}')
        else:
            print(f'The {self.file_list[self.file_index]} is not marked')

    def del_metadata(self):
        dflimg = DFLIMG.DFLJPG.load(self.img_path)
        dflimg.set_dict(dict_data={})
        dflimg.save()
        try:
            self.marked_file_list.remove(self.file_index)
        except:
            pass
        self.markers = self.default_marker.copy()
        self.draw_pts()
        self.marker_nmr = 0
        print('metadata is deleted')

    def cheng_nmb_img(self):
        self.write_metadata()
        if self.key == 100:
            index_cheng = 1
            while cv2.waitKey(1) == 100:
                index_cheng += 1
        elif self.key == 97:
            index_cheng = -1
            while cv2.waitKey(1) == 97:
                index_cheng -= 1
        elif self.key == 4: # 4 = 'Ctrl+d'
            index_cheng = self.list_manager.get_next_mamarked_idx(self.file_index)
            print(index_cheng)
            self.file_index = 0
        elif self.key == 1: # 1 = 'Ctrl+a'
            index_cheng = self.list_manager.previous_mamarked_idx(self.file_index)
            self.file_index = 0
        new_index = self.file_index + index_cheng
        if 0 <= new_index < self.len_file_list:
            self.file_index += index_cheng
            self.file_data_crate()
        elif new_index >= self.len_file_list:
            self.file_index = self.len_file_list - 1
            self.file_data_crate()
        else:
            self.file_index = 0
            self.file_data_crate()

    def show_window(self):
        cv2.imshow(self.wnd_name, self.img_view)

    def make_inf_bar(self):
        try:
            if self.key == 9:
                if self.show_inf_bar:
                    self.show_inf_bar = None
                else:
                    self.show_inf_bar = True
        except:
            pass
        if self.show_inf_bar:
            coords = [5, 5]
            rect_color = (128, 128, 128)
            text_color = (255, 255, 255)
            offset = 10
            text_size = 0.7
            shift = int(30*text_size)
            inf_data = (f'file name - {self.file_list[self.file_index]}', f'{len(self.marked_file_list)} images are labeled', 'a - previous', 'd - next', 'backspace - del markers')
            cv2.rectangle(self.img, coords, (400, 140), rect_color, -1)
            for i in inf_data:
                coords[1] += shift
                cv2.putText(self.img, i, (coords[0]+offset, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)
            if self.show_prd:
                coords[1] += shift
                cv2.putText(self.img, 'prd', (coords[0] + offset, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)
            self.draw_pts()
        else:
            self.img = cv2.imread(self.img_path)
            self.draw_pts()
dir_path = 'd:\Work Area\Xseg_exstract\\frames\\'
Creator = PointsCreator(dir_path)
points = Creator.start()