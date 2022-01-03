from Help_fn.mydef import *

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
        self.default_marker = np.array([[0, 0]]*5, np.int32)
        self.wnd_name = 'Main'
        #-----------------------------
        self.file_index = 0
        self.is_render = True
        # -----------------------------
        self.show_mod = 0 # 0 - 'minual', 1 = 'inter_points', 2 = 'prd', 3 = predict_inter
        self.show_inf_bar = 1
        #-----------------------------
        self.inter_points = Inter_points(idx=self.file_index,
                                         file_list=self.file_list,
                                         dir_path=self.dir_path,
                                         previous_idx=self.list_manager.get_previous_marked_idx(self.file_index),
                                         next_idx= self.list_manager.get_next_marked_idx(self.file_index)
                                         )
        # -----------------------------
        cv2.namedWindow(self.wnd_name)
        cv2.setMouseCallback(self.wnd_name, self.mouse_callback)

    def start(self):
        self.file_data_crate()
        self.make_inf_bar()
        while self.is_render:
            self.keyboard_handler()
        cv2.destroyAllWindows()
        self.write_metadata()
        # print(f'{len(self.marked_file_list)} images were labeled')

    def file_data_crate(self):
        self.file_name = self.file_list[self.file_index]
        self.img_path = dir_path + self.file_name
        self.img = cv2.imread(self.img_path)
        self.img_view = self.img.copy()
        self.read_metadata()
        if self.show_mod == 1:
            self.inter_points.set_data(idx=self.file_index,
                                       previous_idx=self.list_manager.get_previous_marked_idx(self.file_index),
                                       next_idx=self.list_manager.get_next_marked_idx(self.file_index),
                                       marker=self.markers)
        self.make_inf_bar()

    def draw_pts(self):
        img = self.img.copy()
        overlay = img.copy()
        colors = [[0, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 255]]
        if self.show_mod == 1:
            self.alpha = 0.4
            render_points = self.inter_points.get_points()
        elif self.show_mod == 2:
            self.alpha = 0.5
            render_points = self.prd
        elif self.show_mod == 3:
            self.alpha = 0.5
            try:
                render_points = self.prd_inter
            except:
                render_points = self.default_marker
        else:
            self.alpha = 0.5
            render_points = self.markers
        for i, point in enumerate(render_points):
            if point.any():
                overlay = cv2.circle(overlay, tuple(point), self.marker_radius, tuple(colors[i]), thickness=-1)
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        self.img_view = img
        self.show_window()

    def mouse_callback(self, ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONUP:
            if self.show_mod == 2 and not self.markers.any():
                self.save_inter_points()
                self.show_mod = 0
            self.markers[self.marker_nmr] = [x, y]
            self.make_inf_bar()
            self.marker_nmr = (self.marker_nmr + 1) % 5

    def keyboard_handler(self):
        self.key = cv2.waitKey(1)
        if 48 < self.key < 54:
            self.marker_nmr = (self.key-49)
        elif self.key in {1, 4, 97, 100}: # 1 = 'Ctrl+a', 4 = 'Ctrl+d', 97 = 'a', 100 = 'd'
            self.cheng_nmb_img()
        elif self.key == 9:
            self.make_inf_bar()
        elif self.key == 8:
            self.del_metadata()
        elif self.key == 114: # 'r'
            self.show_mod = (self.show_mod + 1) % 4
            self.make_inf_bar()
        elif self.key == 115: # 's'
            self.save_inter_points()
            self.draw_pts()
        elif self.key == 27:
            self.is_render = False

    def read_metadata(self):
        self.marker_nmr = 0
        dflimg = DFLIMG.DFLJPG.load(self.img_path)
        self.markers = self.prd = self.default_marker.copy()
        try:
            self.markers = dflimg.get_dict()['keypoints']
            if len(self.markers) != 5:
                self.markers = np.append(self.markers, [[0, 0]], axis=0)
        except:
            pass
        try:
            self.prd = np.int16(dflimg.get_dict()['predict'])
        except:
            pass
        try:
            self.prd_inter = np.int16(dflimg.get_dict()['predict_inter'])
        except:
            pass

    def write_metadata(self):
        if self.markers.any():
            dflimg = DFLIMG.DFLJPG.load(self.img_path)
            meta = dflimg.get_dict()
            if not self.markers[4].any():
                self.markers[4] = (self.markers[1] + self.markers[2])/2
            try:
                meta['keypoints'] = self.markers
            except:
                meta = {'keypoints': self.markers}
            dflimg.set_dict(dict_data=meta)
            dflimg.save()
            self.list_manager.set_marked_fale_True(self.file_index)
        #     print(f'metadata write in {self.file_list[self.file_index]}:\n{self.markers}')
        # else:
        #     print(f'The {self.file_list[self.file_index]} is not marked')

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
        self.list_manager.set_marked_fale_None(self.file_index)
        print('metadata is deleted')

    def save_inter_points(self):
        self.markers = self.inter_points.get_points()
        print('Inter points save in keypoints')

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
            index_cheng = self.list_manager.get_next_marked_idx(self.file_index)
            self.file_index = 0
        elif self.key == 1: # 1 = 'Ctrl+a'
            index_cheng = self.list_manager.get_previous_marked_idx(self.file_index)
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
            if self.key == 9: # 9 - 'Tab'
                self.show_inf_bar = (self.show_inf_bar + 1) % 2
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

            show_mod_names = ('target', 'Inter', 'prd', 'prd_inter')
            coords[1] += shift
            cv2.putText(self.img, show_mod_names[self.show_mod], (coords[0] + offset, coords[1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1)

            self.draw_pts()
        else:
            self.img = cv2.imread(self.img_path)
            self.draw_pts()

Creator = PointsCreator(dir_path)
Creator.start()