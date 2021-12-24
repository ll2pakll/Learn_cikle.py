import cv2
import DFLIMG
import math
import albumentations as A

class Sample_maker:
    def __init__(self, img_path, pad_factor=1, resolution=512, shift_centr_factor=(1, 1), show_points=False):
        self.image = cv2.imread(img_path)
        self.points = DFLIMG.DFLJPG.load(img_path).get_dict()['target'] # потом заменить на predict
        if show_points:
            self.draw_pts()
        self.change_image = self.image.copy()
        self.change_points = self.points.copy()
        self.pad_factor = pad_factor
        self.resolution = resolution
        self.shift_centr_factor = shift_centr_factor
        self.height = int((self.change_points[3][1] - self.change_points[0][1]))

    def all_transformations(self):
        self.centr_and_rotate().crop().resize()
        return self

    def make_angle(self):
        b = self.points[3][1] - self.points[0][1]
        a = self.points[3][0] - self.points[0][0]
        angle = math.degrees(math.atan(a / b))
        return angle

    def centr_and_rotate(self):
        (y, x, d) = self.change_image.shape

        centr_y = y//2 - int(self.height*self.shift_centr_factor[1] - self.height)
        centr_x = x//2 - int(self.height*self.shift_centr_factor[0] - self.height)

        if centr_y < 0:
            y_centr = 0
        elif centr_y > y:
            y_centr = y
        else:
            y_centr = centr_y

        if centr_x < 0:
            x_centr = 0
        elif centr_x > x:
            x_centr = x
        else:
            x_centr = centr_x

        img_centr = (y_centr, x_centr)
        centr_point = (int(self.change_points[1][0]), int(self.change_points[1][1]))
        shift_y = img_centr[0] - centr_point[1]
        shift_x = img_centr[1] - centr_point[0]
        transform = A.Compose([
            A.Affine(rotate=self.make_angle(), p=1),
            A.Affine(translate_px={'x':shift_x, 'y':shift_y}, p=1)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=None),
                              p=1)
        transformed = transform(image=self.change_image, keypoints=self.change_points)
        self.change_image, self.change_points = transformed['image'], transformed['keypoints']
        return self

    def crop(self):
        a = [self.pad_factor, self.shift_centr_factor[0], self.shift_centr_factor[1]]
        for i, z in enumerate(a):
            if z < 1:
                a[i] = 1/z
        transform = A.Compose([
            A.CropAndPad(px=int(self.height*int(max(a)) - self.height) * 2, keep_size=None)
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=None),
            p=1)
        transformed = transform(image=self.change_image, keypoints=self.change_points)
        image = transformed['image']
        keypoints = transformed['keypoints']

        (y, x, d) = image.shape
        img_centr_x = x // 2
        pad_size = int(self.height * self.pad_factor - self.height)
        hight_half_and_pad = self.height//2 + pad_size
        transform = A.Compose([
            A.Crop(x_min=img_centr_x - hight_half_and_pad, y_min=int(keypoints[0][1] - pad_size),
                   x_max=img_centr_x + hight_half_and_pad, y_max=int(keypoints[3][1] + pad_size))
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=None),
            p=1)
        transformed = transform(image=image, keypoints=keypoints)
        self.change_image, self.change_points = transformed['image'], transformed['keypoints']
        return self

    def resize(self):
        transform = A.Compose([
            A.Resize(self.resolution, self.resolution)],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=None),
            p=1)
        transformed = transform(image=self.change_image, keypoints=self.change_points)
        self.change_image, self.change_points = transformed['image'], transformed['keypoints']
        return self

    def draw_pts(self):
        overlay = self.image.copy()
        colors = [[0, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0]]
        marker_radius = 10
        alpha = 0.8
        for i, point in enumerate(self.points):
            if point.any():
                overlay = cv2.circle(overlay, tuple(point), marker_radius, tuple(colors[i]), thickness=-1)
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)

    def image_return(self):
        return self.change_image

    def points_return(self):
        return self.change_points

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


name_window = 'Window'
img_path ='d:\Work Area\Xseg_exstract\\frames\\0001.jpg'
path_write = 'd:\Work Area\Xseg_exstract\\img_test.jpg'

sempler = Sample_maker(img_path, pad_factor=1.1, resolution=512, shift_centr_factor=(1.1, 1), show_points=True)

sempler.all_transformations()

image_cheng = sempler.image_return()

print(sempler.points_return())
cv2.imwrite(path_write, image_cheng)

viewImage(image_cheng, name_window)
