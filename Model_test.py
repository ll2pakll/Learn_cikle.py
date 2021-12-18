from Help_fn.my_model import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor


def ImagetoTensor(imgpath, batch_size=batch_size, todevice=True):
    imgpath = imgpath
    batch_size = batch_size
    todevice = todevice
    img = Image.open(imgpath)
    matrix = to_tensor(img)
    matrix = torch.unsqueeze(matrix, 0)
    matrix_tuple = tuple(matrix for n in range(batch_size))
    batch_torch = torch.vstack(matrix_tuple)
    if todevice:
        batch_torch = batch_torch.to(device)
    return batch_torch

def data_creater(annotation_file, img_dir, idx=None, marker_size=10, alpha=0.5, make_lable=None):
    ''' annotation file - адрес расположения файла с метками
        img_dir - адрес папки с изображениями
        idx - порядковый номер изображения для нанесения меток, если не указано, то метки наносятся на все изображения
        marker_size - размер маркеров на изображениях, по умолчанию 10
        alpha - коэффициэнт прозначности меток, вариьирует от 0 до 1, по умолчанию = 0,5
        make_lable - если True, то функция становится генератором изобраэений с метками, если None - функция возвращает
        кортеж с тремя списками из файла с метками: первый - имена изображений, второй - пути к изображениям,
        третий - метки изображений
    '''
    annotation = pd.read_csv(annotation_file)
    idx=idx
    make_lable=make_lable
    imgs_path = []
    imgs_name = []
    lables = []
    for s in annotation.iloc:
        imgs_name.append(s[0])
        imgs_path.append(f"{img_dir}\\{str(s[0])}")
        a = []
        for j in range(1, 9):
            a.append(s[j])
        lables.append(a)
    imgs_path, imgs_name, lables = np.array(imgs_path), np.array(imgs_name), np.array(lables)
    if make_lable:
        def make_lable_img(idx=idx, marker_size=marker_size, alpha=alpha, imgs_path=imgs_path, imgs_name=imgs_name, lables=lables, img_dir_train=img_dir_train):
            cikle = None
            if idx == None:
                idx = 0
                cikle = True
            img_lable_dir = f'{img_dir_train}\\img_lable'
            img_lable_prd_dir = f'{img_dir_train}\\img_lable_prd'
            try:
                os.mkdir(img_lable_dir)
            except:
                pass
            try:
                os.mkdir(img_lable_prd_dir)
            except:
                pass
            def support(id, prd=None, lables=lables):
                lables = lables
                id = id
                img = cv2.imread(imgs_path[id])
                overlay = img.copy()
                if prd:
                    print(lables[id])
                    with torch.no_grad():
                        img_batch = ImagetoTensor(imgs_path[id])
                        x = model(img_batch)
                        x = x * size_img
                        lables = x.type(torch.int32).to("cpu").numpy()
                    c = id
                    id = 0
                    print(lables[id])
                if lables[id][0] + lables[id][1] >= 1:
                    cv2.circle(overlay, (lables[id][0], lables[id][1]), marker_size, (0, 255, 255), thickness=-1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                if lables[id][2] + lables[id][3] >= 1:
                    cv2.circle(overlay, (lables[id][2], lables[id][3]), marker_size, (255, 0, 0), thickness=-1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                if lables[id][4] + lables[id][5] >= 1:
                    cv2.circle(overlay, (lables[id][4], lables[id][5]), marker_size, (0, 0, 255), thickness=-1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                if lables[id][6] + lables[id][7] >= 1:
                    cv2.circle(overlay, (lables[id][6], lables[id][7]), marker_size, (0, 255, 0), thickness=-1)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                if prd:
                    id = c
                    cv2.imwrite(f"{img_lable_prd_dir}\\{imgs_name[id]}", img)
                else:
                    cv2.imwrite(f"{img_lable_dir}\\{imgs_name[id]}", img)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if cikle:
                for i, t in enumerate(imgs_name):
                    support(i)
                    support(i, prd=True)
            else:
                figure = plt.figure(figsize=(40, 40))
                plt.title(str(f"""
                        {imgs_name[idx]}
                        {lables[idx]}"""))
                figure.add_subplot(2, 1, 1)
                plt.imshow(support(idx))
                plt.axis("off")
                figure.add_subplot(2, 1, 2)
                plt.imshow(support(idx, prd=True))
                plt.axis("off")
                plt.show()
        make_lable_img()
    else:
        return (imgs_name, imgs_path, lables)



model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

annotation_file_train = "d:\Work Area\gatasets\lessons\\annotations_file_train.csv"
img_dir_train = "d:\Work Area\gatasets\lessons\dir_img_train"

trein_data_loader = data_creater(annotation_file_train, img_dir_train, make_lable=True)