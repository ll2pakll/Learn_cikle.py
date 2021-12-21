import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Help_fn.mydef import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

loss_fn = nn.MSELoss()
size_img = 1920

dir_path = 'd:\Work Area\Xseg_exstract\\frames\\'

class CustomImageDataset(Dataset):
    def __init__(self, anatation, img_dir, transform=None, target_transform=None):
        self.img_labels = anatation
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels[0])

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_labels[0][idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels[1][idx]
        image, label = augumentator(image, label) #аугументатор надо отключить при получении PRD
        # label = np.float32(label.ravel()) # эту строку надо отключить при обучении сети
        label = torch.from_numpy(label)
        label /= size_img
        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        # print(type(image))
        # print(image.dtype)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label[:2], label[2:4], label[4:6], label[6:]

class NeuralNetwork(nn.Module):
    def __init__(self):
        mean_layer = 512
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(807296, mean_layer)
        self.fc2 = nn.Linear(mean_layer, mean_layer)
        self.fc3_1 = nn.Linear(mean_layer, 2)
        self.fc3_2 = nn.Linear(mean_layer, 2)
        self.fc3_3 = nn.Linear(mean_layer, 2)
        self.fc3_4 = nn.Linear(mean_layer, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pt_1 = self.fc3_1(x)
        pt_2 = self.fc3_2(x)
        pt_3 = self.fc3_3(x)
        pt_4 = self.fc3_4(x)
        return [pt_1, pt_2, pt_3, pt_4]

def train_loop(dataloader, model, loss_fn, optimizer, spisok_loss=[1]):
    test_loss, b = [0] * 5, 0
    for batch, (X, pt1, pt2, pt3, pt4) in enumerate(dataloader):
        X, pt1, pt2, pt3, pt4 = X.to(device), pt1.to(device), \
                                pt2.to(device), pt3.to(device), pt4.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred[0], pt1) + loss_fn(pred[1], pt2) + \
               loss_fn(pred[2], pt3) + loss_fn(pred[3], pt4)
        test_loss[b % 5] = (loss.item())
        b += 1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_loss = set(test_loss)
    test_loss.discard(0)
    test_loss_med = sum(test_loss) / len(test_loss)
    spisok_loss.append(test_loss_med)
    print(f'loss: {test_loss_med:>5f}')

def test_loop(dataloader, model, loss_fn, spisok_loss=[1], make_predict=None):
    test_loss, b = [0]*5, 0

    with torch.no_grad():
        if make_predict:
            predict_list = []
        for batch, (X, pt1, pt2, pt3, pt4) in enumerate(dataloader):
            X, pt1, pt2, pt3, pt4 = X.to(device), pt1.to(device), \
                                    pt2.to(device), pt3.to(device), pt4.to(device)
            pred = model(X)
            if make_predict:
                predict_list.append(predict_to_markers(pred))
        if make_predict:
            return predict_list
    test_loss = set(test_loss)
    test_loss.discard(0)
    test_loss_med = sum(test_loss)/len(test_loss)
    spisok_loss.append(test_loss_med)
    print(f'loss: {test_loss_med:>5f}')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose(
    [transforms.ToTensor(),
     normalize
     ])
# anatation = list_marked_filse_names(dir_path)
# save_picle_file(anatation, 'anatation')

anatation = load_picle_file('anatation')

# загрузка датасетов
training_data = CustomImageDataset(
    anatation=anatation,
    img_dir=dir_path,
    transform=transform
)

test_data = CustomImageDataset(
    anatation=anatation,
    img_dir=dir_path,
    transform=transform
)
