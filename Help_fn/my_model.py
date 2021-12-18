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
        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        mean_layer = 1024
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*119*212, mean_layer)
        self.fc2 = nn.Linear(mean_layer, mean_layer)
        self.fc3 = nn.Linear(mean_layer, 8)

    def forward(self, x):
        # -> n, 3, 512, 512
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 254, 254
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 125, 1255
        # print(x.shape)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

def train_loop(dataloader, model, loss_fn, optimizer, spisok_loss=[1]):
    test_loss, b = [0] * 5, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss[b % 5] = (loss_fn(pred, y).item())
            b += 1
            if make_predict:
                predict_list.append(predict_to_markers(pred))
        if make_predict:
            return predict_list
    test_loss = set(test_loss)
    test_loss.discard(0)
    test_loss_med = sum(test_loss)/len(test_loss)
    spisok_loss.append(test_loss_med)
    print(f'loss: {test_loss_med:>5f}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Resize(size_img)
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
