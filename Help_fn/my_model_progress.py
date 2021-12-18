from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch import nn
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

batch_size = 3
loss_fn = nn.MSELoss()
size_img = 1920

annotation_file_train = "d:\Work Area\gatasets\lessons\\annotations_file_train.csv"
annotation_file_test = "d:\Work Area\gatasets\lessons\\annotations_file_test.csv"
img_dir_train = "d:\Work Area\gatasets\lessons\dir_img_train"
img_dir_test = "d:\Work Area\gatasets\lessons\dir_img_test"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        a = []
        for j in range(1, 9):
            a.append(self.img_labels.iloc[idx, j])
        label = torch.tensor(a).type(torch.float32)
        label /= size_img
        # print(label.shape)
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
        mean_layer = 512
        chnl1 = 6
        chnl2 = int(chnl1*1.3)
        chnl3 = int(chnl2*2)
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, chnl1, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(chnl1, chnl2, 3)
        self.conv3 = nn.Conv2d(chnl2, chnl3, 3)
        self.fc1 = nn.Linear(14*131*236, mean_layer)
        self.fc2 = nn.Linear(mean_layer, mean_layer)
        self.fc3 = nn.Linear(mean_layer, 8)

    def forward(self, x):
        x = self.pool2(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        # x = x.view(-1, 16*42*75)            # -> n, 400
        x = x.flatten(1)
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print(X.shape)
        # print(type(X))
        # print(X.dtype)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}")

def test_loop(dataloader, model, loss_fn, spisok_loss):
    test_loss, b = [0]*5, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss[b % 5] = (loss_fn(pred, y).item())
            b += 1
    test_loss = set(test_loss)
    test_loss.discard(0)
    test_loss_med = sum(test_loss)/len(test_loss)
    spisok_loss.append(test_loss_med)
    print(f'loss: {test_loss_med:>5f}')

transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Resize(size_img)
     ])

# загрузка датасетов
training_data = CustomImageDataset(
    annotations_file=annotation_file_train,
    img_dir=img_dir_train,
    transform=transform

)

test_data = CustomImageDataset(
    annotations_file=annotation_file_test,
    img_dir=img_dir_test,
    transform=transform

)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)