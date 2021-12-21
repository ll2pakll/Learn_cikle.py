import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Help_fn.mydef import *
from torchvision import models, transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

loss_fn = nn.MSELoss()
size_img = 1920

dir_path = 'd:\Work Area\Xseg_exstract\\frames\\'
num_classes = 8
feature_extract = True
model_name = "squeezenet"
savepath = savepath = 'd:\Work Area\Xseg_exstract\weights\\' + model_name + '.wgh'
spisok_loss_path = 'spisok_loss'

class CustomImageDataset(Dataset):
    def __init__(self, anatation, img_dir, transform=None, target_transform=None, make_predict=None):
        self.img_labels = anatation
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.make_predict = make_predict

    def __len__(self):
        return len(self.img_labels[0])

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_labels[0][idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels[1][idx]
        if self.make_predict:
            label = np.float32(label.ravel())  # эту строку надо отключить при обучении сети
        else:
            image, label = augumentator(image, label) #аугументатор надо отключить при получении PRD
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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    if os.path.exists(savepath):
        model_ft.load_state_dict(torch.load(savepath))
        spisok_loss = load_picle_file(spisok_loss_path)
        print(f'веса загружены из {savepath}')
    else:
        spisok_loss = [0]

    return model_ft, input_size, spisok_loss

def train_loop(dataloader, model, loss_fn, optimizer, spisok_loss=[1]):
    test_loss, b = [0] * 5, 0
    model.train()
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
    model.eval()
    with torch.no_grad():
        if make_predict:
            predict_list = []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
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
