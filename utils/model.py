
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
import torchvision.models as models
from utils.utility import load_config

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomCNN(nn.Module):
    class CustomCNN(nn.Module):
        def __init__(self, input_channels, num_classes, number_channels=3):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(16)  
            self.fc1 = nn.Linear(number_channels * 16 * 16, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x






class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomCNN(nn.Module):
    class CustomCNN(nn.Module):
        def __init__(self, input_channels, num_classes, number_channels=3):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(16) 
            self.fc1 = nn.Linear(number_channels * 16 * 16, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)
            return x





def get_model(model_name, dataset_name):


    if dataset_name == 'mnist':
            input_size = 224 * 224  # 28x28
            num_classes = 10  
            number_channels = 3
    elif dataset_name == 'emnist':
            input_size = 224 * 224 
            num_classes = 47  
            number_channels = 3
    elif dataset_name in ['cifar10', 'cifar100']:
            input_size = 224 * 224 * 3
           
            number_channels = 3
            num_classes = 10 if dataset_name == 'cifar10' else 100
    elif dataset_name == 'tiny_imagenet':
            input_size = 224 * 224 * 3  
            num_classes = 200  
            number_channels = 3
    elif dataset_name == 'gtsrb':
            input_size = 3*224*224
            num_classes = 43 
            number_channels = 3

    if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,  num_classes)
            return model
    elif model_name == 'resnet34':
            model = models.resnet34(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            return model
    elif model_name == 'mobilenet':

            model = models.mobilenet_v2(weights=None, num_classes=num_classes)
            return model



if __name__ == '__main__':
    path = 'config.yaml'
    config = load_config(path)
    





