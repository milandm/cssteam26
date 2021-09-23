import torch.nn as nn

from models.safe_resnet import resnet18
from models.safe_vgg16 import SafeVGG16
from models.safe_resnet_cifar import ResNet18


def build_model(model_name, num_classes):
    if model_name == 'safe_vgg16':
        return SafeVGG16(num_classes=num_classes)
    if model_name == 'safe_resnet18':
        model = resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    if model_name == 'safe_resnet18_cifar':
        model = ResNet18(num_classes)
        return model

    exit('{} model is not supported'.format(model_name))
