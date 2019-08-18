import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# Input sizes of the networks
# 'alexnet' : (224,224),
# 'densenet': (224,224),
# 'resnet' : (224,224),
# 'inception' : (299,299),
# 'squeezenet' : (224,224),#not 255,255 acc. to https://github.com/pytorch/pytorch/issues/1120
# 'vgg' : (224,224)
#


IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])


def process_image_for_imagenet(img, data_size, size=224, normalize=IMAGENET_NORMALIZE):
    """ img must be a PIL image"""

    if data_size > size:
        size = data_size

    scaler = transforms.Resize((size, size))
    to_tensor = transforms.ToTensor()

    if torch.cuda.is_available():
        return Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).cuda()
    else:
        return Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))



"""
AlexNet
"""


def get_features_alexnet(t_img, model):
    out = model(t_img).data
    return out.view(out.size(0), -1).squeeze()


"""
Dense Net
"""


def get_features_densenet(t_img, model, pooling=True):
    features = model.features(t_img)
    out = F.relu(features, inplace=True)
    if pooling:
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(
            features.size(0), -1)
    else:
        out = out.view(features.size(0), -1)
    return out.data.squeeze()


"""
VGG Net
"""


def get_features_vgg(t_img, model, pooling=True):

    # TODO: keep if using adaptative average pooling
    # if not pooling:
    #     model = model.features
    # else:
    #     model.classifier = Identity()
    out = model(t_img).data
    return out.view(out.size()[0], -1).squeeze()


"""
ResNet
"""


def get_features_resnet(t_img, model, pooling=True):
    if not pooling:
        model.avgpool = Identity()
    model.fc = Identity()  # to simulate identity (or nn.Sequential())
    return model(t_img).data.squeeze()


class FeaturesExtractor():

    def __init__(self, arch, normalize=None):
        self.arch = arch
        self.model = self._get_model()
        if torch.cuda.is_available():
            print("## CUDA available")
            self.model.cuda()
        self.model.eval()  # set evaluation mode
        if normalize is None:
            self.normalize = IMAGENET_NORMALIZE
        else:
            self.normalize = normalize

    def _get_model(self):

        model = models.__dict__[self.arch](pretrained=True)

        if self.arch == "alexnet":
            return model.features
        elif "vgg" in self.arch:
            return model.features

        return model

    def set_normalize(self, normalize):
        self.normalize = normalize

    def get_features_from_img(self, img, data_size, pooling=True):

        img = process_image_for_imagenet(
            img, data_size, normalize=self.normalize)

        if self.arch == "alexnet":
            return get_features_alexnet(img, self.model)
        elif "densenet" in self.arch:
            return get_features_densenet(img, self.model, pooling)
        elif "vgg" in self.arch:
            return get_features_vgg(img, self.model, pooling)
        elif "resnet" in self.arch:
            return get_features_resnet(img, self.model, pooling)
        # Problem with inception_v3
        # elif self.arch == "inception_v3":
        #     return get_features_inception

