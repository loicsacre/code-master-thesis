################################################################################################
#                                                                                              #
# Architectures similar "MatchNet" but with pre-trained tower on AlexNet                       #
#                                                                                              #
################################################################################################

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from abc import abstractmethod

num_output_channels = 3
transform_composition = []
transform_composition.append(transforms.Grayscale(
    num_output_channels=num_output_channels))
transform_composition.append(transforms.ToTensor())
# TODO: Normalize like imagenet or matchnet ??
# transform_composition.append(transforms.Normalize(
#     [0.5]*num_output_channels, [0.625]*num_output_channels))
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
transform_transfernet = transforms.Compose(transform_composition)


model_parameters = {
    "alexnet": {
        "down": (256, 16),
        "avgpool": 8,
        "fc1": (16*8*8*2, 128),
        "fc2": (128, 64),
        "fc3": (64, 2),
    },
    "vgg16_bn": {
        "down": (512, 64),
        "avgpool": 6,
        "fc1": (64*6*6*2, 64*6),
        "fc2": (64*6, 64),
        "fc3": (64, 2),
    }
}


class TransferNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.features = models.__dict__[model_name](pretrained=True).features

        for param in self.features.parameters():
            param.requires_grad = False

        params = model_parameters[model_name]

        self.down = self.conv(params["down"])
        self.avgpool = nn.AdaptiveAvgPool2d(params["avgpool"])

        # TODO: change a make wrapper for linear + relu (the model were trained on models
        # with an other implemenation without inheritance)
        self.classifier = nn.Sequential(
            nn.Linear(params["fc1"][0], params["fc1"][1]),  # fc1
            nn.ReLU(True),
            nn.Linear(params["fc2"][0], params["fc2"][1]),  # fc2
            nn.ReLU(True),
            nn.Linear(params["fc3"][0], params["fc3"][1]),  # fc3
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )

    def conv(self, channels, kernel_size=1, stride=1, padding=0):

        ch_in, ch_out = channels
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def tower(self, x):
        x = self.features(torch.squeeze(x, 1))
        x = self.down(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):

        x = torch.chunk(x, 2, dim=1)

        f = [[]]*2
        for i, xi in enumerate(x):
            f[i] = self.tower(xi)

        x = torch.cat(f, dim=1)

        return self.classifier(x)[:, 1]


class TransferAlexNet(TransferNet):
    def __init__(self):
        super().__init__("alexnet")


class TransferVGGNet(TransferNet):
    def __init__(self):
        super().__init__("vgg16_bn")

############## For evaluation #####################


class TransferNetEval(TransferNet):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.reference = None
        self.target_cache = None

    def set_reference(self, x):
        self.reference = self.tower(x)

    def reset(self):
        self.reference = None
        self.target_cache = None

    def forward_with_reference(self, x, i):

        features = None

        if self.target_cache is None:
            features = self.tower(x)
            self.target_cache = features
        elif i > self.target_cache.shape[0] - 1:
            features = self.tower(x)
            self.target_cache = torch.cat([self.target_cache, features], dim=0)
        else:
            features = torch.unsqueeze(self.target_cache[i], dim=0)

        x = torch.cat((self.reference, features), dim=1)

        return self.classifier(x)[:, 1]


class TransferAlexNetEval(TransferNetEval):

    def __init__(self):
        super().__init__("alexnet")


class TransferVGGNetEval(TransferNetEval):

    def __init__(self):
        super().__init__("vgg16_bn")


# size = 300
# a = torch.rand((1, 1, 3, size, size))
# b = torch.rand((1, 1, 3, size, size))

# c = torch.cat((a, b), dim=1)
# print(c.size())

# checkpoint = torch.load(
#     "./results/transferVggnet_0.01_0.0_64-1205591.check", map_location='cpu')
# model = TransferVGGNetEval()
# model.load_state_dict(checkpoint['state_dict'])


# model.set_reference(a)

# out = model.forward_with_reference(b, 0)
# # out = model.forward_with_reference(b, 1)
# # out = model.forward_with_reference(b, 0)


# # print(out.size())
# # out = model.forward_with_reference(b)

# # out = model(c)
# print(out.size())
