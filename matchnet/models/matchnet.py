import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms

################################################################################################
#                                                                                              #
# Architectures from "MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching" #
#                                                                                              #
################################################################################################

# See https://github.com/hanxf/matchnet/tree/master/models

num_output_channels = 1
resize = 64
transform_composition = []
transform_composition.append(transforms.Grayscale(
    num_output_channels=num_output_channels))
transform_composition.append(transforms.Resize((resize, resize)))
transform_composition.append(transforms.ToTensor())
transform_composition.append(transforms.Normalize(
    [0.5]*num_output_channels, [0.625]*num_output_channels))
transform_matchnet = transforms.Compose(transform_composition)


class FeaturesNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            self.conv(1, 24, kernel_size=7, stride=1, padding=3),  # conv0
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool0
            self.conv(24, 64, kernel_size=5, stride=1, padding=2),  # conv1
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # pool1
            self.conv(64, 96, kernel_size=3, stride=1, padding=1),  # conv2
            self.conv(96, 96, kernel_size=3, stride=1, padding=1),  # conv3
            self.conv(96, 64, kernel_size=3, stride=1, padding=1),  # conv4
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # pool4
        )

    def conv(self, ch_in, ch_out, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)  # size : 64x8x8
        x = x.view(x.size(0), -1)
        return x


class MatchNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = FeaturesNet()

        self.classifier = nn.Sequential(
            nn.Linear(64*8*8*2, 64*8*2),  # fc1  (8192, 1024)
            nn.ReLU(True),
            nn.Linear(64*8*2, 64*8*2),  # fc2 (1024, 1024)
            nn.ReLU(True),
            nn.Linear(64*8*2, 2),  # fc3 (1024, 2)
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = torch.chunk(x, 2, dim=1)

        x = torch.cat((self.features(torch.squeeze(x[0], 1)),
                       self.features(torch.squeeze(x[1], 1))),
                      dim=1)

        return self.classifier(x)[:, 1]  # return only the second column


class MatchNetEval(MatchNet):
    def __init__(self):
        super().__init__()

        self.reference = None
        self.target_cache = None

    def set_reference(self, x):
        self.reference = self.features(x)

    def reset(self):
        self.reference = None
        self.target_cache = None

    def forward_with_reference(self, x, i):

        features = None

        if self.target_cache is None:
            features = self.features(x)
            self.target_cache = features
        elif i > self.target_cache.shape[0] - 1:
            # print("## In cache")
            features = self.features(x)
            self.target_cache = torch.cat([self.target_cache, features], dim=0)
        else:
            # print("## From cache")
            features = torch.unsqueeze(self.target_cache[i], dim=0)

        x = torch.cat((self.reference,
                       features),
                      dim=1)

        return self.classifier(x)[:, 1]
