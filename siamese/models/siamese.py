################################################################################################
#                                                                                              #
# Architectures similar "MatchNet" but with pre-trained tower on AlexNet                       #
#                                                                                              #
################################################################################################

import torch
import torch.nn as nn
import torchvision.models as models

# TODO: adapt like matchnet/models/transfernet, use inheritance

class SiameseAlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.reference = None

        self.features = models.__dict__["alexnet"](pretrained=False).features

        self.down = self.conv(256, 128)
        self.avgpool = nn.AdaptiveAvgPool2d(8)

        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 64*8*8),  # fc1  (8192, 1024)
            nn.ReLU(True),
            nn.Linear(64*8*8, 64*8*8),  # fc2 (1024, 1024)
            nn.ReLU(True),
            nn.Linear(64*8*8, 64*8*4)
        )

    def conv(self, ch_in, ch_out, kernel_size=1, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward_triplet(self, x1, x2, x3):

        x1 = self.forward(x1)
        x2 = self.forward(x2)
        x3 = self.forward(x3)
        return x1, x2, x3

    def forward(self, x):

        x = self.features(torch.squeeze(x, 1))
        x = self.down(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalization
        x = x/x.pow(2).sum(1, keepdim=True).sqrt()
        return x


class SiameseAlexNetEval(SiameseAlexNet):

    def __init__(self):
        super().__init__()

        self.reference = None
        self.target_cache = None

    def set_reference(self, x):
        self.reference = self.forward(x)

    def reset(self):
        print("## Reseting model", flush=True)
        self.reference = None
        self.target_cache = None

    def forward_with_reference(self, x, i):

        if self.target_cache is None:
            self.target_cache = self.forward(x)
        elif i > self.target_cache.shape[0] - 1:
            self.target_cache = torch.cat(
                [self.target_cache, self.forward(x)], dim=0)

        features = torch.unsqueeze(self.target_cache[i], dim=0)

        return (self.reference-features).pow(2).sum(1)
