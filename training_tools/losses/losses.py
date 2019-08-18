import torch
import torch.nn.functional as F
import torch.nn as nn

# https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18

BCELoss = nn.BCELoss()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class HingeBasedLoss(nn.Module):
    """
    Hinge-based loss term and squared l2-norm regularization.
    Based on: https://arxiv.org/pdf/1504.03641.pdf
    """

    def __init__(self):
        super(HingeBasedLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = 1 - torch.mul(torch.squeeze(output), target)
        hinge_loss = F.relu(hinge_loss)

        return torch.sum(hinge_loss)
