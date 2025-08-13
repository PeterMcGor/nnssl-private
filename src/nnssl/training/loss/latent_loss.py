import torch
from torch import nn
import torch.nn.functional as F


class BottleNeckContrastiveLoss(nn.Module):
    def __init__(self, feat_weight: float = 0.1):
        super().__init__()
        self.feat_weight = feat_weight

    def forward(self, batch, output, mask, latent):
        print(latent[0].shape)
        return 0