import torch
import torch.nn as nn
from torchvision import models
import numpy as np

from basicsr.archs.ddcolor_arch_utils.unet import _conv
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DynamicUNetDiscriminator(nn.Module):

    def __init__(self, n_channels: int = 3, nf: int = 256, n_blocks: int = 3):
        super().__init__()
        layers = [_conv(n_channels, nf, ks=4, stride=2)]
        for i in range(n_blocks):
            layers += [
                _conv(nf, nf, ks=3, stride=1),
                _conv(nf, nf * 2, ks=4, stride=2, self_attention=(i == 0)),
            ]
            nf *= 2
        layers += [_conv(nf, nf, ks=3, stride=1), _conv(nf, 1, ks=4, bias=False, padding=0, use_activ=False)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        return out
