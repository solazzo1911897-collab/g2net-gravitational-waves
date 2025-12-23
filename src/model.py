import warnings
warnings.filterwarnings("ignore") # Silence nnAudio import warnings

import torch
import torch.nn as nn
import timm
from nnAudio.Spectrogram import CQT1992v2
from .config import CFG

class G2NetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(G2NetModel, self).__init__()
        # Ensure n_bins warning is suppressed by passing correct arguments
        # If fmax is provided, n_bins is ignored, so we rely on fmax/fmin
        self.cqt = CQT1992v2(
            sr=CFG.sr, fmin=CFG.fmin, fmax=CFG.fmax,
            hop_length=CFG.hop_length,
            output_format="Magnitude", verbose=False
        )
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=pretrained, in_chans=3,
            num_classes=1, drop_rate=0.3, drop_path_rate=0.2
        )

    def forward(self, x):
        bs, ch, time_dim = x.shape
        x = x.view(bs * ch, time_dim)
        x = self.cqt(x)
        x = torch.log1p(x)
        x = x.view(bs, ch, x.size(1), x.size(2))
        
        # Standardize
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-7)
        
        if x.shape[2] != CFG.img_size:
            x = torch.nn.functional.interpolate(x, size=(CFG.img_size, CFG.img_size), 
                                              mode='bilinear', align_corners=False)
        return self.backbone(x)