import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from nnAudio.Spectrogram import CQT1992v2
from .config import CFG

class G2NetModel(nn.Module):
    """
    Deep Learning Model for Gravitational Wave Detection.
    
    Architecture:
    1. Preprocessing: Constant Q-Transform (CQT) layer (1D Wave -> 2D Image).
    2. Backbone: EfficientNet-B2 (Pretrained on ImageNet).
    3. Head: Binary Classification (1 output unit).
    """
    
    def __init__(self, cfg=CFG, pretrained=True):
        super(G2NetModel, self).__init__()
        self.cfg = cfg
        
        # --- 1. Spectrogram Layer (On-the-fly generation) ---
        # Converts time-series data to frequency domain images using GPU
        self.cqt = CQT1992v2(
            sr=cfg.sr,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            hop_length=cfg.hop_length,
            bins_per_octave=cfg.bins_per_octave,
            output_format="Magnitude",
            verbose=False
        )
        
        # --- 2. CNN Backbone (EfficientNet) ---
        # Using 'tf_efficientnet_b2_ns' as per V6 experiments
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=pretrained,
            in_chans=3,       # Input: 3 detectors (LIGO H, LIGO L, Virgo)
            num_classes=1,    # Output: Probability of CW signal
            drop_rate=0.3,
            drop_path_rate=0.2
        )

    def forward(self, x):
        """
        Forward pass of the network.
        Input shape: (batch_size, 3, 4096)
        Output shape: (batch_size, 1)
        """
        bs, ch, time_dim = x.shape
        
        # Reshape to process all channels through CQT layer
        x = x.view(bs * ch, time_dim)
        
        # 1. CQT Transform
        x = self.cqt(x)                 # Shape: (batch*3, 1, freq, time)
        
        # 2. Log Scaling (Amplitude normalization)
        x = torch.log1p(x)
        
        # Reshape back to separate channels: (batch, 3, freq, time)
        x = x.view(bs, ch, x.size(1), x.size(2))
        
        # 3. Standardization (Instance Normalization)
        # Normalize per image to handle varying noise floors
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-7)
        
        # 4. Resize to match EfficientNet input requirements
        if x.shape[2] != self.cfg.img_size:
            x = F.interpolate(
                x, 
                size=(self.cfg.img_size, self.cfg.img_size), 
                mode='bilinear', 
                align_corners=False
            )
            
        # 5. CNN Backbone Inference
        output = self.backbone(x)
        
        return output