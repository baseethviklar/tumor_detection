# models/vit_model.py

import torch
import torch.nn as nn
from transformers import ViTModel

class ViTForBrainTumorDetection(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTForBrainTumorDetection, self).__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # Add classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit.config.hidden_size),
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ViT expects input tensor of shape (batch_size, num_channels, height, width)
        outputs = self.vit(x)
        # Use the [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0]
        # Apply classification head
        logits = self.classifier(cls_token)
        return type('Outputs', (), {'logits': logits})
