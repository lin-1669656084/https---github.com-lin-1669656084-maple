# maple.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaPLe(nn.Module):
    def __init__(
        self, input_dim=28 * 28, num_classes=10, prompt_dim=100, mask_ratio=0.5
    ):
        super(MaPLe, self).__init__()
        self.prompt_dim = prompt_dim
        self.mask_ratio = mask_ratio
        self.embedding = nn.Linear(input_dim, prompt_dim)
        self.prompt_mask = nn.Parameter(torch.ones(prompt_dim), requires_grad=True)
        self.classifier = nn.Linear(prompt_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)  # flatten
        x = self.embedding(x)  # project to prompt space

        # Apply learnable mask
        mask = torch.sigmoid(self.prompt_mask)  # values between 0 and 1
        masked_x = x * mask  # element-wise multiplication

        logits = self.classifier(masked_x)
        return logits
