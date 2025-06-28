import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169

class TBlock(nn.Module):
    def __init__(self, in_channels):
        super(TBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.fc2 = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))

        gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        attn = torch.sigmoid(self.fc2(F.relu(self.fc1(gap)))).view(x.size(0), -1, 1, 1)
        x3 = x * attn

        out = F.relu(x1 + x2 + x3)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        x = x + attn_output

        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x

class NovelHybridTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(NovelHybridTransformer, self).__init__()
        self.backbone = densenet169(pretrained=True).features
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.t_block = TBlock(in_channels=1664)
        self.reduce_conv = nn.Conv2d(1664, 64, kernel_size=1)
        self.transformer1 = TransformerEncoderBlock(embed_dim=64, num_heads=4, ff_dim=128)
        self.transformer2 = TransformerEncoderBlock(embed_dim=64, num_heads=4, ff_dim=128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.t_block(x)
        x = self.reduce_conv(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, tokens, embed_dim)

        x = self.transformer1(x)
        x = self.transformer2(x)

        x = x.mean(dim=1)  # Global average over sequence
        out = self.classifier(x)
        return out


