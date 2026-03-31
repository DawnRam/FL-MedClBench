import torch
import torch.nn as nn


from models_dict.resnet2d import PopulationNormalization


def _norm_layer(norm_type, num_channels):
    if norm_type == 'bn':
        return nn.BatchNorm3d(num_channels)
    elif norm_type == 'pn':
        return PopulationNormalization(num_channels)
    else:  # 'in'
        return nn.InstanceNorm3d(num_channels, affine=True)


def _conv_block(in_ch, out_ch, norm_type='bn'):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        _norm_layer(norm_type, out_ch),
        nn.ReLU(inplace=True),
    )


class Med3DCNN(nn.Module):
    """Lightweight 3D CNN for medical image classification.

    Returns (logit, feature) for compatibility with MOON / FedROD / Ditto.
    Output layer key: 'fc'  (used by FedPer / FedROD aggregation).
    BN layers named 'encN.1.*' / 'enc5.1.*' → contain 'bn'? No — use custom
    block naming below so 'bn' appears in key names for FedBN compat.
    """

    def __init__(self, num_classes: int = 2, norm: str = 'bn'):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            _norm_layer(norm, 32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            _norm_layer(norm, 64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1, bias=False),
            _norm_layer(norm, 128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            _norm_layer(norm, 256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.enc4(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        out = self.fc(feat)
        return out, feat
