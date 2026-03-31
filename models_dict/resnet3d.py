import torch
import torch.nn as nn


from models_dict.resnet2d import PopulationNormalization


def _norm3d(norm_type, num_channels):
    if norm_type == 'bn':
        return nn.BatchNorm3d(num_channels)
    elif norm_type == 'pn':
        return PopulationNormalization(num_channels)
    else:  # 'in'
        return nn.InstanceNorm3d(num_channels, affine=True)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm='bn'):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = _norm3d(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = _norm3d(norm, planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """3D ResNet for medical image classification.

    Returns (logit, feature) — feature is the global-average-pooled
    representation before the final FC layer.

    Key naming conventions (for FL algorithms):
      - BN layers: named 'bn1', 'layer*.*.bn1', 'layer*.*.bn2'  → contain 'bn'
      - Output layer: 'fc'  → used by FedPer / FedROD
    """

    def __init__(self, block, layers, num_classes=2, norm='bn'):
        super().__init__()
        self.inplanes = 64
        self.norm_type = norm

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _norm3d(norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], norm=norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm=norm)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, norm='bn'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                _norm3d(norm, planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm=norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=norm))
        return nn.Sequential(*layers)

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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        out = self.fc(feat)
        return out, feat


def ResNet3D10(num_classes=2, norm='bn'):
    return ResNet3D(BasicBlock3D, [1, 1, 1, 1], num_classes=num_classes, norm=norm)


def ResNet3D18(num_classes=2, norm='bn'):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, norm=norm)
