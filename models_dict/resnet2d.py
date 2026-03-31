"""resnet2d.py — 2D ResNet for FL-MedClsBench classification.

Input:  (B, 3, H, W)
Output: (logit, feature)  — compatible with MOON / FedROD / Ditto

Output layer key: 'fc'   → used by FedPer / FedROD
BN layer keys:   'bn1', 'layer*.*.bn1', 'layer*.*.bn2'  → contain 'bn'

ResNet50Pretrained: torchvision ImageNet weights, same state_dict key layout,
                    drop-in replacement for ResNet50_2D.

norm='pn'  → PopulationNormalization (Wang et al., CVPR 2025):
             pop_mean and pop_gamma are learnable parameters (gradient-updated).
             Identical to the FL-MedSegBench implementation.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# ---------------------------------------------------------------------------
# Population Normalization — identical to FL-MedSegBench implementation
# ---------------------------------------------------------------------------

class PopulationNormalization(nn.Module):
    """Population Normalization (PN) for federated learning.

    pop_mean and pop_gamma are learnable parameters trained via gradient
    descent, making them fully compatible with FedAvg aggregation.
    Identical implementation to FL-MedSegBench/models_dict/unet2d.py.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(PopulationNormalization, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

        self.pop_mean  = Parameter(torch.zeros(num_features))
        self.pop_gamma = Parameter(torch.zeros(num_features))  # log-variance; var = exp(pop_gamma)

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias   = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.zeros_(self.pop_mean)
        nn.init.zeros_(self.pop_gamma)  # exp(0) = 1.0 → safe default variance

    def forward(self, x):
        if self.training:
            dims = [0, 2, 3] if len(x.shape) == 4 else [0, 2, 3, 4]
            batch_mean = x.mean(dim=dims)
            batch_var  = x.var(dim=dims, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * batch_var.detach()

        mean = self.pop_mean
        # pop_gamma is stored in log-space so var is always positive.
        # Direct storage caused instability: deep ResNet50 layers have
        # running_var ≈ 0.001 → gradient ∂L/∂var ∝ 1/(var+ε)^1.5 ≈ 31623×
        # the gradient at var=1 → Adam pushes var negative in one step → NaN.
        # Log-space: ∂L/∂log_var = ∂L/∂var × var  (auto-scaled, always finite).
        var  = self.pop_gamma.exp()

        if len(x.shape) == 4:
            x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        else:
            x_norm = (x - mean[None, :, None, None, None]) / torch.sqrt(var[None, :, None, None, None] + self.eps)

        if self.affine:
            if len(x.shape) == 4:
                x_norm = x_norm * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            else:
                x_norm = x_norm * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return x_norm

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


def _replace_bn_with_pn(module: nn.Module) -> nn.Module:
    """Recursively replace BatchNorm2d with PopulationNormalization.

    Initialises pop_mean/pop_gamma from BN running stats and weight/bias
    from BN affine params to preserve pretrained feature distribution.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            pn = PopulationNormalization(child.num_features, eps=child.eps)
            with torch.no_grad():
                pn.pop_mean .copy_(child.running_mean)
                # pop_gamma is now log-variance: init to log(running_var) so
                # exp(pop_gamma) = running_var → initial PN ≡ BN (preserves
                # pretrained feature scale). log-space ensures gradient is
                # ∂L/∂var × var (auto-scaled), preventing blow-up for tiny var.
                pn.pop_gamma.copy_(child.running_var.clamp(min=1e-8).log())
                if child.weight is not None:
                    pn.weight.copy_(child.weight)
                if child.bias is not None:
                    pn.bias  .copy_(child.bias)
            setattr(module, name, pn)
        else:
            _replace_bn_with_pn(child)
    return module


def _norm2d(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == 'bn':
        return nn.BatchNorm2d(num_channels)
    elif norm_type == 'pn':
        return PopulationNormalization(num_channels)
    else:  # 'in'
        return nn.InstanceNorm2d(num_channels, affine=True)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm='bn'):
        super().__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, 3, stride=stride,
                                    padding=1, bias=False)
        self.bn1        = _norm2d(norm, planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2        = _norm2d(norm, planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm='bn'):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1   = _norm2d(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                                padding=1, bias=False)
        self.bn2   = _norm2d(norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = _norm2d(norm, planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet2D(nn.Module):
    """2D ResNet for slice-level medical image classification.

    Input:  (B, 3, H, W)
    Output: (logit [B, num_classes], feature [B, feat_dim])
    """

    def __init__(self, block, layers, num_classes=2, norm='bn'):
        super().__init__()
        self.inplanes = 64

        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        self.bn1     = _norm2d(norm, 64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1  = self._make_layer(block, 64,  layers[0], norm=norm)
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2,
                                         norm=norm)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2,
                                         norm=norm)
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2,
                                         norm=norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, norm='bn'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          1, stride=stride, bias=False),
                _norm2d(norm, planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm=norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=norm))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x    = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x)
        x    = self.layer4(x)
        x    = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        out  = self.fc(feat)
        return out, feat


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def ResNet18_2D(num_classes=2, norm='bn'):
    return ResNet2D(BasicBlock2D, [2, 2, 2, 2],
                   num_classes=num_classes, norm=norm)


def ResNet50_2D(num_classes=2, norm='bn'):
    return ResNet2D(Bottleneck2D, [3, 4, 6, 3],
                   num_classes=num_classes, norm=norm)


# ---------------------------------------------------------------------------
# Pretrained ResNet50 — ImageNet weights via torchvision
# State-dict keys are identical to torchvision ResNet50, so FedBN / SioBN
# ('bn' in key) and FedPer / FedRoD ('fc' in key) work without changes.
# norm='pn': BN layers are replaced with PopulationNorm2d after loading
#            weights, initialised from BN running stats.
# ---------------------------------------------------------------------------

class _ResNet50Pretrained(nn.Module):
    """torchvision ResNet50 with ImageNet weights, returning (logit, feature).

    norm='bn' (default): standard BN from torchvision weights.
    norm='pn': BN replaced with PopulationNorm2d (Wang et al., CVPR 2025),
               μ/σ² initialised from BN running_mean/running_var.
    """

    def __init__(self, num_classes: int, norm: str = 'bn'):
        super().__init__()
        import torchvision.models as tv
        backbone = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1)
        # Replace classification head
        feat_dim    = backbone.fc.in_features   # 2048
        backbone.fc = nn.Linear(feat_dim, num_classes)
        nn.init.normal_(backbone.fc.weight, std=0.01)
        nn.init.zeros_(backbone.fc.bias)

        # Expose sub-modules with original names so state_dict keys are
        # identical to torchvision (conv1, bn1, layer1-4, fc …)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc      = backbone.fc

        if norm == 'pn':
            _replace_bn_with_pn(self)

    def forward(self, x):
        x    = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x)
        x    = self.layer4(x)
        x    = self.avgpool(x)
        feat = x.view(x.size(0), -1)
        out  = self.fc(feat)
        return out, feat


def ResNet50_Pretrained(num_classes=2, norm='bn'):
    return _ResNet50Pretrained(num_classes=num_classes, norm=norm)
