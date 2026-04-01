"""utils.py — Utilities for FL-MedClsBench."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from torch.backends import cudnn
from torch.optim import Optimizer
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                              f1_score, roc_auc_score)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Method → client/server method mapping
# ---------------------------------------------------------------------------

def set_server_method(args):
    mapping = {
        'LocalTrain': ('local_train', 'singleset'),
        'FedAvg':     ('local_train', 'fedavg'),
        'FedProx':    ('fedprox',     'fedavg'),
        'MOON':       ('moon',        'fedavg'),
        'FedAWA':     ('local_train', 'fedawa'),
        'FedNova':    ('local_train', 'fednova'),
        'PN':         ('local_train', 'fedavg'),
        'FedRDN':     ('fedrdn',      'fedavg'),
        'FedLWS':     ('local_train', 'fedlws'),
        'FedBN':      ('local_train', 'fedbn'),
        'SioBN':      ('local_train', 'siobn'),
        'FedPer':     ('local_train', 'fedper'),
        'FedRoD':     ('fedrod',      'fedavg'),
        'Ditto':      ('ditto',       'fedavg'),
    }
    if args.method not in mapping:
        raise ValueError(f'Unknown method: {args.method}')
    args.client_method, args.server_method = mapping[args.method]
    return args


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_model(model_type: str, args):
    from models_dict.med3dcnn import Med3DCNN
    from models_dict.resnet3d import ResNet3D10, ResNet3D18
    from models_dict.resnet2d import ResNet18_2D, ResNet50_2D, ResNet50_Pretrained

    norm = 'pn' if args.method == 'PN' else 'bn'
    num_classes = args.num_classes
    pretrained  = getattr(args, 'pretrained', False)

    if model_type == 'Med3DCNN':
        return Med3DCNN(num_classes=num_classes, norm=norm)
    elif model_type == 'ResNet3D10':
        return ResNet3D10(num_classes=num_classes, norm=norm)
    elif model_type == 'ResNet3D18':
        return ResNet3D18(num_classes=num_classes, norm=norm)
    elif model_type == 'ResNet18':
        return ResNet18_2D(num_classes=num_classes, norm=norm)
    elif model_type == 'ResNet50':
        # PN must train from scratch: pretrained BN→PN replacement causes conv
        # weight collapse via FedAvg sign-cancellation on heterogeneous clients.
        if pretrained and args.method != 'PN':
            return ResNet50_Pretrained(num_classes=num_classes, norm=norm)
        return ResNet50_2D(num_classes=num_classes, norm=norm)
    else:
        raise ValueError(f'Unknown model type: {model_type}')


# ---------------------------------------------------------------------------
# Optimizer initialisation
# ---------------------------------------------------------------------------

def init_optimizer(node_id, model, args):
    if args.client_method == 'scaffold':
        return ScaffoldOptimizer(model.parameters(),
                                 lr=args.lr, weight_decay=args.local_wd_rate)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.local_wd_rate)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.local_wd_rate)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def model_parameter_vector(args, model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)


def set_params(model, model_state_dict, exclude_keys=set()):
    with torch.no_grad():
        for key in model_state_dict:
            if key not in exclude_keys:
                model.state_dict()[key].copy_(model_state_dict[key])
    return model


def freeze_layers(model, layers_to_freeze):
    for name, p in model.named_parameters():
        try:
            p.requires_grad = (name not in layers_to_freeze)
        except Exception:
            pass
    return model


# ---------------------------------------------------------------------------
# Dataset statistics (for FedRDN)
# ---------------------------------------------------------------------------

def compute_dataset_statistics(dataloader):
    """Compute per-channel mean and std over the full training loader.

    Supports both 2D slices (B,C,H,W) and 3D volumes (B,C,D,H,W).
    """
    n_pixels = torch.zeros(1).cuda()
    ch_sum   = None
    ch_sq    = None

    for images, _ in dataloader:
        images = images.float().cuda()
        if ch_sum is None:
            c = images.size(1)
            ch_sum = torch.zeros(c).cuda()
            ch_sq  = torch.zeros(c).cuda()
        if images.dim() == 4:   # 2D: B,C,H,W
            n_pixels += images.size(0) * images.size(2) * images.size(3)
            ch_sum   += images.sum(dim=[0, 2, 3])
            ch_sq    += (images ** 2).sum(dim=[0, 2, 3])
        else:                   # 3D: B,C,D,H,W
            n_pixels += (images.size(0) * images.size(2)
                         * images.size(3) * images.size(4))
            ch_sum   += images.sum(dim=[0, 2, 3, 4])
            ch_sq    += (images ** 2).sum(dim=[0, 2, 3, 4])

    mean = ch_sum / n_pixels
    std  = torch.sqrt(ch_sq / n_pixels - mean ** 2 + 1e-8)
    return mean, std


# ---------------------------------------------------------------------------
# FedRDN normalization transform
# ---------------------------------------------------------------------------

class FedRDNTransform:
    def __init__(self, local_stats, global_stats, mode='train', p=1.0):
        self.local_mean   = local_stats[0]
        self.local_std    = local_stats[1]
        self.global_means = [s[0] for s in global_stats]
        self.global_stds  = [s[1] for s in global_stats]
        self.mode = mode
        self.p    = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        if self.mode == 'train':
            idx  = random.randint(0, len(self.global_means) - 1)
            mean = self.global_means[idx]
            std  = self.global_stds[idx]
        else:
            mean = self.local_mean
            std  = self.local_std
        # Supports 2D (B,C,H,W) and 3D (B,C,D,H,W)
        if x.dim() == 4:
            return (x - mean.view(1, -1, 1, 1)) / (std.view(1, -1, 1, 1) + 1e-8)
        else:
            return (x - mean.view(1, -1, 1, 1, 1)) / (std.view(1, -1, 1, 1, 1) + 1e-8)


# ---------------------------------------------------------------------------
# Validation (classification)
# ---------------------------------------------------------------------------

def _forward_one(args, node, images: torch.Tensor) -> torch.Tensor:
    """Run forward pass and return softmax probabilities (B, C)."""
    model = node.model
    if args.method == 'FedRoD':
        logit, feat = model(images)
        logit = logit + node.p_head(feat)
    elif args.method == 'Ditto':
        logit, _ = node.p_model(images)
    elif args.method == 'FedRDN':
        images = node.FedRDNTransform_test(images)
        logit, _ = model(images)
    else:
        logit, _ = model(images)
    return logit


def validate(args, node, loader):
    """Patient-level evaluation.

    `loader` yields batches of (slices_tensor, label) where slices_tensor
    is (N_slices, 3, H, W) for a single patient.

    For each patient we average the per-slice softmax probabilities and
    derive a patient-level hard prediction.

    Returns: loss, acc*100, recall*100, prec*100, f1*100, auc*100
    """
    model = node.model
    model.eval()

    patient_probs   = []   # (num_patients, num_classes)
    patient_labels  = []
    loss_total      = 0.0

    with torch.no_grad():
        for batch in loader:
            # batch is a list of length 1 (batch_size=1 in patient loader)
            slices_tensor, label = batch[0]   # (N, 3, H, W), scalar/tensor
            label = int(label)

            # Forward all slices (may be large; process in mini-chunks)
            slices_tensor = slices_tensor.cuda()
            chunk_size    = 16
            slice_logits  = []
            for i in range(0, slices_tensor.size(0), chunk_size):
                chunk = slices_tensor[i:i + chunk_size]
                slice_logits.append(_forward_one(args, node, chunk))
            logits = torch.cat(slice_logits, dim=0)          # (N, C)

            # Patient-level average probability
            prob = logits.softmax(dim=1).mean(dim=0)          # (C,)
            patient_probs.append(prob)

            label_t = torch.tensor(label).cuda()
            loss_total += F.cross_entropy(prob.unsqueeze(0),
                                           label_t.unsqueeze(0)).item()
            patient_labels.append(label)

    probs_np  = torch.stack(patient_probs).cpu().numpy()      # (P, C)
    labels_np = np.array(patient_labels)
    preds_np  = probs_np.argmax(axis=1)

    acc    = accuracy_score(labels_np, preds_np)
    recall = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    prec   = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    f1     = f1_score(labels_np, preds_np, average='macro', zero_division=0)

    try:
        if args.num_classes == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            # Some sites may not have all classes in val/test — filter to present ones
            present = np.sort(np.unique(labels_np))
            if len(present) < 2:
                auc = 0.0
            else:
                probs_filtered = probs_np[:, present]
                probs_filtered = probs_filtered / probs_filtered.sum(axis=1, keepdims=True)
                auc = roc_auc_score(labels_np, probs_filtered,
                                    labels=present,
                                    multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    loss = loss_total / max(len(patient_labels), 1)
    return loss, acc * 100, recall * 100, prec * 100, f1 * 100, auc * 100


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def lr_scheduler(rounds, client_nodes, args):
    if (rounds + 1) % args.stepsize == 0:
        args.lr /= 2.0
        for node in client_nodes.values():
            node.args.lr = args.lr
            node.optimizer.param_groups[0]['lr'] = args.lr
    print(f'Learning rate={args.lr:.6f}')


def cosine_lr(base_lr: float, rnd: int, T: int) -> float:
    """Cosine annealing: lr decays from base_lr to 0 over T rounds."""
    import math
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * rnd / T))


# ---------------------------------------------------------------------------
# Scaffold optimizer
# ---------------------------------------------------------------------------

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):
        for group in self.param_groups:
            for p, c, ci in zip(group['params'],
                                server_controls.values(),
                                client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp * group['lr']
