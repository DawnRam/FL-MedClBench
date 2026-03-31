"""client_funct.py — Client-side training for FL-MedClsBench (classification).

Bug-fix log vs. original:
  - FedProx  : removed erroneous `if idx > 0` guard on proximal term.
               Paper (Li et al. 2020) applies the proximal penalty at every step.
  - Ditto    : swapped update order — global model updated first (standard local
               train), then personalized model updated with proximal objective.
               Paper (Li et al. 2021 ICML) specifies this order explicitly.
  - FedRoD   : added Balanced Softmax Loss (BSM) for the generic predictor.
               Paper (Chen & Chao 2022 ICLR) is built on BSM as its core
               contribution; the original code only used plain CE on the sum of
               both heads, completely omitting BSM.
  - MOON     : temperature τ is now taken from args.temperature (default 0.5)
               instead of being hardcoded, enabling hyperparameter search.
"""

import copy
from collections import Counter

import torch
import torch.nn.functional as F

from utils import model_parameter_vector


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def Client_update(args, client_nodes, central_node, select_list):
    """Run local training for all selected clients.

    Returns:
        client_nodes, client_losses (dict), client_accs (dict)
    """
    client_losses = {}
    client_accs   = {}

    for i in select_list:
        epoch_losses, epoch_accs = [], []

        if args.client_method == 'local_train' or args.client_method == 'fedrdn':
            for _ in range(args.E):
                l, a = _run_local(args, client_nodes[i])
                epoch_losses.append(l); epoch_accs.append(a)

        elif args.client_method == 'fedprox':
            for _ in range(args.E):
                l, a = _run_fedprox(args, client_nodes[i], central_node)
                epoch_losses.append(l); epoch_accs.append(a)

        elif args.client_method == 'fedrod':
            for _ in range(args.E):
                l, a = _run_fedrod(args, client_nodes[i])
                epoch_losses.append(l); epoch_accs.append(a)

        elif args.client_method == 'feddyn':
            global_vec = model_parameter_vector(
                args, central_node.model).detach().clone()
            for _ in range(args.E):
                l, a = _run_feddyn(global_vec, args, client_nodes[i])
                epoch_losses.append(l); epoch_accs.append(a)
            v1 = model_parameter_vector(args, client_nodes[i].model).detach()
            client_nodes[i].old_grad -= args.mu * (v1 - global_vec)

        elif args.client_method == 'scaffold':
            x_before = copy.deepcopy(client_nodes[i].model)
            for _ in range(args.E):
                l, a = _run_scaffold(args, client_nodes[i], central_node)
                epoch_losses.append(l); epoch_accs.append(a)
            ann = copy.deepcopy(client_nodes[i].model)
            local_steps = args.E * len(client_nodes[i].train_loader)
            for k, v in x_before.named_parameters():
                cur = dict(ann.named_parameters())[k]
                client_nodes[i].control[k] = (
                    client_nodes[i].control[k]
                    - central_node.control[k]
                    + (v.data - cur.data) / (local_steps * args.lr)
                )
                client_nodes[i].delta_y[k]       = cur.data - v.data
                client_nodes[i].delta_control[k] = (
                    client_nodes[i].control[k]
                    - client_nodes[i].control[k])  # zero by design

        elif args.client_method == 'ditto':
            # FIX (Li et al. 2021): global model updated FIRST, then personalized.
            # Original code had personalized first, which used a stale global anchor.
            for _ in range(args.E):
                _run_local(args, client_nodes[i])          # step 1: global update
            for _ in range(args.E):
                l, a = _run_ditto(args, client_nodes[i], central_node)
                epoch_losses.append(l); epoch_accs.append(a)

        elif args.client_method == 'moon':
            for _ in range(args.E):
                l, a = _run_moon(args, client_nodes[i], central_node)
                epoch_losses.append(l); epoch_accs.append(a)
            client_nodes[i].pre_model.load_state_dict(
                copy.deepcopy(client_nodes[i].model.state_dict()))

        else:
            raise ValueError(f'Unknown client method: {args.client_method}')

        client_losses[i] = sum(epoch_losses) / max(len(epoch_losses), 1)
        client_accs[i]   = sum(epoch_accs)   / max(len(epoch_accs), 1)

    return client_nodes, client_losses, client_accs


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ce():
    return torch.nn.CrossEntropyLoss()


def _balanced_softmax_loss(logit: torch.Tensor,
                           labels: torch.Tensor,
                           sample_per_class: list) -> torch.Tensor:
    """Balanced Softmax Loss (BSM) — FedRoD (Chen & Chao, ICLR 2022).

    Adds log(n_c) offset to logits before cross-entropy to correct for
    non-IID class-frequency imbalance across clients.

    Args:
        logit            : (B, C) raw logits from the generic head.
        labels           : (B,) ground-truth class indices.
        sample_per_class : list of length C with per-class training counts.
    """
    n_c = torch.tensor(sample_per_class, dtype=torch.float32,
                       device=logit.device).clamp(min=1.0)
    logit_adj = logit + n_c.log().unsqueeze(0)     # (B, C)
    return F.cross_entropy(logit_adj, labels)


def _get_class_counts(node) -> list:
    """Return per-class sample counts from node.train_loader.dataset.labels."""
    labels = node.train_loader.dataset.labels
    counts = Counter(labels)
    return [counts.get(c, 1) for c in range(node.num_classes)]


# ─────────────────────────────────────────────────────────────────────────────
# Individual training functions
# ─────────────────────────────────────────────────────────────────────────────

def _run_local(args, node):
    node.model.train()
    criterion = _ce()
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()

        if args.client_method == 'fedrdn':
            images = node.FedRDNTransform_train(images)

        node.optimizer.zero_grad()
        logit, _ = node.model(images)
        loss = criterion(logit, labels)
        loss.backward()
        node.optimizer.step()

        loss_sum += loss.item()
        correct  += (logit.argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_fedprox(args, node, central_node):
    """FedProx — Li et al., MLSys 2020.

    Fix: proximal term applied at EVERY mini-batch (removed `if idx > 0`).
    Paper objective: F_k(w) + (mu/2)||w - w_t||^2, uniform across all steps.
    """
    node.model.train()
    criterion = _ce()
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        node.optimizer.zero_grad()
        logit, _ = node.model(images)
        loss = criterion(logit, labels)

        # Proximal term — applied at every step (FIX: was `if idx > 0`)
        w_diff = torch.tensor(0.0, device='cuda')
        for w, wt in zip(central_node.model.parameters(),
                         node.model.parameters()):
            w_diff += torch.pow(torch.norm(w - wt), 2)
        loss += (args.mu / 2.0) * w_diff

        loss.backward()
        node.optimizer.step()

        loss_sum += loss.item()
        correct  += (logit.argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_fedrod(args, node):
    """FedRoD — Chen & Chao, ICLR 2022.

    Fix: added Balanced Softmax Loss (BSM) on the generic head.
    The original code only computed CE on the combined output (logit + logit_p),
    which completely omits BSM — the core contribution of FedRoD.

    Training objective (Algorithm 1 of the paper):
      L = BSM(h_g(x), y)                   ← generic head with class-balance
        + CE(h_g(x) + h_p(x), y)           ← combined personalized predictor
    Both losses share the backbone gradient; p_head gradient is from CE only.
    """
    node.model.train()
    criterion_ce = _ce()
    sample_per_class = _get_class_counts(node)
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        node.optimizer.zero_grad()

        logit_g, feat = node.model(images)           # generic head
        logit_p       = node.p_head(feat)            # personalized head

        loss_bsm = _balanced_softmax_loss(logit_g, labels, sample_per_class)
        loss_ce  = criterion_ce(logit_g + logit_p, labels)
        loss     = loss_bsm + loss_ce

        loss.backward()
        node.optimizer.step()

        loss_sum += loss.item()
        correct  += ((logit_g + logit_p).argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_feddyn(global_vec, args, node):
    node.model.train()
    criterion = _ce()
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        node.optimizer.zero_grad()
        logit, _ = node.model(images)
        loss = criterion(logit, labels)

        v1     = model_parameter_vector(args, node.model)
        loss  += (args.mu / 2.0) * torch.norm(v1 - global_vec, 2) ** 2
        loss  -= torch.dot(node.old_grad, v1)

        loss.backward()
        node.optimizer.step()

        loss_sum += loss.item()
        correct  += (logit.argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_scaffold(args, node, central_node):
    node.model.train()
    criterion = _ce()
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        logit, _ = node.model(images)
        loss = criterion(logit, labels)
        loss.backward()
        node.optimizer.step(central_node.control, node.control)
        node.optimizer.zero_grad()

        loss_sum += loss.item()
        correct  += (logit.argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_ditto(args, node, central_node):
    """Ditto personalized model update — Li et al., ICML 2021.

    Called AFTER the global model has been updated via _run_local.
    The proximal anchor is `central_node.model` (the freshly aggregated global
    model received from the server), not the locally updated global model.

    Objective: min_v  F_k(v)  +  (lambda/2) * ||v - w_global||^2
    """
    node.p_model.train()
    criterion = _ce()
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        node.p_optimizer.zero_grad()
        logit, _ = node.p_model(images)
        loss = criterion(logit, labels)

        # Proximal term toward the global model (w_global from server)
        w_diff = torch.tensor(0.0, device='cuda')
        for w, wt in zip(central_node.model.parameters(),
                         node.p_model.parameters()):
            w_diff += torch.pow(torch.norm(w - wt), 2)
        loss += (args.mu / 2.0) * w_diff

        loss.backward()
        node.p_optimizer.step()

        loss_sum += loss.item()
        correct  += (logit.argmax(1) == labels).sum().item()
        n        += images.size(0)

    return loss_sum / max(n, 1), correct / max(n, 1) * 100


def _run_moon(args, node, central_node):
    """MOON — Li et al., CVPR 2021.

    Fix: temperature τ read from args.temperature (default 0.5) instead of
    hardcoded 0.5, enabling proper hyperparameter search.

    Contrastive loss:
      l_con = -log( exp(sim(z, z_g)/τ) / (exp(sim(z, z_g)/τ) + exp(sim(z, z_prev)/τ)) )
    Total loss:
      L = CE(output, y) + μ * l_con
    """
    node.model.train()
    node.pre_model.eval()
    central_node.model.eval()
    cos = torch.nn.CosineSimilarity(dim=-1)
    criterion = _ce()
    tau = getattr(args, 'temperature', 0.5)
    loss_sum = correct = n = 0

    for images, labels in node.train_loader:
        images, labels = images.cuda(), labels.cuda()
        node.optimizer.zero_grad()
        output, fea_cur = node.model(images)

        with torch.no_grad():
            _, fea_global = central_node.model(images)
            _, fea_pre    = node.pre_model(images)

        fea_cur    = fea_cur.view(images.size(0), -1)
        fea_global = fea_global.view(images.size(0), -1)
        fea_pre    = fea_pre.view(images.size(0), -1)

        pos_sim = cos(fea_cur, fea_global).unsqueeze(-1) / tau
        neg_sim = cos(fea_cur, fea_pre).unsqueeze(-1)    / tau
        con_loss = -torch.log(
            torch.exp(pos_sim) /
            (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)
        ).mean()

        loss = criterion(output, labels) + args.mu * con_loss
        loss.backward()
        node.optimizer.step()

        loss_sum += loss.item()
        correct  += (output.argmax(1) == labels).sum().item()
        n        += images.size(0)

    central_node.model.train()
    return loss_sum / max(n, 1), correct / max(n, 1) * 100
