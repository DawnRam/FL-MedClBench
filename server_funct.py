"""server_funct.py — Server-side aggregation for FL-MedClsBench.

Adapted from FL-MedSegBench/server_funct.py.
Key changes vs segmentation version:
  - FedPer: output layer key = 'fc' (classification head)
  - FedBN / SioBN: match 'bn' layer name pattern (works for ResNet3D / Med3DCNN)
  - FedAWA / FedLWS: unchanged (model-agnostic)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# ---------------------------------------------------------------------------
# Helper: receive & normalize aggregation weights
# ---------------------------------------------------------------------------

def _get_agg_weights(size_agg_weights, select_list):
    w = [size_agg_weights[i] for i in select_list]
    total = sum(w)
    return [x / total for x in w]


def Server_update(args, central_node, client_nodes, select_list,
                  size_agg_weights, epoch=0):

    agg_weights = _get_agg_weights(size_agg_weights, select_list)

    # ------------------------------------------------------------------ FedAvg
    if args.server_method == 'fedavg':
        for key in central_node.model.state_dict():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, ci in enumerate(select_list):
                    temp += agg_weights[i] * client_nodes[ci].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(temp)

    # ----------------------------------------------------------------- FedNova
    elif args.server_method == 'fednova':
        # FIX (Wang et al. NeurIPS 2020): tau_i = E * |D_i| / B (total local
        # steps per round). Original code used only len(train_loader) = |D_i|/B,
        # missing the E factor and underestimating tau_eff by E when E > 1.
        client_step = [args.E * len(client_nodes[i].train_loader)
                       for i in select_list]
        tao_eff = sum(agg_weights[j] * client_step[j]
                      for j in range(len(select_list)))
        correct_term = sum(agg_weights[j] / client_step[j] * tao_eff
                           for j in range(len(select_list)))

        for key in central_node.model.state_dict():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                temp = (1.0 - correct_term) * central_node.model.state_dict()[key]
                for j, ci in enumerate(select_list):
                    temp += (agg_weights[j] / client_step[j] * tao_eff
                             * client_nodes[ci].model.state_dict()[key])
                central_node.model.state_dict()[key].data.copy_(temp)
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(temp)

    # ------------------------------------------------------------------ FedPer
    elif args.server_method == 'fedper':
        # Keep output layer ('fc') personalised — do not aggregate
        for key in central_node.model.state_dict():
            if 'fc' in key:
                continue
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key],
                                        dtype=torch.float32)
                for i, ci in enumerate(select_list):
                    temp += agg_weights[i] * client_nodes[ci].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(temp)

    # ------------------------------------------------------------------- FedDYN
    elif args.server_method == 'feddyn':
        uploaded = [copy.deepcopy(client_nodes[i].model) for i in select_list]
        delta = copy.deepcopy(uploaded[0])
        for p in delta.parameters():
            p.data = torch.zeros_like(p.data)
        for j, cm in enumerate(uploaded):
            for sp, cp, dp in zip(central_node.model.parameters(),
                                  cm.parameters(), delta.parameters()):
                dp.data += (cp - sp) * agg_weights[j]
        for stp, dp in zip(central_node.server_state.parameters(),
                           delta.parameters()):
            stp.data -= args.mu * dp

        central_node.model = copy.deepcopy(uploaded[0])
        for p in central_node.model.parameters():
            p.data = torch.zeros_like(p.data)
        for j, cm in enumerate(uploaded):
            for sp, cp in zip(central_node.model.parameters(), cm.parameters()):
                sp.data += cp.data * agg_weights[j]
        for sp, stp in zip(central_node.model.parameters(),
                           central_node.server_state.parameters()):
            sp.data -= (1.0 / args.mu) * stp

        for key in central_node.model.state_dict():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(
                        central_node.model.state_dict()[key])

    # ------------------------------------------------------------------- FedAWA
    elif args.server_method == 'fedawa':
        global global_T_weight
        if epoch == 0:
            global_T_weight = list(agg_weights)

        def get_flat_weights(node):
            parts = []
            for key in node.model.state_dict():
                if all(s not in key for s in
                       ('num_batches_tracked', 'running_mean', 'running_var')):
                    parts.append(node.model.state_dict()[key].clone().detach().reshape(-1))
            return torch.cat(parts)

        flat_clients = torch.stack([get_flat_weights(client_nodes[i])
                                    for i in select_list])
        flat_global  = get_flat_weights(central_node)

        T_w = torch.tensor(global_T_weight, dtype=torch.float32).cuda()
        T_w = Variable(T_w, requires_grad=True)
        if args.server_optimizer == 'adam':
            att_opt = optim.Adam([T_w], lr=0.001, betas=(0.5, 0.999))
        else:
            att_opt = optim.SGD([T_w], lr=0.01, momentum=0.9, weight_decay=5e-4)

        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)

        for _ in range(args.server_epochs):
            prob = torch.softmax(T_w, dim=0)
            x_col   = flat_global.detach().unsqueeze(0).unsqueeze(-2)
            y_lin   = flat_clients.detach().unsqueeze(-3)
            if args.reg_distance == 'cos':
                C = 1 - d_cosine(x_col, y_lin)
            else:
                C = torch.mean((torch.abs(x_col - y_lin)) ** 2, -1)
            reg_loss = torch.sum(prob * C, dim=(-2, -1))

            client_grad  = flat_clients - flat_global.detach()
            col_sum      = torch.matmul(prob.unsqueeze(0), client_grad)
            l2_dist      = torch.norm(client_grad.unsqueeze(0)
                                      - col_sum.unsqueeze(1), p=2, dim=2)
            sim_loss = torch.sum(prob * l2_dist, dim=(-2, -1))

            loss_att = sim_loss + reg_loss
            att_opt.zero_grad()
            loss_att.backward()
            att_opt.step()

        global_T_weight = T_w.data.tolist()
        prob = torch.softmax(T_w, dim=0).detach()

        for key in central_node.model.state_dict():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key])
                for i, ci in enumerate(select_list):
                    temp += prob[i] * client_nodes[ci].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(
                    temp / prob.sum())
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(
                        central_node.model.state_dict()[key])

    # ------------------------------------------------------------------- FedLWS
    elif args.server_method == 'fedlws':
        client_params = [copy.deepcopy(client_nodes[i].model.state_dict())
                         for i in select_list]
        global_params  = copy.deepcopy(central_node.model.state_dict())
        fedavg_params  = copy.deepcopy(client_params[0])

        for name in client_params[0]:
            fedavg_params[name] = sum(
                client_params[k][name] * agg_weights[k]
                for k in range(len(client_params)))

        # Layer-wise scaling
        # Note: skip BN running stats (running_mean/running_var/num_batches_tracked)
        # from gamma computation and application — they should not be scaled.
        _BN_SKIP = ('running_mean', 'running_var', 'num_batches_tracked')

        def _layer_tensors(params):
            layers, buf = [], torch.tensor([]).cuda()
            for name in params:
                if any(s in name for s in _BN_SKIP):
                    continue
                buf = torch.cat((buf, params[name].reshape(-1).float()))
                if 'bias' in name:
                    layers.append(buf)
                    buf = torch.tensor([]).cuda()
            return layers

        cur_w  = _layer_tensors(fedavg_params)
        last_w = _layer_tensors(global_params)
        clients_w = [_layer_tensors(cp) for cp in client_params]

        layer_gammas = []
        for i in range(len(last_w)):
            grad       = torch.norm(cur_w[i] - last_w[i], p=2)
            layer_grad = [clients_w[k][i] - last_w[i]
                          for k in range(len(client_params))]
            global_grad = torch.mean(torch.stack(layer_grad), dim=0)
            l2_norms    = [torch.norm(lg - global_grad, p=2) for lg in layer_grad]
            tau = args.beta * (sum(l2_norms) / len(l2_norms))
            tau = torch.clamp(tau, min=args.min_tau, max=args.max_tau)
            gamma = torch.norm(last_w[i], p=2) / (
                torch.norm(last_w[i], p=2) + tau * grad)
            layer_gammas.append(gamma)

        cur_layer = 0
        for name in fedavg_params:
            if any(s in name for s in _BN_SKIP):
                continue  # leave running stats unchanged
            fedavg_params[name] = fedavg_params[name] * layer_gammas[cur_layer]
            if 'bias' in name:
                cur_layer += 1

        central_node.model.load_state_dict(fedavg_params)
        for key in central_node.model.state_dict():
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(
                        central_node.model.state_dict()[key])

    # ----------------------------------------------------------------- Scaffold
    elif args.server_method == 'scaffold':
        fedavg_params = {}
        for k, v in client_nodes[select_list[0]].model.state_dict().items():
            vals, ck_vals = [], []
            for j, ci in zip(select_list, agg_weights):
                if k in client_nodes[select_list[0]].delta_y:
                    vals.append(client_nodes[j].delta_y[k] / len(select_list))
                    ck_vals.append(client_nodes[j].delta_control[k] / len(select_list))
                else:
                    vals.append(client_nodes[j].model.state_dict()[k] / len(select_list))
            if k in client_nodes[select_list[0]].delta_y:
                fedavg_params[k] = sum(vals) + central_node.model.state_dict()[k]
                central_node.control[k].data += (
                    sum(ck_vals) * (len(select_list) / args.node_num))
            else:
                fedavg_params[k] = sum(vals)
        central_node.model.load_state_dict(fedavg_params)
        for key in central_node.model.state_dict():
            if 'num_batches_tracked' not in key:
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(
                        central_node.model.state_dict()[key])

    # ------------------------------------------------------------------- FedBN
    elif args.server_method == 'fedbn':
        # Aggregate everything except BatchNorm layers
        for key in central_node.model.state_dict():
            if 'bn' in key or 'BatchNorm' in key:
                continue
            if 'num_batches_tracked' in key:
                central_node.model.state_dict()[key].data.copy_(
                    client_nodes[select_list[0]].model.state_dict()[key])
            else:
                temp = torch.zeros_like(central_node.model.state_dict()[key],
                                        dtype=torch.float32)
                for i, ci in enumerate(select_list):
                    temp += agg_weights[i] * client_nodes[ci].model.state_dict()[key]
                central_node.model.state_dict()[key].data.copy_(temp)
                for ci in select_list:
                    client_nodes[ci].model.state_dict()[key].data.copy_(temp)

    # ------------------------------------------------------------------- SioBN
    elif args.server_method == 'siobn':
        for key in central_node.model.state_dict():
            if 'bn' in key or 'BatchNorm' in key:
                # Aggregate only affine parameters (weight, bias), not statistics
                if 'weight' in key or 'bias' in key:
                    temp = torch.zeros_like(central_node.model.state_dict()[key],
                                            dtype=torch.float32)
                    for i, ci in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[ci].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for ci in select_list:
                        client_nodes[ci].model.state_dict()[key].data.copy_(temp)
            else:
                if 'num_batches_tracked' in key:
                    central_node.model.state_dict()[key].data.copy_(
                        client_nodes[select_list[0]].model.state_dict()[key])
                else:
                    temp = torch.zeros_like(central_node.model.state_dict()[key],
                                            dtype=torch.float32)
                    for i, ci in enumerate(select_list):
                        temp += agg_weights[i] * client_nodes[ci].model.state_dict()[key]
                    central_node.model.state_dict()[key].data.copy_(temp)
                    for ci in select_list:
                        client_nodes[ci].model.state_dict()[key].data.copy_(temp)

    # --------------------------------------------------------------- SingleSet
    elif args.server_method == 'singleset':
        pass  # No aggregation (local training baseline)

    else:
        raise ValueError(f'Undefined server method: {args.server_method}')

    return central_node, client_nodes
