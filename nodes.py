"""nodes.py — Node (client/server) definitions for FL-MedClsBench."""

import copy
import numpy as np
import torch
from utils import init_model, init_optimizer, model_parameter_vector
from utils import compute_dataset_statistics


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:

    def __init__(self, args, client_id, client_name,
                 train_loader, val_loader, test_loader):
        self.client_id   = client_id
        self.client_name = client_name
        self.args        = args
        self.node_num    = args.node_num
        self.num_classes = args.num_classes

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.model = init_model(args.local_model, args).cuda()

        # ---- FedROD personalised head ----
        if args.client_method == 'fedrod':
            self.p_head = copy.deepcopy(self.model.fc).cuda()

        # ---- FedDYN ----
        if args.client_method == 'feddyn':
            v = model_parameter_vector(args, self.model)
            self.old_grad = torch.zeros_like(v)
        if 'feddyn' in getattr(args, 'server_method', ''):
            self.server_state = copy.deepcopy(self.model)
            for p in self.server_state.parameters():
                p.data = torch.zeros_like(p.data)

        # ---- Scaffold ----
        if args.method == 'Scaffold':
            self.control       = {k: torch.zeros_like(v.data)
                                  for k, v in self.model.named_parameters()}
            self.delta_control = {k: torch.zeros_like(v.data)
                                  for k, v in self.model.named_parameters()}
            self.delta_y       = {k: torch.zeros_like(v.data)
                                  for k, v in self.model.named_parameters()}

        # ---- MOON previous model ----
        if args.method == 'MOON':
            self.pre_model = copy.deepcopy(self.model)

        # ---- FedRDN local statistics ----
        if args.method == 'FedRDN' and train_loader is not None:
            self.local_stats          = compute_dataset_statistics(train_loader)
            self.FedRDNTransform_train = None
            self.FedRDNTransform_test  = None

        # ---- Ditto personalised model ----
        if args.method == 'Ditto':
            self.p_model = init_model(args.local_model, args).cuda()
            if args.optimizer == 'adam':
                self.p_optimizer = torch.optim.Adam(
                    self.p_model.parameters(),
                    lr=getattr(args, 'lr_per', args.lr),
                    weight_decay=args.local_wd_rate)
            else:
                self.p_optimizer = torch.optim.SGD(
                    self.p_model.parameters(),
                    lr=getattr(args, 'lr_per', args.lr),
                    momentum=args.momentum,
                    weight_decay=args.local_wd_rate)

        self.optimizer = init_optimizer(client_id, self.model, args)

        self.averager  = Averager(client_id, client_name)
        self.maxer     = Maxer(client_id, client_name)
        self.recorder  = Recorder(client_id, client_name)


# ---------------------------------------------------------------------------
# Metric trackers — classification
# ---------------------------------------------------------------------------

class Averager:
    """Accumulates metrics over the last N rounds (used for 'last' reporting)."""

    def __init__(self, client_id, client_name):
        self.client_id   = client_id
        self.client_name = client_name
        self.reset()

    def reset(self):
        self.loss = self.acc = self.recall = self.prec = self.f1 = self.auc = 0.0
        self.num  = 0

    def update(self, loss, acc, recall, prec, f1, auc):
        self.loss   += loss
        self.acc    += acc
        self.recall += recall
        self.prec   += prec
        self.f1     += f1
        self.auc    += auc
        self.num    += 1

    def log(self, is_log=False):
        n = max(self.num, 1)
        loss, acc, recall, prec, f1, auc = (
            self.loss / n, self.acc / n, self.recall / n,
            self.prec / n, self.f1 / n, self.auc / n)
        if is_log:
            print('Test  {:<12.12}, loss:{:.5f}, ACC:{:.3f}, '
                  'Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}'.format(
                      self.client_name, loss, acc, recall, prec, f1, auc),
                  flush=True)
        return loss, acc, recall, prec, f1, auc


class Maxer:
    """Tracks best validation metric (used for model selection)."""

    def __init__(self, client_id, client_name):
        self.client_id   = client_id
        self.client_name = client_name
        self.reset()

    def reset(self):
        self.loss = self.acc = self.recall = self.prec = \
            self.f1 = self.auc = 0.0
        self.epoch = 0

    def update(self, epoch, loss, acc, recall, prec, f1, auc):
        if acc > self.acc:
            self.loss, self.acc, self.recall, self.prec, \
                self.f1, self.auc, self.epoch = \
                loss, acc, recall, prec, f1, auc, epoch
        return acc > self.acc  # True if improved

    def log(self, is_log=False):
        if is_log:
            print('BestV {:<12.12}, loss:{:.5f}, ACC:{:.3f}, '
                  'Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}, '
                  'Epoch:{}'.format(self.client_name,
                      self.loss, self.acc, self.recall,
                      self.prec, self.f1, self.auc, self.epoch), flush=True)
        return self.loss, self.acc, self.recall, self.prec, \
               self.f1, self.auc, self.epoch


class Recorder:
    """Records test metrics corresponding to best val epoch."""

    def __init__(self, client_id, client_name):
        self.client_id   = client_id
        self.client_name = client_name
        self.reset()

    def reset(self):
        self.loss = self.acc = self.recall = self.prec = \
            self.f1 = self.auc = 0.0
        self.epoch = 0

    def update(self, epoch, loss, acc, recall, prec, f1, auc):
        self.loss, self.acc, self.recall, self.prec, \
            self.f1, self.auc, self.epoch = \
            loss, acc, recall, prec, f1, auc, epoch

    def log(self, is_log=False):
        if is_log:
            print('BestT {:<12.12}, loss:{:.5f}, ACC:{:.3f}, '
                  'Recall:{:.3f}, Prec:{:.3f}, F1:{:.3f}, AUC:{:.3f}, '
                  'Epoch:{}'.format(self.client_name,
                      self.loss, self.acc, self.recall,
                      self.prec, self.f1, self.auc, self.epoch), flush=True)
        return self.loss, self.acc, self.recall, self.prec, \
               self.f1, self.auc, self.epoch


class SeedAverager:
    """Aggregates best-test metrics across multiple random seeds."""

    def __init__(self, client_id, client_name):
        self.client_id   = client_id
        self.client_name = client_name
        self.acc = []; self.recall = []; self.prec = []
        self.f1  = []; self.auc    = []

    def update(self, acc=0, recall=0, prec=0, f1=0, auc=0):
        self.acc.append(acc); self.recall.append(recall)
        self.prec.append(prec); self.f1.append(f1); self.auc.append(auc)

    def log(self, is_log=False, details=False):
        m_acc,  s_acc   = np.mean(self.acc),    np.std(self.acc)
        m_rec,  s_rec   = np.mean(self.recall), np.std(self.recall)
        m_prec, s_prec  = np.mean(self.prec),   np.std(self.prec)
        m_f1,   s_f1    = np.mean(self.f1),     np.std(self.f1)
        m_auc,  s_auc   = np.mean(self.auc),    np.std(self.auc)
        if details:
            print(f'Client: {self.client_name}')
            print(f'  ACC   : {self.acc}')
            print(f'  Recall: {self.recall}')
            print(f'  Prec  : {self.prec}')
            print(f'  F1    : {self.f1}')
            print(f'  AUC   : {self.auc}')
            print('---')
        if is_log:
            print(
                'Test  {:<12.12}, '
                'ACC:{:.3f}[{:.3f}], Recall:{:.3f}[{:.3f}], '
                'Prec:{:.3f}[{:.3f}], F1:{:.3f}[{:.3f}], '
                'AUC:{:.3f}[{:.3f}]'.format(
                    self.client_name,
                    m_acc, s_acc, m_rec, s_rec,
                    m_prec, s_prec, m_f1, s_f1, m_auc, s_auc),
                flush=True)
        return (m_acc, s_acc, m_rec, s_rec, m_prec, s_prec,
                m_f1, s_f1, m_auc, s_auc)
