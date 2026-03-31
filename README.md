# FL-MedClsBench

**Federated Learning Benchmark for Medical Image Classification**

A benchmark framework for evaluating federated and personalized federated learning methods on multi-center medical image classification.

---

## Dataset: FL_Skin

Multi-center skin lesion classification dataset with 8 diagnostic categories.

| Site | Dataset | Approx. Samples |
|------|---------|-----------------|
| Site 1 | Derm7pt | ~685 |
| Site 2 | HAM10000 | ~7,000 |
| Site 3 | ISIC_2019 | ~10,700 |
| Site 4 | PAD-UFES-20 | ~1,600 |

**8 Classes:**

| Label | Category |
|-------|----------|
| 0 | Melanoma (mel) |
| 1 | Melanocytic nevus (nv) |
| 2 | Basal cell carcinoma (bcc) |
| 3 | Actinic keratosis (ak) |
| 4 | Benign keratosis (bkl) |
| 5 | Dermatofibroma (df) |
| 6 | Vascular lesion (vasc) |
| 7 | Squamous cell carcinoma (scc) |

- **Format**: 2D RGB images (`.jpg` / `.png`)
- **Input size**: 224×224
- **Splits**: Predefined train/val/test splits for seeds 0, 1, 2
- **CSV format**: `{Site}/train_seed{seed}.csv` with columns `name, label`

---

## Implemented Methods

### Federated Learning (Global)

| Method | Client | Server | Reference |
|--------|--------|--------|-----------|
| LocalTrain | local_train | singleset | Baseline |
| FedAvg | local_train | fedavg | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) |
| FedProx | fedprox | fedavg | [Li et al., 2020](https://arxiv.org/abs/1812.06127) |
| MOON | moon | fedavg | [Li et al., 2021](https://arxiv.org/abs/2103.16257) |
| FedAWA | local_train | fedawa | — |
| FedNova | local_train | fednova | [Wang et al., 2020](https://arxiv.org/abs/2007.06234) |
| PN | local_train | fedavg | — |
| FedRDN | fedrdn | fedavg | — |
| FedLWS | local_train | fedlws | — |

### Personalized Federated Learning

| Method | Client | Server | Reference |
|--------|--------|--------|-----------|
| FedBN | local_train | fedbn | [Li et al., 2021](https://arxiv.org/abs/2102.07623) |
| SioBN | local_train | siobn | — |
| FedPer | local_train | fedper | [Arivazhagan et al., 2019](https://arxiv.org/abs/1912.00818) |
| FedROD | fedrod | fedavg | [Chen & Chao, 2022](https://arxiv.org/abs/2203.02338) |
| Ditto | ditto | fedavg | [Li et al., 2021](https://arxiv.org/abs/2012.04235) |

---

## Models

| Model | Input | Description |
|-------|-------|-------------|
| **ResNet50** (default) | (3, 224, 224) | 2D ResNet-50 with ImageNet pretraining option |
| ResNet18 | (3, 224, 224) | 2D ResNet-18 (lightweight) |

All models return `(logit, feature)` tuples for compatibility with MOON, FedROD, and Ditto.

---

## Installation

```bash
conda create -n fedcls python=3.9
conda activate fedcls
pip install -r requirements.txt
```

---

## Usage

### Basic training

```bash
python main_cls.py \
    --method FedAvg \
    --dataset FLSkin \
    --local_model ResNet50 \
    --data_path ../FL_Skin \
    --num_classes 8 \
    --T 500 --E 5 \
    --lr 0.0001 \
    --batchsize 24 \
    --optimizer adam \
    --device 0 \
    --exp_name bench
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | FedAvg | FL method name |
| `--dataset` | FedBCa | Dataset: `FLSkin` |
| `--local_model` | ResNet50 | Model architecture |
| `--data_path` | ../FedBCa | Path to dataset root |
| `--num_classes` | 2 | Number of classes |
| `--T` | 500 | Communication rounds |
| `--E` | 5 | Local epochs per round |
| `--lr` | 0.0001 | Learning rate |
| `--batchsize` | 24 | Batch size |
| `--optimizer` | adam | Optimizer: `adam` / `sgd` |
| `--device` | 0 | GPU device ID |
| `--seed` | 0 | Random seed |
| `--pretrained` | False | Use ImageNet pretrained weights |

### Method-specific arguments

| Argument | Methods | Description |
|----------|---------|-------------|
| `--mu` | FedProx, MOON, Ditto | Proximal / regularization coefficient |
| `--temperature` | MOON | Contrastive learning temperature |
| `--lr_per` | Ditto | Personalized model learning rate |
| `--beta` | FedLWS | Layer-wise scaling beta |
| `--server_epochs` | FedAWA | Server-side optimization steps |
| `--reg_distance` | FedAWA | Distance metric: `cos` or `euc` |

### Run scripts

Per-method scripts are provided in `scripts/`:

```bash
cd scripts
bash run_FedAvg.sh 0       # Train FedAvg on GPU 0
bash run_FedProx.sh 1      # Train FedProx on GPU 1
bash run_Ditto.sh 2        # Train Ditto on GPU 2
```

---

## Evaluation Metrics

- **Accuracy (ACC)**
- **Recall**
- **Precision**
- **F1 Score**
- **AUC**

Reported as:
- **Best**: test metrics at the epoch with best validation performance
- **Last**: mean of test metrics over the final 5 epochs

---

## Output Structure

Each experiment saves to `results/<exp_name>/`:

```
results/bench/
├── config.json              # Full hyperparameter settings
├── metrics_seed0.csv        # Per-round metrics (train/val/test)
├── metrics_seed1.csv
├── metrics_seed2.csv
├── summary.json             # Final mean ± std across seeds
└── curves_seed0.png         # Loss & metric curves
```

---

## File Structure

```
FL-MedClsBench/
├── main_cls.py          # Main training entry point
├── datasets.py          # Data loaders
├── nodes.py             # Client / Server Node definitions
├── server_funct.py      # Server-side aggregation algorithms
├── client_funct.py      # Client-side training functions
├── utils.py             # Utilities (init_model, validate, etc.)
├── models_dict/
│   ├── __init__.py
│   ├── resnet2d.py      # 2D ResNet-18 / ResNet-50
│   ├── med3dcnn.py      # Simple 3D CNN
│   └── resnet3d.py      # 3D ResNet-10 / ResNet-18
├── scripts/             # Per-method run scripts
└── requirements.txt
```
