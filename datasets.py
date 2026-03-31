"""
datasets.py — Multi-dataset dataloader for FL-MedClsBench

Supported datasets:
  1. FedBCa  — 3D T2WI MRI bladder cancer, 4 centers, binary, patient-level eval
  2. FLSkin  — 2D RGB skin lesion images, 4 sites, 8-class, image-level eval

FL_Skin dataset structure:
  FL_Skin/
    {Derm7pt, HAM10000, ISIC_2019, PAD-UFES-20}/
      images/  *.jpg / *.png    ← 2D RGB images
      train_seed{0..2}.csv      ← columns: name, label
      val_seed{0..2}.csv
      test_seed{0..2}.csv
  label_info.csv  → mel=0, nv=1, bcc=2, ak=3, bkl=4, df=5, vasc=6, scc=7

FedBCa dataset structure:
  FedBCa/
    Center{1..4}/
      T2WI/  *.nii.gz          ← patient volumes
      train_seed{0..2}.csv     ← columns: name, label
      val_seed{0..2}.csv
      test_seed{0..2}.csv
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom

SLICE_SIZE = (128, 128)   # H × W after resize


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _load_and_preprocess_volume(nii_path: str) -> np.ndarray:
    """Load NIfTI, return float32 array (D, H, W) normalised to [0, 1].

    NIfTI from FedBCa is stored as (H, W, D); we transpose to (D, H, W).
    """
    vol = nib.load(nii_path).get_fdata().astype(np.float32)
    # FedBCa volumes are (H, W, D) — transpose to (D, H, W)
    vol = vol.transpose(2, 0, 1)
    # Min-max normalisation (patient-wise)
    v_min, v_max = vol.min(), vol.max()
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    return vol


def _resize_slice(slc: np.ndarray, target=SLICE_SIZE) -> np.ndarray:
    """Resize a 2D slice to target (H, W) with bilinear interpolation."""
    fh = target[0] / slc.shape[0]
    fw = target[1] / slc.shape[1]
    return zoom(slc, (fh, fw), order=1).astype(np.float32)


def _slice_to_tensor(slc: np.ndarray) -> torch.Tensor:
    """Convert 2D (H, W) slice to 3-channel tensor (3, H, W)."""
    t = torch.from_numpy(slc).unsqueeze(0)   # (1, H, W)
    return t.repeat(3, 1, 1)                  # (3, H, W)


def _extract_valid_slices(vol: np.ndarray,
                           content_threshold: float = 0.01
                           ) -> List[np.ndarray]:
    """Extract non-trivial axial slices from volume (D, H, W)."""
    slices = []
    for d in range(vol.shape[0]):
        s = vol[d]
        if s.max() > content_threshold:       # skip near-empty slices
            slices.append(_resize_slice(s))
    return slices


def _augment_slice(slc: np.ndarray) -> np.ndarray:
    """Random augmentation on a 2D (H, W) slice."""
    # Horizontal flip
    if np.random.rand() > 0.5:
        slc = np.flip(slc, axis=1)
    # Vertical flip
    if np.random.rand() > 0.5:
        slc = np.flip(slc, axis=0)
    # Intensity jitter
    scale = np.random.uniform(0.9, 1.1)
    shift = np.random.uniform(-0.05, 0.05)
    slc = np.clip(slc * scale + shift, 0.0, 1.0)
    return slc.copy().astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset — training (slice level, all slices from each patient)
# ---------------------------------------------------------------------------

class FedBCaSliceDataset(Dataset):
    """Each item is a single 2D axial slice with the patient-level label.

    Used for *training*.  The whole split CSV is expanded into individual
    slices at initialisation time.
    """

    def __init__(self, data_root: str, center: str, split: str,
                 seed: int, augment: bool = False):
        self.augment = augment
        csv_path = os.path.join(data_root, center, f'{split}_seed{seed}.csv')
        df = pd.read_csv(csv_path)
        img_dir = os.path.join(data_root, center, 'T2WI')

        self.slices: List[np.ndarray] = []
        self.labels: List[int]        = []

        for _, row in df.iterrows():
            label  = int(row['label'])
            fpath  = os.path.join(img_dir, row['name'])
            vol    = _load_and_preprocess_volume(fpath)
            slcs   = _extract_valid_slices(vol)
            self.slices.extend(slcs)
            self.labels.extend([label] * len(slcs))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.augment:
            slc = _augment_slice(slc)
        img = _slice_to_tensor(slc)   # (3, 128, 128)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Dataset — evaluation (patient level)
# ---------------------------------------------------------------------------

class FedBCaPatientDataset(Dataset):
    """Each item is a (slices_tensor, label) pair for one patient.

    slices_tensor shape: (N_slices, 3, H, W)

    Used for *validation and test* to enable patient-level metric computation.
    """

    def __init__(self, data_root: str, center: str, split: str, seed: int):
        csv_path = os.path.join(data_root, center, f'{split}_seed{seed}.csv')
        df = pd.read_csv(csv_path)
        img_dir = os.path.join(data_root, center, 'T2WI')

        self.patients: List[Tuple[torch.Tensor, int]] = []

        for _, row in df.iterrows():
            label  = int(row['label'])
            fpath  = os.path.join(img_dir, row['name'])
            vol    = _load_and_preprocess_volume(fpath)
            slcs   = _extract_valid_slices(vol)
            if not slcs:
                # Fallback: use middle slice
                mid  = vol.shape[0] // 2
                slcs = [_resize_slice(vol[mid])]
            tensors = torch.stack([_slice_to_tensor(s) for s in slcs])  # (N, 3, H, W)
            self.patients.append((tensors, label))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.patients[idx]   # (N_slices×3×H×W tensor, label)


# ---------------------------------------------------------------------------
# Collate for patient dataset (variable #slices per patient)
# ---------------------------------------------------------------------------

def patient_collate_fn(batch):
    """Return list of (slices_tensor, label) — do NOT stack (variable N)."""
    return batch


# ---------------------------------------------------------------------------
# FL_Skin dataset (2D RGB image-level classification)
# ---------------------------------------------------------------------------

import torchvision.transforms as T

# ImageNet statistics for pretrained ResNet
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_SKIN_TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_SKIN_TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class FLSkinTrainDataset(Dataset):
    """2D skin lesion training dataset — image-level labels.

    Each item: (tensor [3,224,224], label)
    """

    def __init__(self, data_root: str, site: str, split: str, seed: int,
                 augment: bool = False):
        from PIL import Image as PilImage
        self._PilImage = PilImage

        csv_path = os.path.join(data_root, site, f'{split}_seed{seed}.csv')
        df = pd.read_csv(csv_path)

        self.img_dir  = os.path.join(data_root, site, 'images')
        self.names    = df['name'].tolist()
        self.labels   = [int(x) for x in df['label'].tolist()]
        self.transform = _SKIN_TRAIN_TRANSFORM if augment else _SKIN_TEST_TRANSFORM

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.names[idx])
        img = self._PilImage.open(img_path).convert('RGB')
        return self.transform(img), self.labels[idx]


class FLSkinEvalDataset(Dataset):
    """2D skin lesion eval dataset — wraps each image as a single-slice
    'patient' so it is compatible with the existing validate() interface.

    Each item: (tensor [1,3,224,224], label)  — same shape as patient loader.
    """

    def __init__(self, data_root: str, site: str, split: str, seed: int):
        from PIL import Image as PilImage
        self._PilImage = PilImage

        csv_path = os.path.join(data_root, site, f'{split}_seed{seed}.csv')
        df = pd.read_csv(csv_path)

        self.img_dir  = os.path.join(data_root, site, 'images')
        self.names    = df['name'].tolist()
        self.labels   = [int(x) for x in df['label'].tolist()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.names[idx])
        img = self._PilImage.open(img_path).convert('RGB')
        tensor = _SKIN_TEST_TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor, self.labels[idx]


# ---------------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------------

class Data:
    """Creates all DataLoaders for FL-MedClsBench.

    Supports:
      - dataset='FedBCa'  → NIfTI 3D MRI, patient-level eval
      - dataset='FLSkin'  → 2D RGB images, image-level eval (via single-slice wrapper)
    """

    def __init__(self, args):
        self.train_loaders = []
        self.val_loaders   = []
        self.test_loaders  = []

        dataset = getattr(args, 'dataset', 'FedBCa')

        if dataset == 'FLSkin':
            self._build_skin(args)
        else:
            self._build_fedbca(args)

    def _build_fedbca(self, args):
        for center in args.client_names:
            train_ds = FedBCaSliceDataset(
                args.data_path, center, 'train', args.random_seed, augment=True)
            val_ds   = FedBCaPatientDataset(
                args.data_path, center, 'val',   args.random_seed)
            test_ds  = FedBCaPatientDataset(
                args.data_path, center, 'test',  args.random_seed)

            self.train_loaders.append(DataLoader(
                train_ds, batch_size=args.batchsize,
                shuffle=True, num_workers=2, pin_memory=True,
                drop_last=len(train_ds) > args.batchsize))

            self.val_loaders.append(DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=patient_collate_fn))
            self.test_loaders.append(DataLoader(
                test_ds, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=patient_collate_fn))

    def _build_skin(self, args):
        for site in args.client_names:
            train_ds = FLSkinTrainDataset(
                args.data_path, site, 'train', args.random_seed, augment=True)
            val_ds   = FLSkinEvalDataset(
                args.data_path, site, 'val',   args.random_seed)
            test_ds  = FLSkinEvalDataset(
                args.data_path, site, 'test',  args.random_seed)

            self.train_loaders.append(DataLoader(
                train_ds, batch_size=args.batchsize,
                shuffle=True, num_workers=4, pin_memory=True,
                drop_last=len(train_ds) > args.batchsize))

            # Use patient_collate_fn: each image is a single-slice 'patient'
            self.val_loaders.append(DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=patient_collate_fn))
            self.test_loaders.append(DataLoader(
                test_ds, batch_size=1, shuffle=False,
                num_workers=2, collate_fn=patient_collate_fn))
