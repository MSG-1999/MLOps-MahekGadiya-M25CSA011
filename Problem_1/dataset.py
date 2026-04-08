"""
q1/dataset.py  —  CIFAR-100 data loading with RAM cache.

Speed trick: images are resized 32→224 ONCE, stored as a (N,3,224,224)
float32 tensor in RAM.  All 10 experiments share the same tensor — zero
extra resize cost after the first build (~30-60 s one-time).
"""

import sys
import time

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from q1.config import (
    BATCH_SIZE, DATA_DIR,
    CIFAR100_MEAN, CIFAR100_STD,
    DEVICE, NUM_WORKERS,
)


# ── Cached dataset ────────────────────────────────────────────────────────────
class CachedCIFAR100(Dataset):
    """
    Resizes ALL images to 224×224 once via a fast parallel loader and caches
    them as a single float32 tensor in RAM.  Subsequent accesses are free.
    """
    _cache: dict = {}

    def __init__(self, train: bool, transform):
        key = "train" if train else "val"
        if key not in CachedCIFAR100._cache:
            print(f"\n  [Cache] Building '{key}' split (resize 32→224, one-time) …",
                  flush=True)
            t0  = time.time()
            raw = torchvision.datasets.CIFAR100(
                DATA_DIR, train=train, download=True,
                transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
            )
            loader = DataLoader(
                raw, batch_size=512, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=False,
            )
            imgs_list, lbl_list = [], []
            for ib, lb in tqdm(loader, desc=f"  [Cache] {key}",
                               file=sys.stdout, dynamic_ncols=True):
                imgs_list.append(ib)
                lbl_list.append(lb)
            imgs   = torch.cat(imgs_list, dim=0)
            labels = torch.cat(lbl_list,  dim=0)
            CachedCIFAR100._cache[key] = (imgs, labels)
            print(
                f"  [Cache] Done in {time.time()-t0:.1f}s  "
                f"shape={tuple(imgs.shape)}  RAM≈{imgs.nbytes/1e9:.2f} GB",
                flush=True,
            )
        self.imgs, self.labels = CachedCIFAR100._cache[key]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.transform(to_pil_image(self.imgs[idx]))
        return img, self.labels[idx].item()


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_loaders(batch_size: int = BATCH_SIZE):
    """
    Build and return (train_loader, val_loader).
    Images in the cache are already 224×224 — NO T.Resize here.
    Call once; share across all experiments.
    """
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(224, padding=28),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        prefetch_factor=4,
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(
        CachedCIFAR100(True,  train_tf), batch_size, shuffle=True,  **kw
    )
    val_loader = DataLoader(
        CachedCIFAR100(False, val_tf),   batch_size, shuffle=False, **kw
    )
    return train_loader, val_loader
