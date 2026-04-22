"""
train.py
UNet-based Image Segmentation on CityScapes dataset
- 23 segmentation classes
- 80/20 train-test split with seed 42
- Computes mIOU and mDice during training
- Saves plots and best model
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 23
IMG_H, IMG_W = 96, 128
BATCH_SIZE   = 8
EPOCHS       = 15
LR           = 1e-3
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = "data"
SAVE_DIR     = "Question2"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Using device: {DEVICE}")

# ─── Dataset ───────────────────────────────────────────────────────────────────
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read Image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        # Read Mask
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img  = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return img, mask


# ─── UNet ──────────────────────────────────────────────────────────────────────
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        self.up4   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4  = double_conv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = double_conv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = double_conv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = double_conv(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ─── Metrics ───────────────────────────────────────────────────────────────────
def compute_miou(preds, masks, num_classes=NUM_CLASSES):
    iou_list = []
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()
        if union == 0:
            continue
        iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0

def compute_mdice(preds, masks, num_classes=NUM_CLASSES):
    dice_list = []
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)
        intersection = (pred_cls & mask_cls).sum()
        denom = pred_cls.sum() + mask_cls.sum()
        if denom == 0:
            continue
        dice_list.append(2 * intersection / denom)
    return np.mean(dice_list) if dice_list else 0.0


# ─── Data Loading ──────────────────────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "CameraRGB", "*")))
mask_paths  = sorted(glob.glob(os.path.join(DATA_DIR, "CameraMask", "*")))

print(f"Total images: {len(image_paths)}, masks: {len(mask_paths)}")

train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=SEED
)

train_ds = CityscapesDataset(train_imgs, train_masks)
test_ds  = CityscapesDataset(test_imgs,  test_masks)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

# ─── Training ──────────────────────────────────────────────────────────────────
model     = UNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

train_losses, train_mious, train_mdices = [], [], []
best_miou = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss, epoch_miou, epoch_mdice = 0, 0, 0

    for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        epoch_loss  += loss.item()
        epoch_miou  += compute_miou(preds, masks)
        epoch_mdice += compute_mdice(preds, masks)

    n = len(train_dl)
    avg_loss  = epoch_loss  / n
    avg_miou  = epoch_miou  / n
    avg_mdice = epoch_mdice / n

    train_losses.append(avg_loss)
    train_mious.append(avg_miou)
    train_mdices.append(avg_mdice)
    scheduler.step(avg_loss)

    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | mIOU: {avg_miou:.4f} | mDice: {avg_mdice:.4f}")

    if avg_miou > best_miou:
        best_miou = avg_miou
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"  → Saved best model (mIOU: {best_miou:.4f})")

# ─── Plots ─────────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(epochs, train_losses,  'b-o', linewidth=2); axes[0].set_title("Training Loss");  axes[0].set_xlabel("Epoch"); axes[0].grid(True)
axes[1].plot(epochs, train_mious,   'g-o', linewidth=2); axes[1].set_title("Training mIOU");  axes[1].set_xlabel("Epoch"); axes[1].grid(True)
axes[2].plot(epochs, train_mdices,  'r-o', linewidth=2); axes[2].set_title("Training mDice"); axes[2].set_xlabel("Epoch"); axes[2].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_plots.png"), dpi=150)
plt.close()
print("Saved training_plots.png")

# Save individually too
for data, name, color in [(train_losses, "loss", "blue"), (train_mious, "miou", "green"), (train_mdices, "mdice", "red")]:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, data, color=color, linewidth=2, marker='o')
    plt.title(f"Training {name.upper()}"); plt.xlabel("Epoch"); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, f"training_{name}.png"), dpi=150)
    plt.close()

# ─── Test Evaluation ───────────────────────────────────────────────────────────
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
model.eval()
test_miou, test_mdice = 0, 0

with torch.no_grad():
    for imgs, masks in tqdm(test_dl, desc="Evaluating on test set"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1)
        test_miou  += compute_miou(preds, masks)
        test_mdice += compute_mdice(preds, masks)

test_miou  /= len(test_dl)
test_mdice /= len(test_dl)

print(f"\n{'='*40}")
print(f"Test mIOU  : {test_miou:.4f}")
print(f"Test mDice : {test_mdice:.4f}")
print(f"{'='*40}")

# Save results to file for Streamlit app
import json
results = {
    "test_miou":    round(test_miou, 4),
    "test_mdice":   round(test_mdice, 4),
    "train_losses": train_losses,
    "train_mious":  train_mious,
    "train_mdices": train_mdices,
}
with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
    json.dump(results, f)
print("Saved results.json")