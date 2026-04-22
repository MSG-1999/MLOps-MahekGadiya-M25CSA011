"""
app.py
2-page Streamlit app for CityScapes Image Segmentation
Page 1: Training plots + test metrics
Page 2: Upload 4 images → show ground-truth & predicted masks
"""

import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import cv2
from PIL import Image
import glob

# ─── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 23
IMG_H, IMG_W = 96, 128
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR  = "Question2"
DATA_DIR  = "data"

st.set_page_config(page_title="CityScapes Segmentation", layout="wide", page_icon="🏙️")

# ─── UNet (must match train.py) ────────────────────────────────────────────────
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


@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    model_path = os.path.join(SAVE_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def mask_to_color(mask_np):
    """Convert class index mask to RGB color image."""
    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    colored = (cmap(mask_np / NUM_CLASSES)[:, :, :3] * 255).astype(np.uint8)
    return colored


def predict(model, img_np):
    """Run model on a single numpy image (H,W,3) uint8."""
    img = cv2.resize(img_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
    pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def get_gt_mask(mask_path):
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    mask = np.max(mask, axis=-1)
    return mask


# ─── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title(" CityScapes Segmentation")
page = st.sidebar.radio("Navigate", [" Page 1: Training Results", "🖼️ Page 2: Run Inference"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Page 1: Training Results":
    st.title(" Training Results — CityScapes UNet Segmentation")

    results_path = os.path.join(SAVE_DIR, "results.json")
    if not os.path.exists(results_path):
        st.error("❌ results.json not found. Please run train.py first.")
        st.stop()

    with open(results_path) as f:
        results = json.load(f)

    test_miou   = results["test_miou"]
    test_mdice  = results["test_mdice"]
    train_losses = results["train_losses"]
    train_mious  = results["train_mious"]
    train_mdices = results["train_mdices"]
    epochs = list(range(1, len(train_losses) + 1))

    # ── Test Metrics ──
    st.subheader("🎯 Test Set Performance")
    col1, col2 = st.columns(2)
    col1.metric("Test mIOU",  f"{test_miou:.4f}",  delta="Target > 0.48")
    col2.metric("Test mDice", f"{test_mdice:.4f}", delta="Target > 0.48")

    st.divider()

    # ── Training Plots ──
    st.subheader("📈 Training Curves")

    # Check if saved plot image exists
    plot_path = os.path.join(SAVE_DIR, "training_plots.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Training Loss, mIOU, mDice", use_column_width=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(epochs, train_losses,  'b-o', linewidth=2); axes[0].set_title("Training Loss");  axes[0].set_xlabel("Epoch"); axes[0].grid(True)
        axes[1].plot(epochs, train_mious,   'g-o', linewidth=2); axes[1].set_title("Training mIOU");  axes[1].set_xlabel("Epoch"); axes[1].grid(True)
        axes[2].plot(epochs, train_mdices,  'r-o', linewidth=2); axes[2].set_title("Training mDice"); axes[2].set_xlabel("Epoch"); axes[2].grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Individual plots ──
    col1, col2, col3 = st.columns(3)
    for col, name, color, data in [
        (col1, "Loss",  "steelblue",  train_losses),
        (col2, "mIOU",  "seagreen",   train_mious),
        (col3, "mDice", "tomato",     train_mdices),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, data, color=color, linewidth=2, marker='o', markersize=4)
        ax.set_title(f"Training {name}"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
        col.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🖼️ Page 2: Image Segmentation Inference")
    st.markdown("Upload **4 images from the test set** to see Ground Truth vs Predicted segmentation masks.")

    model_path = os.path.join(SAVE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        st.error("❌ best_model.pth not found. Please run train.py first.")
        st.stop()

    model = load_model()
    st.success(f"✅ Model loaded successfully (device: {DEVICE})")

    # ── Get test image/mask paths ──
    from sklearn.model_selection import train_test_split
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "CameraRGB", "*")))
    mask_paths  = sorted(glob.glob(os.path.join(DATA_DIR, "CameraMask", "*")))
    _, test_imgs, _, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    st.info(f"📂 {len(test_imgs)} test images available. Upload any 4 from the test set, or use auto-select below.")

    # ── Auto-select 4 test images ──
    if st.button("🎲 Auto-select 4 random test images"):
        indices = np.random.choice(len(test_imgs), 4, replace=False)
        st.session_state["selected_indices"] = indices.tolist()

    uploaded = st.file_uploader("Or upload 4 test images:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # ── Display results ──
    def show_results(img_list, mask_list):
        for i, (img_np, mask_path) in enumerate(zip(img_list, mask_list)):
            st.markdown(f"---\n### Image {i+1}")
            pred_mask = predict(model, img_np)
            gt_mask   = get_gt_mask(mask_path)

            col1, col2, col3 = st.columns(3)
            img_resized = cv2.resize(img_np, (IMG_W, IMG_H))
            col1.image(img_resized, caption="Input Image", use_column_width=True)
            col2.image(mask_to_color(gt_mask),   caption="Ground Truth Mask",  use_column_width=True)
            col3.image(mask_to_color(pred_mask),  caption="Predicted Mask",     use_column_width=True)

    if uploaded and len(uploaded) >= 4:
        imgs = []
        for f in uploaded[:4]:
            arr = np.array(Image.open(f).convert("RGB"))
            imgs.append(arr)
        # Try to match uploaded filenames to test set masks
        matched_masks = test_masks[:4]
        show_results(imgs, matched_masks)

    elif "selected_indices" in st.session_state:
        indices = st.session_state["selected_indices"]
        imgs, masks = [], []
        for idx in indices:
            img = cv2.imread(test_imgs[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            masks.append(test_masks[idx])
        show_results(imgs, masks)
    else:
        st.markdown(" Click **Auto-select** or upload 4 images to get started.")

        # Show color legend
        st.subheader(" Segmentation Class Color Map")
        cmap = plt.get_cmap("tab20", NUM_CLASSES)
        fig, ax = plt.subplots(figsize=(12, 2))
        for i in range(NUM_CLASSES):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=cmap(i / NUM_CLASSES)))
            ax.text(i + 0.5, 0.5, str(i), ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        ax.set_xlim(0, NUM_CLASSES); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title("Class IDs (0-22)")
        st.pyplot(fig)
        plt.close()