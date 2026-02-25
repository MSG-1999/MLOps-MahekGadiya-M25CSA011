import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
import sys

# ── Config ─────────────────────────────────────────────────────────
NUM_CLASSES    = 10
DATA_DIR       = "data"
MODEL_PATH     = "model.pth"
# Q6: image at data/test/5/0005.jpg
DEFAULT_IMAGE  = os.path.join(DATA_DIR, "test", "5", "0005.jpg")
# ───────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── Model ──────────────────────────────────────────────────────────
def build_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

# ── Load test set ──────────────────────────────────────────────────
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
class_names  = test_dataset.classes
print(f"Classes  : {class_names}")
print(f"Test set : {len(test_dataset)} images\n")

model = build_model()

# ── Run inference on full test set ─────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Overall metrics ────────────────────────────────────────────────
overall_acc = accuracy_score(all_labels, all_preds) * 100
f1          = f1_score(all_labels, all_preds, average="weighted")

print("=" * 55)
print(f"  Overall Accuracy : {overall_acc:.2f}%")
print(f"  F1 Score         : {f1:.4f}")
print("=" * 55)

# ── Class-wise accuracy ────────────────────────────────────────────
print("\nClass-wise Accuracy:")
class5_acc = None
for i, cls in enumerate(class_names):
    mask    = all_labels == i
    cls_acc = accuracy_score(all_labels[mask], all_preds[mask]) * 100
    tag     = "  ← CLASS 5" if cls == "5" else ""
    print(f"  Class {cls}: {cls_acc:.2f}%{tag}")
    if cls == "5":
        class5_acc = cls_acc

print(f"\n  Class 5 Accuracy : {class5_acc:.2f}%")
print("=" * 55)

# ── Full classification report ─────────────────────────────────────
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── Confusion matrix ───────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — Test Set")
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved → confusion_matrix.png\n")

# ── Single image prediction (Q6) ───────────────────────────────────
img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
print(f"Single image test: {img_path}")

if os.path.exists(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        conf, pred_idx = probs.max(1)
    print(f"  Predicted Class : {pred_idx.item()} ({class_names[pred_idx.item()]})")
    print(f"  Confidence      : {conf.item()*100:.2f}%")
else:
    # Try to find any image in test/5/
    fallback_dir = os.path.join(DATA_DIR, "test", "5")
    if os.path.exists(fallback_dir):
        imgs = [f for f in os.listdir(fallback_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if imgs:
            img_path = os.path.join(fallback_dir, sorted(imgs)[0])
            print(f"  [INFO] 0005.jpg not found, using: {img_path}")
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)
                conf, pred_idx = probs.max(1)
            print(f"  Predicted Class : {pred_idx.item()} ({class_names[pred_idx.item()]})")
            print(f"  Confidence      : {conf.item()*100:.2f}%")
    else:
        print(f"  [WARNING] Image not found: {img_path}")
