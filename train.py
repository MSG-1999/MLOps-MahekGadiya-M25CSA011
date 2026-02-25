import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import json

# ── Hyperparameters (change these for Q7) ──────────────────────────
LEARNING_RATE  = 0.001
OPTIMIZER_NAME = "adam"   # "adam" | "sgd" | "rmsprop"
BATCH_SIZE     = 32
NUM_EPOCHS     = 10
NUM_CLASSES    = 10
DATA_DIR       = "data"
MODEL_SAVE_PATH = "model.pth"
VAL_SPLIT       = 0.2
# ───────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Transforms ────────────────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

aug_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── Dataset: split train → train + val ────────────────────────────
full_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                    transform=aug_transform)

val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

class_names = full_dataset.classes
print(f"Classes     : {class_names}")
print(f"Train size  : {train_size}  |  Val size: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────────────────────
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

if OPTIMIZER_NAME == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER_NAME == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
elif OPTIMIZER_NAME == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")

print(f"\nConfig → LR={LEARNING_RATE} | Optimizer={OPTIMIZER_NAME} | Batch={BATCH_SIZE} | Epochs={NUM_EPOCHS}\n")

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = 100.0 * correct / total

    # Validate
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    val_loss = running_loss / total
    val_acc  = 100.0 * correct / total

    train_losses.append(train_loss);  val_losses.append(val_loss)
    train_accs.append(train_acc);     val_accs.append(val_acc)

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel saved → {MODEL_SAVE_PATH}")

# Plot curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(train_losses, label="Train Loss", marker="o")
ax1.plot(val_losses,   label="Val Loss",   marker="o")
ax1.set_title("Loss Curve"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True)

ax2.plot(train_accs, label="Train Acc", marker="o")
ax2.plot(val_accs,   label="Val Acc",   marker="o")
ax2.set_title("Accuracy Curve"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.legend(); ax2.grid(True)

tag = f"lr{LEARNING_RATE}_opt{OPTIMIZER_NAME}_bs{BATCH_SIZE}"
plt.suptitle(f"Training Curves | {tag}")
plt.tight_layout()
plt.savefig(f"training_curves_{tag}.png")
print(f"Curves saved → training_curves_{tag}.png")

with open(f"results_{tag}.json", "w") as f:
    json.dump({
        "lr": LEARNING_RATE, "optimizer": OPTIMIZER_NAME,
        "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
        "final_train_acc": round(train_accs[-1], 2),
        "final_val_acc":   round(val_accs[-1],   2),
    }, f, indent=2)
print(f"Results saved → results_{tag}.json")
