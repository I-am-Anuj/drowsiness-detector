import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import time
import csv

# ---------- Config ----------
DATA_DIR = "dataset/dataset_new"
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 12
DEVICE = torch.device("cpu")

# ---------- Transforms ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------- Same 70:30 split, same seed ----------
full_train_aug = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
full_train_noaug = datasets.ImageFolder(f"{DATA_DIR}/train", transform=val_transform)
class_names = full_train_aug.classes

train_size = int(0.7 * len(full_train_aug))
val_size = len(full_train_aug) - train_size
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = torch.utils.data.random_split(
    range(len(full_train_aug)), [train_size, val_size], generator=generator
)

train_subset = Subset(full_train_aug, train_indices.indices)
val_subset = Subset(full_train_noaug, val_indices.indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Classes: {class_names}")
print(f"Train: {len(train_subset)}  |  Validation: {len(val_subset)}")

# ---------- Load pretrained MobileNetV3-Small ----------
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few feature blocks so they can adapt to our task
for layer in model.features[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

# Replace and unfreeze the final classification layer
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, len(class_names))
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ---------- Loss & optimizer ----------
criterion = nn.CrossEntropyLoss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.0001)

# ---------- History tracking ----------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# ---------- Training loop ----------
def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if train_mode:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

print("\nStarting training (fine-tuning)...\n")
for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, train_mode=False)
    elapsed = time.time() - start

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"{elapsed:.1f}s")

torch.save(model.state_dict(), "mobilenet_finetuned.pth")
print("\nModel saved as mobilenet_finetuned.pth")

# ---------- Save training history to CSV ----------
with open("mobilenet_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    for i in range(len(history["train_loss"])):
        writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i],
                          history["val_loss"][i], history["val_acc"][i]])

print("History saved to mobilenet_history.csv")