import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
import csv

# ---------- Config ----------
DATA_DIR = "dataset/dataset_new"
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 15
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

# ---------- Load & split data (70:30 train:val) ----------
full_train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
class_names = full_train.classes

train_size = int(0.7 * len(full_train))
val_size = len(full_train) - train_size
train_subset, val_subset = random_split(full_train, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

full_train_noaug = datasets.ImageFolder(f"{DATA_DIR}/train", transform=val_transform)
val_subset = torch.utils.data.Subset(full_train_noaug, val_subset.indices)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Classes: {class_names}")
print(f"Train: {len(train_subset)}  |  Validation: {len(val_subset)}")

# ---------- Custom lightweight CNN ----------
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DrowsinessCNN(num_classes=len(class_names)).to(DEVICE)

# ---------- Loss & optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

print("\nStarting training...\n")
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

# ---------- Save the trained model ----------
torch.save(model.state_dict(), "cnn_baseline.pth")
print("\nModel saved as cnn_baseline.pth")

# ---------- Save training history to CSV ----------
with open("cnn_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    for i in range(len(history["train_loss"])):
        writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i],
                          history["val_loss"][i], history["val_acc"][i]])

print("History saved to cnn_history.csv")