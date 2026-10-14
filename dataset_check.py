import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "dataset/dataset_new"
IMG_SIZE = 96          # small size keeps training fast on CPU
BATCH_SIZE = 32

# ---------- Transforms ----------
# Training: light augmentation to improve generalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),                      # converts image to [0,1] tensor, shape (C,H,W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (needed for MobileNet later)
                          std=[0.229, 0.224, 0.225]),
])

# Test: no augmentation, just resize + normalize
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

# ---------- Datasets ----------
train_dataset = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)

# ---------- Dataloaders ----------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------- Sanity check ----------
print("Classes found:", train_dataset.classes)
print("Class-to-index mapping:", train_dataset.class_to_idx)
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)   # should be [32, 3, 96, 96]
print("Labels in this batch:", labels[:10])