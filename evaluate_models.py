import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

DATA_DIR = "dataset/dataset_new"
IMG_SIZE = 96
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

# ---------- Test set (same transform used during training validation, no augmentation) ----------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
class_names = test_dataset.classes

# ---------- Rebuild model architectures (must match training exactly) ----------
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

def load_cnn():
    model = DrowsinessCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("cnn_baseline.pth", map_location=DEVICE))
    model.eval()
    return model

def load_mobilenet():
    model = models.mobilenet_v3_small(weights=None)  # architecture only, we load our own trained weights
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load("mobilenet_finetuned.pth", map_location=DEVICE))
    model.eval()
    return model

# ---------- Evaluation function ----------
def evaluate(model, name):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    print(f"\n{'='*50}")
    print(f"Results for: {name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print("Rows = actual class, Columns = predicted class")
    print("Class order:", class_names)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# ---------- Run evaluation ----------
cnn_model = load_cnn()
mobilenet_model = load_mobilenet()

cnn_results = evaluate(cnn_model, "Custom CNN (from scratch)")
mobilenet_results = evaluate(mobilenet_model, "MobileNetV3-Small (fine-tuned)")

# ---------- Final comparison table ----s------
print(f"\n{'='*50}")
print("FINAL COMPARISON")
print(f"{'='*50}")
print(f"{'Model':<30}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<10}")
for name, res in [("Custom CNN", cnn_results), ("MobileNetV3-Small", mobilenet_results)]:
    print(f"{name:<30}{res['accuracy']*100:<12.2f}{res['precision']:<12.4f}{res['recall']:<12.4f}{res['f1']:<10.4f}")