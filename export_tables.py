import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import csv

DATA_DIR = "dataset/dataset_new"
IMG_SIZE = 96
BATCH_SIZE = 32
DEVICE = torch.device("cpu")

# ---------- Test set ----------
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
class_names = test_dataset.classes

# ---------- Model architectures ----------
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
        return self.classifier(x)

def load_cnn():
    model = DrowsinessCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("cnn_baseline.pth", map_location=DEVICE))
    model.eval()
    return model

def load_mobilenet():
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load("mobilenet_finetuned.pth", map_location=DEVICE))
    model.eval()
    return model

def get_predictions(model):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

# ---------- Table-saving helpers ----------
def save_table_image(headers, rows, filename, title, col_widths=None, figsize=(8, 2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    if col_widths:
        for i, w in enumerate(col_widths):
            for j in range(len(rows) + 1):
                table[(j, i)].set_width(w)
    # Bold header row
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight="bold")
        table[(0, i)].set_facecolor("#D9D9D9")
    plt.title(title, fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

def save_csv(headers, rows, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved: {filename}")

# ---------- Run evaluation for both models ----------
results_summary = []

for model_fn, model_name in [(load_cnn, "Custom CNN"), (load_mobilenet, "MobileNetV3-Small")]:
    model = model_fn()
    labels, preds = get_predictions(model)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    results_summary.append([model_name, f"{accuracy*100:.2f}%", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])

    # ---------- Confusion matrix table ----------
    cm = confusion_matrix(labels, preds)
    cm_headers = ["Actual \\ Predicted"] + class_names
    cm_rows = [[class_names[i]] + list(cm[i]) for i in range(len(class_names))]

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    save_csv(cm_headers, cm_rows, f"confusion_matrix_{safe_name}.csv")
    save_table_image(cm_headers, cm_rows, f"confusion_matrix_{safe_name}.png",
                      f"Confusion Matrix — {model_name}", figsize=(7, 2.5))

# ---------- Overall performance comparison table (like Table 3 in the paper) ----------
perf_headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
save_csv(perf_headers, results_summary, "performance_metrics.csv")
save_table_image(perf_headers, results_summary, "performance_metrics.png",
                  "Performance Metrics for Different Models", figsize=(8, 2))

print("\nAll tables generated successfully.")