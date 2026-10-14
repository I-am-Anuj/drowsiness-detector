import csv
import matplotlib.pyplot as plt

def load_history(filepath):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, train_acc, val_loss, val_acc

def plot_model(filepath, model_name):
    epochs, train_loss, train_acc, val_loss, val_acc = load_history(filepath)

    # ---------- Accuracy plot ----------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
    plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
    plt.title(f"{model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_accuracy.png", dpi=150)
    plt.close()

    # ---------- Loss plot ----------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, marker='o', label='Training Loss')
    plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
    plt.title(f"{model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_loss.png", dpi=150)
    plt.close()

    print(f"Saved: {model_name.lower().replace(' ', '_')}_accuracy.png and _loss.png")

# ---------- Generate graphs for both models ----------
plot_model("cnn_history.csv", "CNN")
plot_model("mobilenet_history.csv", "MobileNetV3")

# ---------- Combined comparison plot (both models' validation accuracy) ----------
cnn_epochs, _, _, _, cnn_val_acc = load_history("cnn_history.csv")
mn_epochs, _, _, _, mn_val_acc = load_history("mobilenet_history.csv")

plt.figure(figsize=(7, 5))
plt.plot(cnn_epochs, cnn_val_acc, marker='o', label='Custom CNN (Val Accuracy)')
plt.plot(mn_epochs, mn_val_acc, marker='o', label='MobileNetV3 (Val Accuracy)')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

print("Saved: model_comparison.png")