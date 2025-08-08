import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from model import CNN_GRU_Attention  # Make sure your model file is updated

# Load features
X, y = joblib.load("features.pkl")
print(f"Loaded data: {X.shape}, labels: {len(y)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Dataset and split
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_GRU_Attention(input_dim=374, num_classes=len(le.classes_)).to(device)

# Weighted loss for imbalance
class_counts = np.bincount(y_encoded)
weights = 1. / class_counts
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # decay LR every 15 epochs

# Training loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0

for epoch in range(60):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs, _ = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    train_acc = correct / total
    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs, _ = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()

            preds = outputs.argmax(1)
            val_correct += (preds == yb).sum().item()
            val_total += yb.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    val_acc = val_correct / val_total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_acc)
    scheduler.step()

    print(f"Epoch {epoch+1}/60 - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved!")

# Save encoder and scaler
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# 📊 Plot functions
def save_plot(data, label, filename):
    plt.figure()
    plt.plot(data, label=label)
    plt.title(label)
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_plot(train_losses, "Train Loss", "train_loss.png")
save_plot(val_losses, "Validation Loss", "val_loss.png")
save_plot(train_accuracies, "Train Accuracy", "train_acc.png")
save_plot(val_accuracies, "Validation Accuracy", "val_acc.png")

# 📊 Class-wise Precision
report = classification_report(all_targets, all_preds, target_names=le.classes_, output_dict=True)
precision_per_class = [report[c]['precision'] for c in le.classes_]

plt.figure()
plt.bar(le.classes_, precision_per_class)
plt.title("Class-wise Precision")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_precision.png")
plt.close()

# 📊 Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("✅ Training complete. All graphs and model files saved.")

