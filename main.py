import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from model import MultiModalModel, EfficientNetV2Student, KnowledgeDistillationLoss

# Hyperparameters
HYPERPARAMETERS = {
    "img_size": 224,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-5,
    "alpha": 0.5,
    "temperature": 3.0,
    "num_classes": 2,
    "train_split": 0.8
}

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((HYPERPARAMETERS["img_size"], HYPERPARAMETERS["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = "./JUST"
dataset = ImageFolder(root=data_dir, transform=transform)
train_size = int(len(dataset) * HYPERPARAMETERS["train_split"])
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False)

# Initialize Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = MultiModalModel(img_size=HYPERPARAMETERS["img_size"],
                                 patch_size=16,
                                 num_classes=HYPERPARAMETERS["num_classes"]).to(device)
student_model = EfficientNetV2Student(num_classes=HYPERPARAMETERS["num_classes"]).to(device)

# Loss and Optimizer
criterion = KnowledgeDistillationLoss(alpha=HYPERPARAMETERS["alpha"],
                                       temperature=HYPERPARAMETERS["temperature"])
optimizer = optim.Adam(student_model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

# Training and Validation
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(HYPERPARAMETERS["num_epochs"]):
    student_model.train()
    teacher_model.eval()
    train_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(inputs)

        student_logits = student_model(inputs)
        loss = criterion(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct / total)

    student_model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            teacher_logits = teacher_model(inputs)
            student_logits = student_model(inputs)

            loss = criterion(student_logits, teacher_logits, labels)

            val_loss += loss.item()
            _, predicted = student_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(correct / total)

    print(f"Epoch [{epoch + 1}/{HYPERPARAMETERS['num_epochs']}] "
          f"Train Acc: {train_accuracies[-1]:.4f} "
          f"Val Acc: {val_accuracies[-1]:.4f}")

# Save Model
torch.save(student_model.state_dict(), "efficientnetv2_student.pth")

# Plot and Save Metrics
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.title('Training and Validation Metrics')
plt.savefig('metrics.png')
