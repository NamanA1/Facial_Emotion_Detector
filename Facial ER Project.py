import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalization for grayscale or single channel
])

# Dataset paths
data_dir = r"C:\Users\aryan\fer2013"
train_dir = data_dir + r"\train"
test_dir = data_dir + r"\test"

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load pre-trained ResNet18
resnet18 = models.resnet18(pretrained=True)

# Change final layer to match 7 emotion classes
num_classes = 7  # FER-2013 has 7 emotions
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=0.001)

print(f"Training images: {len(train_dataset)}")
print(f"Classes: {train_dataset.classes}")
print("Starting training...")


# Training loop
num_epochs = 5


for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1} - Loss: {running_loss:.3f} - Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(resnet18.state_dict(), "emotion_model.pth")
print("✅ Model saved as emotion_model.pth")
