import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define a dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Load image
        label = self.labels[idx]  # Get corresponding label

        if self.transform:
            image = self.transform(image)  # Apply transformations

        # Convert label to tensor (ensure it is an integer tensor)
        label = torch.tensor(int(label))  # Convert label to tensor

        return image, label

# Define your model class
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        # Load a pre-trained EfficientNet model
        self.model = models.efficientnet_b0(pretrained=True)
        # Replace the final layer to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load image paths and labels from text files
def load_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load the paths and labels
train_image_paths = load_file('Text/train_paths.txt')
train_labels = load_file('Text/train_labels.txt')
val_image_paths = load_file('Text/val_paths.txt')
val_labels = load_file('Text/val_labels.txt')
test_image_paths = load_file('Text/test_paths.txt')
test_labels = load_file('Text/test_labels.txt')

# Convert labels to integers based on the unique labels
label_map = {label: idx for idx, label in enumerate(np.unique(train_labels))}
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]
test_labels = [label_map[label] for label in test_labels]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Create datasets
train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform)
val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform)
test_dataset = CustomImageDataset(test_image_paths, test_labels, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
num_classes = len(np.unique(train_labels))  # Number of unique classes
model = YourModel(num_classes=num_classes)  # Use YourModel class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up the training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10  # Adjust based on your needs

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available
        optimizer.zero_grad()  # Zero gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Define the folder path where you want to save the model
folder_path = 'Trained Model'
os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

# Define the full file path
file_path = os.path.join(folder_path, 'model.pth')

# Save the model state dictionary to the specified path
torch.save(model.state_dict(), file_path)
print(f"Model saved to {file_path}")
