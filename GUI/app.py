import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# Define the model class (same as in your training script)
class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        self.model = models.efficientnet_b0(weights='DEFAULT')  # Updated for weights
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load class labels from the dataset directory
def load_labels(dataset_path):
    label_map = {}
    class_index = 0
    for province in os.listdir(dataset_path):
        province_path = os.path.join(dataset_path, province)
        if os.path.isdir(province_path):
            for category in os.listdir(province_path):
                category_path = os.path.join(province_path, category)
                if os.path.isdir(category_path):
                    label_map[f"{province}/{category}"] = class_index
                    class_index += 1
    return label_map

# Load your model
dataset_path = 'Dataset'  # Update this path to your dataset
label_map = load_labels(dataset_path)
num_classes = 10  # Set to the actual number of classes used during training
model = YourModel(num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to predict the image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = [k for k, v in label_map.items() if v == predicted.item()][0]
        return predicted_label

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        label_text = predict_image(file_path)
        messagebox.showinfo("Prediction", f'Predicted Class: {label_text}')

# Create the main window
root = tk.Tk()
root.title("Image Classification")

# Create a panel for image display
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Create an upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
