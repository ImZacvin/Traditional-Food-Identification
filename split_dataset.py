import os
import numpy as np
from sklearn.model_selection import train_test_split

# Specify the path to your dataset folder
dataset_path = 'Dataset'

# Initialize lists for file paths and labels
image_paths = []
labels = []

# Load the dataset (this example assumes a nested folder structure)
for province in os.listdir(dataset_path):
    province_path = os.path.join(dataset_path, province)
    if os.path.isdir(province_path):
        for category in os.listdir(province_path):
            category_path = os.path.join(province_path, category)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_file)
                    image_paths.append(img_path)
                    labels.append(province)  # Use the province as the label

# Convert to numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Output the split sizes
print(f'Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')

# Save the paths to text files for later use
np.savetxt('Text/train_paths.txt', X_train, fmt='%s')
np.savetxt('Text/train_labels.txt', y_train, fmt='%s')
np.savetxt('Text/val_paths.txt', X_val, fmt='%s')
np.savetxt('Text/val_labels.txt', y_val, fmt='%s')
np.savetxt('Text/test_paths.txt', X_test, fmt='%s')
np.savetxt('Text/test_labels.txt', y_test, fmt='%s')
