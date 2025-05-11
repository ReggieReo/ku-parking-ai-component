import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Paths (assuming you are in the parent directory of 'dataset')
TRAIN_IMG_DIR = 'dataset/images/train'
TRAIN_LABEL_DIR = 'dataset/labels/train'
VAL_IMG_DIR = 'dataset/images/val'
VAL_LABEL_DIR = 'dataset/labels/val'
DATA_YAML_PATH = 'dataset/data.yaml' # Important for training

# 1. Count of images and labels
num_train_images = len(glob(os.path.join(TRAIN_IMG_DIR, '*.jpg')))
num_train_labels = len(glob(os.path.join(TRAIN_LABEL_DIR, '*.txt')))
num_val_images = len(glob(os.path.join(VAL_IMG_DIR, '*.jpg')))
num_val_labels = len(glob(os.path.join(VAL_LABEL_DIR, '*.txt')))

print(f"Number of training images: {num_train_images}")
print(f"Number of training labels: {num_train_labels}")
print(f"Number of validation images: {num_val_images}")
print(f"Number of validation labels: {num_val_labels}")

if num_train_images != num_train_labels:
    print("Warning: Mismatch in training image and label counts!")
if num_val_images != num_val_labels:
    print("Warning: Mismatch in validation image and label counts!")

# 2. Class distribution (from label files)
class_counts = {0: 0} # Assuming 'car' is class 0
all_label_files = glob(os.path.join(TRAIN_LABEL_DIR, '*.txt')) + \
                  glob(os.path.join(VAL_LABEL_DIR, '*.txt'))

for label_file in all_label_files:
    try:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1
                else:
                    class_counts[class_id] = 1
    except Exception as e:
        print(f"Error reading {label_file}: {e}")


print("\nClass distribution:")
# Assuming names are defined in data.yaml or known
class_names_map = {0: 'car'} # From your data.yaml
for class_id, count in class_counts.items():
    print(f"- {class_names_map.get(class_id, f'Class_{class_id}')}: {count} instances")

# Plot class distribution
if class_counts:
    labels = [class_names_map.get(k, f'Class_{k}') for k in class_counts.keys()]
    counts = list(class_counts.values())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.show()
