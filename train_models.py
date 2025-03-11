# train_models.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from models.vit_model import ViTForBrainTumorDetection
from models.segmentation_model import TumorSegmentationModel


class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, labels, masks=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        if self.masks is not None:
            mask = Image.open(self.masks[idx]).convert('L')
            mask_tensor = transforms.ToTensor()(mask)
            return image_tensor, label, mask_tensor

        return image_tensor, label


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Resize mask to match image
        mask = mask.resize((224, 224), Image.NEAREST)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Convert mask to tensor (0 or 1)
        mask_tensor = transforms.ToTensor()(mask)
        # Threshold to ensure binary mask
        mask_tensor = (mask_tensor > 0.5).float()

        return image, mask_tensor


def train_detection_model(dataset_path, model_save_path, batch_size=32, num_epochs=20):
    # Set up data paths
    image_paths = []
    labels = []

    # Load positive examples (with tumor)
    positive_dir = os.path.join(dataset_path, 'positive')
    for img in os.listdir(positive_dir):
        if img.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(positive_dir, img))
            labels.append(1)  # 1 for positive (tumor)

    # Load negative examples (without tumor)
    negative_dir = os.path.join(dataset_path, 'negative')
    for img in os.listdir(negative_dir):
        if img.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(os.path.join(negative_dir, img))
            labels.append(0)  # 0 for negative (no tumor)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transform)
    val_dataset = BrainTumorDataset(X_val, y_val, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForBrainTumorDetection()
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs.logits, labels)

                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation accuracy: {val_acc:.4f}')


def train_segmentation_model(dataset_path, model_save_path, batch_size=16, num_epochs=30):
    # Set up data paths
    image_paths = []
    mask_paths = []

    # Load images and corresponding masks
    images_dir = os.path.join(dataset_path, 'images')
    masks_dir = os.path.join(dataset_path, 'masks')

    for img in os.listdir(images_dir):
        if img.endswith(('.jpg', '.png', '.jpeg')):
            # Ensure corresponding mask exists
            mask_filename = img.split('.')[0] + '.png'  # Assume masks are .png
            if os.path.exists(os.path.join(masks_dir, mask_filename)):
                image_paths.append(os.path.join(images_dir, img))
                mask_paths.append(os.path.join(masks_dir, mask_filename))

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = SegmentationDataset(X_train, y_train, transform=train_transform)
    val_dataset = SegmentationDataset(X_val, y_val, transform=val_transform)

    # Create data loaders with num_workers=0 to avoid pickling errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TumorSegmentationModel()
    model.to(device)

    # Loss function and optimizer
    # Using Dice Loss for segmentation
    def dice_loss(pred, target):
        smooth = 1.0
        pred = torch.sigmoid(pred)
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)

                # Statistics
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation loss: {val_loss:.4f}')


if __name__ == '__main__':
    # Train detection model
    print("Starting training for tumor detection model...")
    train_detection_model(
        dataset_path='./data/detection',
        model_save_path='./models/vit_brain_tumor_detection.pth',
        batch_size=16,
        num_epochs=20
    )

    # Train segmentation model
    print("\nStarting training for tumor segmentation model...")
    train_segmentation_model(
        dataset_path='./data/segmentation',
        model_save_path='./models/tumor_segmentation.pth',
        batch_size=8,
        num_epochs=30
    )

    print("Training completed!")
