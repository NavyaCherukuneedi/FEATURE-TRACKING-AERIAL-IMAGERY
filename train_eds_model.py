
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import h5py
from PIL import Image
import pandas as pd
import numpy as np
from eds_fusion_model import EDSFusionModel  # Assuming model is saved in the same directory

# Dataset for EDS data
class EDSDataset(Dataset):
    def __init__(self, images_dir, events_file, poses_file, transform=None):
        self.images_dir = images_dir
        self.events_file = h5py.File(events_file, 'r')
        self.poses_data = pd.read_csv(poses_file)
        self.transform = transform

        # Extract event keys
        self.event_x = self.events_file['x'][:]
        self.event_y = self.events_file['y'][:]
        self.event_p = self.events_file['p'][:]
        self.event_t = self.events_file['t'][:]

    def __len__(self):
        return len(self.poses_data)

    def __getitem__(self, idx):
        # Load image
        image_path = f"{self.images_dir}/{self.poses_data.iloc[idx, 0]}"
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get pose
        pose = self.poses_data.iloc[idx, 1:].values.astype(np.float32)

        # Get corresponding events (example: slice events based on some criteria)
        events = np.stack([
            self.event_x[:1000], self.event_y[:1000],
            self.event_p[:1000], self.event_t[:1000]
        ], axis=-1).astype(np.float32)  # Example: sample 1000 events

        return image, events, pose


# Loss Function for Pose Regression
class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Positional loss (X, Y, Z)
        position_loss = self.mse(pred[:, :3], target[:, :3])
        # Quaternion loss (qx, qy, qz, qw)
        quaternion_loss = self.mse(pred[:, 3:], target[:, 3:])
        return position_loss + quaternion_loss


# Training function
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = PoseLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, events, poses in train_loader:
            images, events, poses = images.to(device), events.to(device), poses.to(device)

            optimizer.zero_grad()
            outputs = model(images, events)
            loss = criterion(outputs, poses)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, events, poses in val_loader:
                images, events, poses = images.to(device), events.to(device), poses.to(device)
                outputs = model(images, events)
                loss = criterion(outputs, poses)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training complete!")


# Main script
if __name__ == "__main__":
    # Paths to dataset components
    images_dir = "path/to/images"
    events_file = "path/to/events.h5"
    poses_file = "path/to/camera_poses.csv"

    # Transformations for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and data loaders
    train_dataset = EDSDataset(images_dir, events_file, poses_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Instantiate model
    model = EDSFusionModel()

    # Train the model
    train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu')
