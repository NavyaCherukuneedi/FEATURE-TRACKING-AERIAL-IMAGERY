
import torch
import torch.nn as nn
from torchvision import models

class EventImageFusionModel(nn.Module):
    def __init__(self, num_classes=10, event_feature_dim=128, image_feature_dim=512):
        super(EventImageFusionModel, self).__init__()

        # Image Feature Extractor: Pre-trained ResNet
        self.image_extractor = models.resnet18(pretrained=True)
        self.image_extractor.fc = nn.Linear(self.image_extractor.fc.in_features, image_feature_dim)

        # Event Feature Extractor
        self.event_encoder = nn.Sequential(
            nn.Linear(4, 64),  # Input: [x, y, p, t]
            nn.ReLU(),
            nn.Linear(64, event_feature_dim),
            nn.ReLU()
        )

        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(image_feature_dim + event_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, events):
        # Image features
        image_features = self.image_extractor(image)  # (B, image_feature_dim)

        # Event features
        B, N, _ = events.size()
        events = events.view(B * N, -1)  # Flatten events across the batch
        event_features = self.event_encoder(events)  # (B * N, event_feature_dim)
        event_features = event_features.view(B, N, -1).mean(dim=1)  # Average event features

        # Fuse features
        combined_features = torch.cat([image_features, event_features], dim=1)
        logits = self.fusion_layer(combined_features)

        return logits
