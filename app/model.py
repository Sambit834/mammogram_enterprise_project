import torch
import torch.nn as nn
from torchvision import models

class MammogramClassifier(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=False, freeze_backbone=False):
        super(MammogramClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.is_vit = False
            
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.is_vit = False
            
        elif model_name == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.is_vit = False
            
        elif model_name == 'vit_b_16':
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
            self.is_vit = True
            
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
