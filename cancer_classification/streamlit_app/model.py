import torch
import torch.nn as nn
from torchvision.models import resnet18

class CombinedModel(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super(CombinedModel, self).__init__()

        # Rama de imágenes: ResNet18 preentrenado
        self.resnet = resnet18(pretrained=True)
        num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Eliminar la capa final para usar embeddings

        # Rama de características tabulares
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Capa combinada
        self.combined_branch = nn.Sequential(
            nn.Linear(num_resnet_features + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, features):
        image_features = self.resnet(image)
        tabular_features = self.tabular_branch(features)
        combined = torch.cat((image_features, tabular_features), dim=1)
        return self.combined_branch(combined)
    
