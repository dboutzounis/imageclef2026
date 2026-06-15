import torch
import torch.nn as nn
import torchvision.models as models

FEATURE_DIMS = {
    "efficientnetb0"  : 1280,
    "efficientnetv2s" : 1280,
    "efficientnetv2m" : 1280,
    "densenet121"     : 1024,
    "convnexttiny"    :  768,
    "resnet50"        : 2048
}

def get_backbone(name):
    """Matches Colab extraction logic perfectly."""
    clean_name = name.lower().replace('_', '')
    
    if clean_name == "efficientnetb0":
        return models.efficientnet_b0().features
    elif clean_name == "efficientnetv2s":
        return models.efficientnet_v2_s().features
    elif clean_name == "efficientnetv2m":
        return models.efficientnet_v2_m().features
    elif clean_name == "densenet121":
        return models.densenet121().features
    elif clean_name == "convnexttiny":
        return models.convnext_tiny().features
    elif clean_name == "resnet50":
        backbone = models.resnet50()
        return nn.Sequential(*list(backbone.children())[:-2])
    else:
        raise ValueError(f"Unsupported backbone: {name}")

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, p_trainable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if p_trainable else p
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

    def __repr__(self):
        p_val = round(self.p.item(), 4) if isinstance(self.p, nn.Parameter) else self.p
        return f"GeM(p={p_val}, eps={self.eps})"

class ConceptDetectionModel(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout=0.0, pool_type="avg", gem_p=3.0):
        super().__init__()
        
        clean_name = backbone_name.lower().replace('_', '')
        self.backbone_name = clean_name
        
        self.backbone = get_backbone(clean_name)
        num_features = FEATURE_DIMS.get(clean_name, 1280)

        if pool_type == "gem":
            self.pool = GeM(p=gem_p, p_trainable=True)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)
    
class ConceptCountRegressor(nn.Module):
    def __init__(self, backbone_name, dropout=0.2, pool_type="gem", gem_p=3.0):
        super().__init__()
        
        clean_name = backbone_name.lower().replace('_', '')
        self.backbone_name = clean_name
        self.backbone = get_backbone(clean_name)
        num_features = FEATURE_DIMS.get(clean_name, 1280)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if pool_type == "gem":
            self.pool = GeM(p=gem_p, p_trainable=True)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.regressor(x).squeeze(-1)