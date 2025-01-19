import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

# Teacher Model: MultiModal Model with ViT and VGG+CAM
class ManualViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=6, num_classes=10):
        super(ManualViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Step 1: Patch Embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Step 2: Add Class Token and Positional Embedding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed  # Add positional embedding
        x = self.dropout(x)

        # Step 3: Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        # Step 4: Classification using the CLS token
        cls_token_final = x[:, 0]  # [B, embed_dim]
        return cls_token_final

class VGG16CAM(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG16CAM, self).__init__()
        vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=pretrained)
        self.features = vgg16.features
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classifier = nn.Linear(512, num_classes)  # Classification Layer

    def forward(self, x):
        feature_maps = self.features(x)  # [B, 512, H, W]
        global_features = self.gap(feature_maps).view(feature_maps.size(0), -1)  # [B, 512]
        return global_features, feature_maps

class MultiModalFusion(nn.Module):
    def __init__(self, vit_dim, vgg_dim, num_classes):
        super(MultiModalFusion, self).__init__()
        self.fc_vit = nn.Linear(vit_dim, 512)
        self.fc_vgg = nn.Linear(vgg_dim, 512)
        self.fusion_fc = nn.Linear(512 * 2, num_classes)

    def forward(self, vit_features, vgg_features):
        vit_out = self.fc_vit(vit_features)
        vgg_out = self.fc_vgg(vgg_features)
        fused_features = torch.cat([vit_out, vgg_out], dim=1)
        logits = self.fusion_fc(fused_features)
        return logits

class MultiModalModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=2):
        super(MultiModalModel, self).__init__()
        self.vit = ManualViT(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
        self.vgg_cam = VGG16CAM(pretrained=True)
        self.fusion = MultiModalFusion(vit_dim=768, vgg_dim=512, num_classes=num_classes)

    def forward(self, x):
        # ViT Features
        vit_features = self.vit(x)
        # VGG Features
        vgg_features, _ = self.vgg_cam(x)
        # Fusion
        logits = self.fusion(vit_features, vgg_features)
        return logits

# Student Model: EfficientNetV2
class EfficientNetV2Student(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetV2Student, self).__init__()
        self.model = efficientnet_v2_s(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Knowledge Distillation Loss
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_labels = nn.functional.log_softmax(teacher_logits / self.temperature, dim=1).clamp(min=-1e2, max=1e2)
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1).clamp(min=-1e2, max=1e2)

        # 检查 logits 是否存在异常
        if torch.isnan(soft_labels).any() or torch.isnan(soft_student).any():
            raise ValueError("NaN detected in logits")

        distillation_loss = self.kl_div(soft_student, soft_labels) * (self.temperature ** 2)
        ce_loss = self.ce_loss(student_logits, labels)
        return self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
