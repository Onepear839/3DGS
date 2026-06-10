#
# 改进: VGG感知损失 (Perceptual Loss)
# 基于Johnson et al. "Perceptual Losses for Real-Time Style Transfer" (2016)
# 使用预训练VGG16网络提取特征，计算特征空间的L1距离
# 使3DGS渲染结果在感知质量上更接近真实图像
#

import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(nn.Module):
    \"\"\"VGG感知损失模块
    
    使用预训练VGG16的浅层(relu1_2, relu2_2, relu3_3)特征计算感知距离。
    这些层编码了图像的边缘、纹理和形状信息。
    \"\"\"
    
    def __init__(self, device="cuda"):
        super().__init__()
        # 加载预训练VGG16，仅使用前16层（到conv3_3）
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:16].to(device).eval()
        
        # 冻结VGG参数，不参与训练
        for p in self.features.parameters():
            p.requires_grad = False
            
        # 定义特征提取层名称和各层权重
        self.layers = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
        }
        self.layer_weights = {
            'relu1_2': 1.0 / 2.6,
            'relu2_2': 1.0 / 4.8,
            'relu3_3': 1.0 / 3.7,
        }
        
        # 预处理参数（ImageNet标准化）
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
    def forward(self, pred, target):
        \"\"\"计算感知损失\"\"\"
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        # ImageNet标准化
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # 提取特征并计算各层损失
        pred_features = pred_norm
        target_features = target_norm
        total_loss = 0.0
        
        for layer_name, layer_idx in self.layers.items():
            start_idx = 0 if layer_name == 'relu1_2' else (4 if layer_name == 'relu2_2' else 9)
            for i in range(start_idx, layer_idx + 1):
                pred_features = self.features[i](pred_features)
                target_features = self.features[i](target_features)
            
            layer_loss = nn.functional.l1_loss(pred_features, target_features)
            total_loss += self.layer_weights[layer_name] * layer_loss
            
        return total_loss
