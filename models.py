import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import faster_rcnn, retinanet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import timm
import warnings
warnings.filterwarnings("ignore")
torchvision_version = torchvision.__version__
print(f"Torchvision version: {torchvision_version}")
class YOLOLikeModel(nn.Module):
    def __init__(self, num_classes=5, input_size=640):
        super(YOLOLikeModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, (5 + num_classes) * 3, 1),
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        batch_size = detections.size(0)
        detections = detections.view(batch_size, 3, 5 + self.num_classes, detections.size(2), detections.size(3))
        detections = detections.permute(0, 3, 4, 1, 2).contiguous()
        return detections
class EfficientDetModel(nn.Module):
    def __init__(self, num_classes=5, input_size=640):
        super(EfficientDetModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.fpn = nn.ModuleList([
            nn.Conv2d(320, 256, 1),
            nn.Conv2d(112, 256, 1),
            nn.Conv2d(40, 256, 1),
        ])
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(320, 256, 1),
            nn.Conv2d(112, 256, 1),
            nn.Conv2d(40, 256, 1),
        ])
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
        ])
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
        ])
        self.reg_output = nn.Conv2d(256, 4, 1)
        self.cls_output = nn.Conv2d(256, num_classes, 1)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.backbone(x)
        p5 = self.fpn[0](features[-1])
        p4 = self.fpn[1](features[-2]) + F.interpolate(p5, size=features[-2].shape[-2:], mode='nearest')
        p3 = self.fpn[2](features[-3]) + F.interpolate(p4, size=features[-3].shape[-2:], mode='nearest')
        pyramid_features = [p3, p4, p5]
        reg_outputs = []
        cls_outputs = []
        for i, feat in enumerate(pyramid_features):
            reg_feat = self.reg_heads[i](feat)
            cls_feat = self.cls_heads[i](feat)
            reg_outputs.append(self.reg_output(reg_feat))
            cls_outputs.append(self.cls_output(cls_feat))
        return {
            'regression': reg_outputs,
            'classification': cls_outputs
        }
class CustomSSD(nn.Module):
    def __init__(self, num_classes=5, input_size=640):
        super(CustomSSD, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, 1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
        ])
        self.loc_heads = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, 3, padding=1),
            nn.Conv2d(1024, 6 * 4, 3, padding=1),
            nn.Conv2d(512, 6 * 4, 3, padding=1),
            nn.Conv2d(256, 6 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
            nn.Conv2d(256, 4 * 4, 3, padding=1),
        ])
        self.conf_heads = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, 3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, 3, padding=1),
            nn.Conv2d(512, 6 * num_classes, 3, padding=1),
            nn.Conv2d(256, 6 * num_classes, 3, padding=1),
            nn.Conv2d(256, 4 * num_classes, 3, padding=1),
            nn.Conv2d(256, 4 * num_classes, 3, padding=1),
        ])
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.backbone(x)
        extra_features = []
        out = features
        for layer in self.extra_layers:
            out = layer(out)
            extra_features.append(out)
        locs = []
        confs = []
        for i, (loc_head, conf_head) in enumerate(zip(self.loc_heads, self.conf_heads)):
            locs.append(loc_head(extra_features[i]))
            confs.append(conf_head(extra_features[i]))
        return locs, confs

def get_model(model_name, num_classes=5, **kwargs):
    if model_name == 'yolo':
        return YOLOLikeModel(num_classes=num_classes, **kwargs)
    elif model_name == 'efficientdet':
        return EfficientDetModel(num_classes=num_classes, **kwargs)
    elif model_name == 'ssd':
        return CustomSSD(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    
    models_to_test = ['yolo', 'efficientdet', 'ssd']
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name.upper()} model:")
        model = get_model(model_name)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output type: {type(output)}")
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} tensors")
                        for i, tensor in enumerate(value):
                            print(f"    {i}: {tensor.shape}")
                    else:
                        print(f"  {key}: {value.shape}")
            else:
                print(f"Output shape: {output.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}") 