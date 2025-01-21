import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidualDepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res = stride == 1 and in_channels == out_channels

        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        y = self.expansion(x)
        y = self.depthwise(y)
        y = self.project(y)
        y = self.se(y)
        if self.use_res:
            y = x + y
        return y
    
class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(channels, channels//r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels//r, channels, bias=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        max_out = self.linear(max_pool).view(b, c, 1, 1)
        avg_out = self.linear(avg_pool).view(b, c, 1, 1)
        
        channel_att = torch.sigmoid(max_out + avg_out)
        return channel_att * x

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))
        return spatial_att * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.cam = CAM(channels, reduction_ratio)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x


class LightweightCNN(nn.Module):

    def __init__(self):
        super(LightweightCNN, self).__init__()

        self.layer1 = nn.Sequential(
            InvertedResidualDepthwiseConv2d(3, 16),
            CBAM(16),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            InvertedResidualDepthwiseConv2d(16, 32),
            CBAM(32),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            InvertedResidualDepthwiseConv2d(32, 64),
            CBAM(64),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            InvertedResidualDepthwiseConv2d(64, 128),
            CBAM(128),
            nn.MaxPool2d(2, 2)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 100)
        )

    def forward(self, x, return_features=False):
        features = []
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        
        if return_features:
            return features
        
        x = self.global_avg_pool(x) 
        x = self.classifier(x)
        return x