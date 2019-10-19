import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super().__init__()
        
        self.stride = stride
        between_channels = out_channels * expansion_factor        
        
        self.bn_layer_1x1_before = ConvBnRelu(in_channels, between_channels, kernel_size=1, stride=1)

        self.bn_layer_3x3 = ConvBnRelu(between_channels, between_channels, 
                                       kernel_size=3, stride=stride, padding=1, groups=between_channels)

        self.bn_layer_1x1_after = ConvBn(between_channels, out_channels)

        self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, x):
        h = self.bn_layer_1x1_before(x)
        h = self.bn_layer_3x3(h)
        h = self.bn_layer_1x1_after(h)

        if self.stride == 1:
            return self.skip_add.add(h, x)
        else:
            return h

class MobileNetv2(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        # (expansion_factor, out_channels, num_blocks, stride)
        self.cfg = [
               (6, 32, 2, 2),
               (6, 64, 2, 2),
               (6, 128, 2, 2),
        ]

        self.features = self.make_layers()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(256, num_classes)

    def make_layers(self, in_channels=32):
        layers = [ConvBnRelu(1, in_channels, stride=1, padding=1)]

        for expension_factor, out_channels, num_block, stride in self.cfg:
            strides = [stride] + [1] * (num_block - 1)

            for s in strides:
                layers.append(Block(in_channels, out_channels, stride=s, expansion_factor=expension_factor))
                in_channels = out_channels
        
        layers.append(ConvBnRelu(128, 256, stride=1, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(2).squeeze(2)
        x = self.dropout(x)
        x = self.linear(x)
        return x
