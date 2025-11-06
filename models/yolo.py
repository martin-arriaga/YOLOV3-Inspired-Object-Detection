
from models.darknet import *
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(in_channels, 3 * (num_classes + 5), kernel_size=1)  # 3 anchors

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        H, W = x.shape[2], x.shape[3]
        x = x.view(B, 3, self.num_classes + 5, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # [B, anchors, H, W, 5+C]
        return x

class YOLOv3(nn.Module):
    def __init__(self, num_classes=20):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.backbone = Darknet(config)

        self.scale1 = ScalePrediction(1024, num_classes)
        self.scale2 = ScalePrediction(1024, num_classes)
        self.scale3 = ScalePrediction(512, num_classes)

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv_upsample1 = ConvolutionBlock(1024, 512, kernel_size=1)
        self.conv_upsample2 = ConvolutionBlock(1024, 256, kernel_size=1)

    def forward(self, x):
        small, medium, large = self.backbone(x)  # get 3 scales
        out_large = self.scale1(large)  # 13x13

        x = self.conv_upsample1(large)
        x = self.upsample(x)
        x = torch.cat((x, medium), dim=1)
        out_medium = self.scale2(x)  # 26x26

        x = self.conv_upsample2(x)
        x = self.upsample(x)
        x = torch.cat((x, small), dim=1)
        out_small = self.scale3(x)  # 52x52

        return [out_large, out_medium, out_small]

