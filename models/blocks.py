import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    def __init__(self, inchannels, outchannels,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, bias= False, **kwargs),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, repeats =1):
        super(ResidualBlock,self).__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        for i in range(repeats):
            self.layers.append(nn.Sequential(
                ConvolutionBlock(channels,channels//2, kernel_size=1),
                ConvolutionBlock(channels//2, channels, kernel_size=3, padding=1),
            ))

    def forward(self, x):
        #forward -> pass
        for layer in self.layers:
            if self.use_residual:
                x=x+layer(x)
            else:
                x=layer(x)
        return x