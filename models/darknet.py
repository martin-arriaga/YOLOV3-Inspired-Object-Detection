
from models.blocks import *
config= [
    (32, 3, 1), #---416x416x32--- " 32 3x3 filters applied at stride 1" 416 + 2 - 3 / 1 + 1 =
    (64, 3, 2), #---208x208x64---
    ["R", 1],#---208x208x64---
    (128, 3, 2), #---104x104x128---
    ["R", 2],#---104x104x128---
    (256, 3, 2),#---52x52x256---
    ["R", 8],#---52x52x256---
    (512, 3, 2),#---26x26x512---
    ["R", 8],#---26x26x512---
    (1024, 3, 2),#---13x13x1024---
    ["R", 4],#---13x13x1024---
]


class Darknet(nn.Module):
    def __init__(self,config):
        super(Darknet, self).__init__()
        self.layers = self.createConvLayers(config)

    def createConvLayers(self,config):
        layers = nn.ModuleList()
        inchannels = 3

        for module in config:
            if isinstance(module, tuple):
                outchannels, ksize ,stride = module

                layers.append(ConvolutionBlock(inchannels,outchannels,kernel_size=ksize,stride=stride, padding=ksize//2))
                inchannels = outchannels
            elif isinstance(module, list):
                if module[0] == 'R':
                    repeats = module[1]
                    layers.append(ResidualBlock(inchannels,repeats=repeats))
        return layers

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        route1 = outputs[6]  # 52x52
        route2 = outputs[8]  # 26x26
        route3 = outputs[10]  # 13x13

        return route1, route2, route3



"""
-------------------
Memory & Parameters 
--------------------
input image 416x416x3
416x416x3 = 500k "number of values in the input tensor" no learnable parameters
convolutional layer 1 output [416x416x32]
    ------416-------
    |               |
    |               |
    |               |
    416             |    x 32, "32 416x416 filters stacked each one highlights some feature in the image 
    |               |
    |               |
    |               |
    |----------------
    
    Memory: 416 x 416 x 32 = 5537792 ~ 5.53M values in a tensor 
    Params: K_w x K_h x C_in x C_out = 3 x 3 x 3 x 32 = 864 learnable weights 
    K_w = kernal width 
    K_h = kernal height 
    C_in : number of input channels
    C_out : number of output channels
Convolutional layer 2 output [208x208x64]
     ------208-------
    |               |
    |               |
    |               |
    208             |    x 64, "64 208x208 filters stacked each one highlights some feature in the image 
    |               |
    |               |
    |               |
    |----------------
    
    memory = 208 x 208 x 64 = 276k
    params: 3 x 3 x 32 x 64 = 18432 learnable weights  
    output = {416 + 2 - 3 / 2 ] + 1 = 208.5 ~ 208 "volume"
    final tensor shape = 208x208x64 

    
-------------------------------
Output of a Convolutional layer
--------------------------------
Convolutional layer 1 output [416x416x32]
out = [H_in + 2P - K / S ]+1
K = kernal size of conv kernel
S = stride of conv kernel
"""