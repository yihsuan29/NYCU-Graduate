# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),          
        )
        
        self.shortcut = nn.Sequential()
        
        # to match the dimension
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
                    
    def forward(self, x):
        res = self.residual(x)
        cut = self.shortcut(x)
        output = self.relu(res+cut)
        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_num):
        super(EncoderBlock, self).__init__()
        self.blocks = [ResidualBlock(in_channels, out_channels,stride=2)]
        for _ in range(block_num-1):
            self.blocks.append(ResidualBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        output = self.blocks(x)
        return output, x
        
        
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock,self).__init__()
        
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),            
        )
    
    def forward(self, x, mid_prod):
        x = torch.cat([x, mid_prod], dim=1) 
        x = self.up_sample(x)        
        x = self.conv(x)
        return x


class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        # Input
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        # Encode
        self.encode1 = EncoderBlock(64, 64, 3)
        self.encode2 = EncoderBlock(64, 128, 4)
        self.encode3 = EncoderBlock(128, 256, 6)
        self.encode4 = EncoderBlock(256, 512, 3)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding = 'same') ,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        
        # Decoder
        self.decode4 = DecoderBlock(256+512, 32)
        self.decode3 = DecoderBlock(32+256, 32)
        self.decode2 = DecoderBlock(32+128, 32)
        self.decode1 = DecoderBlock(32+64, 32)
        
        # Output
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),                       
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),            
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        # Input
        x = self.input(x)
        
        # Encode
        e1, mid1 = self.encode1(x)
        e2, mid2 = self.encode2(e1)
        e3, mid3 = self.encode3(e2)
        e4, mid4 = self.encode4(e3)
        
        mid5 = e4
        
        # Bottleneck
        latent = self.bottleneck(e4)
        
        # Decode
        d4 = self.decode4(latent, mid5)
        d3 = self.decode3(d4, mid4)
        d2 = self.decode2(d3, mid3)
        d1 = self.decode1(d2, mid2)
        
        # Output
        output = self.out_conv(d1)
        
        return output
        
        