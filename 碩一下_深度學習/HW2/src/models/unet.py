# Implement your UNet model here
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock,self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),            
        )
        
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2) 
    
    def forward(self, x):
        mid_prod = self.conv(x)
        output = self.down_sample(mid_prod)
        return output, mid_prod
        
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock,self).__init__()
        
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),            
        )
    
    def forward(self, x, mid_prod):
        x = self.up_sample(x)
        
        # Contracting path
        # Step 1. crop
        diff = mid_prod.size()[3] - x.size()[3]
        crop_diff = diff // 2
        mid_prod = mid_prod[:, :, crop_diff: x.size()[3]+crop_diff, crop_diff: x.size()[3]+crop_diff ]
        # Step 2. copy
        x = torch.cat([x, mid_prod], dim=1) 
        
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encode
        self.encode1 = EncoderBlock(in_channels, 64)
        self.encode2 = EncoderBlock(64, 128)
        self.encode3 = EncoderBlock(128, 256)
        self.encode4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, padding = 1) ,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1) ,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
        )
        
        # Decoder
        self.decode4 = DecoderBlock(1024, 512)
        self.decode3 = DecoderBlock(512, 256)
        self.decode2 = DecoderBlock(256, 128)
        self.decode1 = DecoderBlock(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size = 1) 
    
    def forward(self, x):
        # Encode
        e1, mid1 = self.encode1(x)
        e2, mid2 = self.encode2(e1)
        e3, mid3 = self.encode3(e2)
        e4, mid4 = self.encode4(e3)
        
        # Bottleneck
        latent = self.bottleneck(e4)
        
        # Decode
        d4 = self.decode4(latent, mid4)
        d3 = self.decode3(d4, mid3)
        d2 = self.decode2(d3, mid2)
        d1 = self.decode1(d2, mid1)
        
        # Output
        output = self.out_conv(d1)
        
        return output
        
        