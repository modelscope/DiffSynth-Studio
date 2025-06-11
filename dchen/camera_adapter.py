import torch
import torch.nn as nn

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapter, self).__init__()
        
        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        
        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim, kernel_size=kernel_size, stride=stride, padding=0)
        
        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)
        
        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)
        
        # Convolution operation
        x_conv = self.conv(x_unshuffled)
        
        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)
        
        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))
        
        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

# Example usage
# in_dim = 3
# out_dim = 64
# adapter = SimpleAdapterWithReshape(in_dim, out_dim)
# x = torch.randn(1, in_dim, 4, 64, 64)  # e.g., batch size = 1, channels = 3, frames/features = 4
# output = adapter(x)
# print(output.shape)  # Should reflect transformed dimensions
