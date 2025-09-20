import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(kernel_size, sigma,channels):
    """Create a 3D Gaussian kernel."""
    depth, height, width = kernel_size
    dz, dx, dy = (depth - 1) // 2, (height - 1) // 2, (width - 1) // 2
    z = torch.arange(depth) - depth // 2
    x = torch.arange(height) - height // 2
    y = torch.arange(height) - height // 2
    z_grid,x_grid, y_grid = torch.meshgrid(z,x, y,indexing="ij")
    kernel = torch.exp(-(x_grid**2 + y_grid**2+z_grid**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.unsqueeze(0)
    kernel = kernel.unsqueeze(0)
    kernel = kernel.expand(channels,1,depth, height, width)
    return kernel
def gaussian_pooling(input_tensor, kernel_size=(3,9,9), sigma=1.0):
    channels = input_tensor.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma,channels).to(input_tensor.device)
    #padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
    blurred =F.conv3d(input_tensor, kernel, stride=1, padding='same',groups=channels)
    return blurred