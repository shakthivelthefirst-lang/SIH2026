import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    """
    Complex 2D Convolution:
    (W_r + j W_i) * (X_r + j X_i) = (W_r * X_r - W_i * X_i) + j (W_r * X_i + W_i * X_r)
    Input shape: [B, 2, C, H, W] or [B, 2*C, H, W]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_channels))
        self.bias_i = nn.Parameter(torch.zeros(out_channels))

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xr: Real part [B, C, H, W]
            xi: Imaginary part [B, C, H, W]
        Returns:
            yr: Real part [B, C_out, H_out, W_out]
            yi: Imaginary part [B, C_out, H_out, W_out]
        """
        yr = self.conv_r(xr) - self.conv_i(xi) + self.bias_r.view(1, -1, 1, 1)
        yi = self.conv_r(xi) + self.conv_i(xr) + self.bias_i.view(1, -1, 1, 1)
        return yr, yi


class ComplexConvTranspose2d(nn.Module):
    """
    Complex 2D Transposed Convolution:
    (W_r + j W_i) * (X_r + j X_i) = (W_r * X_r - W_i * X_i) + j (W_r * X_i + W_i * X_r)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1), output_padding: tuple = (0, 0)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_trans_r = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False
        )
        self.conv_trans_i = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False
        )
        self.bias_r = nn.Parameter(torch.zeros(out_channels))
        self.bias_i = nn.Parameter(torch.zeros(out_channels))

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yr = self.conv_trans_r(xr) - self.conv_trans_i(xi) + self.bias_r.view(1, -1, 1, 1)
        yi = self.conv_trans_r(xi) + self.conv_trans_i(xr) + self.bias_i.view(1, -1, 1, 1)
        return yr, yi


class ComplexBatchNorm2d(nn.Module):
    """
    Complex Batch Normalization across Real and Imaginary components.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bn_r(xr), self.bn_i(xi)


class ComplexPReLU(nn.Module):
    """
    Complex PReLU activation.
    """
    def __init__(self):
        super().__init__()
        self.prelu_r = nn.PReLU()
        self.prelu_i = nn.PReLU()

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.prelu_r(xr), self.prelu_i(xi)


class ComplexBlock(nn.Module):
    """
    Complex Encoder Block: ComplexConv2d -> ComplexBatchNorm2d -> ComplexPReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1)):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.act = ComplexPReLU()

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yr, yi = self.conv(xr, xi)
        yr, yi = self.bn(yr, yi)
        return self.act(yr, yi)


class ComplexTransBlock(nn.Module):
    """
    Complex Decoder Block: ComplexConvTranspose2d -> ComplexBatchNorm2d -> ComplexPReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3), stride: tuple = (1, 2), padding: tuple = (1, 1), output_padding: tuple = (0, 0), act: bool = True):
        super().__init__()
        self.conv = ComplexConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.act_flag = act
        if act:
            self.bn = ComplexBatchNorm2d(out_channels)
            self.act = ComplexPReLU()

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yr, yi = self.conv(xr, xi)
        if self.act_flag:
            yr, yi = self.bn(yr, yi)
            yr, yi = self.act(yr, yi)
        return yr, yi
