from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch import ops

def _involution2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
    kernel_size_ = _pair(kernel_size)
    stride_ = _pair(stride)
    padding_ = _pair(padding)
    dilation_ = _pair(dilation)

    output: torch.Tensor = ops.involution.involution2d(input, weight, kernel_size_, stride_, padding_, dilation_, groups)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

class Involution2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 7,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 3,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 sigma_mapping: Optional[nn.Module] = None,
                 reduce_ratio: int = 1,
                 ) -> None:
        """2D Involution: https://arxiv.org/pdf/2103.06255.pdf
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (Union[int, Tuple[int, int]], optional): Kernel size to be used. Defaults to 7.
            stride (Union[int, Tuple[int, int]], optional): Stride factor to be utilized. Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): Padding to be used in unfold operation. Defaults to 3.
            dilation (Union[int, Tuple[int, int]], optional): Dilation in unfold to be employed. Defaults to 1.
            groups (int, optional): Number of groups to be employed. Defaults to 1.
            bias (bool, optional): If true bias is utilized in each convolution layer. Defaults to False.
            sigma_mapping (Optional[nn.Module], optional): Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
            reduce_ratio (int, optional): Reduce ration of involution channels. Defaults to 1.
        """
        super(Involution2d, self).__init__()

        assert isinstance(in_channels, int) and in_channels > 0, \
            '"in_channels" must be a positive integer.'
        assert isinstance(out_channels, int) and out_channels > 0, \
            '"out_channels" must be a positive integer.'
        assert isinstance(kernel_size, (int, tuple)), \
            '"kernel_size" must be an int or a tuple of ints.'
        assert isinstance(stride, (int, tuple)), \
            '"stride" must be an int or a tuple of ints.'
        assert isinstance(padding, (int, tuple)), \
            '"padding" must be an int or a tuple of ints.'
        assert isinstance(dilation, (int, tuple)), \
            '"dilation" must be an int or a tuple of ints.'
        assert isinstance(groups, int) and groups > 0, \
            '"groups" must be a positive integer.'
        assert in_channels % groups == 0, '"in_channels" must be divisible by "groups".'
        assert out_channels % groups == 0, '"out_channels" must be divisible by "groups".'
        assert isinstance(bias, bool), '"bias" must be a bool.'
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            '"sigma_mapping" muse be an int or a tuple of ints.'
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, \
            '"reduce_ratio" must be a positive integer.'

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: Tuple[int, int] = _pair(kernel_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)
        self.dilation: Tuple[int, int] = _pair(dilation)
        self.groups: int = groups
        self.bias: bool = bias
        self.reduce_ratio: int = reduce_ratio

        self.sigma_mapping = sigma_mapping if isinstance(sigma_mapping, nn.Module) else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels //
                           self.reduce_ratio, momentum=0.3),
            nn.ReLU()
        )
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=bias) \
            if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(
            kernel_size=self.stride) if self.stride[0] > 1 or self.stride[1] > 1 else nn.Identity()
        self.reduce_mapping = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.out_channels // self.reduce_ratio, kernel_size=1, bias=bias)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups, kernel_size=1, bias=bias)

    def __repr__(self) -> str:
        """Method returns information about the module
        Returns:
            str: Info string
        """
        return (f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel_size=({self.kernel_size[0]}, {self.kernel_size[1]}), '
            f'stride=({self.stride[0]}, {self.stride[1]}), padding=({self.padding[0]}, {self.padding[1]}), dilation=({self.dilation[0], self.dilation[1]}), '
            f'groups={self.groups}, bias={self.bias}, reduce_ratio={self.reduce_ratio}, sigma_mapping={str(self.sigma_mapping)}'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            input (torch.Tensor): Input tensor of the shape [batch size, in channels, height, width]
        Returns:
            torch.Tensor: Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        weight: torch.Tensor = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        input_init: torch.Tensor = self.initial_mapping(input)

        return _involution2d(input_init, weight, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
