# Involution: Inverting the Inherence of Convolution for Visual Recognition

Unofficial PyTorch reimplemention of the paper [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf) by Duo Li, Jie Hu, Changhu Wang et al. published at CVPR 2021.

**This repository includes a PyTorch implementation of 2D Involution using C++/OpenMP/CUDA.**

## Installation

- CUDA

    ```bash
    pip install git+https://github.com/shikishima-TasakiLab/Involution-PyTorch
    ```

- CPU only

    ```bash
    USE_CUDA=0 pip install git+https://github.com/shikishima-TasakiLab/Involution-PyTorch
    ```

## Example Usage

The 2D involution can be used as a `nn.Module` as follows:

```python
import torch
import torch.nn as nn
from involution import Involution2d

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

inv2d: nn.Module = Involution2d(in_channels=4, out_channels=8).to(device)

x: torch.Tensor = torch.rand(2, 4, 8, 8).to(device)

y: torch.Tensor = inv2d(x)
```

The 2D involution takes the following parameters:

|Parameter      |Description                                                                    |Type               |Default|
|---------------|-------------------------------------------------------------------------------|-------------------|-------|
|`in_channels`  |Number of input channels.                                                      |`int`              |   -   |
|`out_channels` |Number of output channels.                                                     |`int`              |   -   |
|`kernel_size`  |Kernel size to be used.                                                        |`int`, `(int, int)`|`7`    |
|`stride`       |Stride factor to be utilized.                                                  |`int`, `(int, int)`|`1`    |
|`padding`      |Padding to be used in unfold operation.                                        |`int`, `(int, int)`|`3`    |
|`dilation`     |Dilation in unfold to be employed.                                             |`int`, `(int, int)`|`1`    |
|`groups`       |Number of groups to be employed.                                               |`int`              |`1`    |
|`bias`         |If true bias is utilized in each convolution layer.                            |`bool`             |`False`|
|`sigma_mapping`|Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized.  |`torch.nn.Module`  |`None` |
|`reduce_ratio` |Reduce ration of involution channels.                                          |`int`              |`1`    |

## Reference

```bibtex
@inproceedings{Li2021,
    author = {Li, Duo and Hu, Jie and Wang, Changhu and Li, Xiangtai and She, Qi and Zhu, Lei and Zhang, Tong and Chen, Qifeng},
    title = {Involution: Inverting the Inherence of Convolution for Visual Recognition},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```
