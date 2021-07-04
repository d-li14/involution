#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace involution {
namespace cuda {

#define CUDA_MAX_THREADS 512u

#define CUDA_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

at::Tensor involution2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const int64_t groups
);

at::Tensor involution2d_backward_grad_input(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const int64_t groups
);

at::Tensor involution2d_backward_grad_weight(
    const at::Tensor& grad,
    const at::Tensor& input,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const int64_t groups
);

std::vector<at::Tensor> involution2d_backward(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    const int64_t groups
);

} // namespace cuda
} // namespace involution
