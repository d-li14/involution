#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

namespace involution {
namespace cpu {

at::Tensor involution2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
);

at::Tensor involution2d_backward_grad_input(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::IntArrayRef& input_shape,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
);

at::Tensor involution2d_backward_grad_weight(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::IntArrayRef& weight_shape,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
);

std::vector<at::Tensor> involution2d_backward(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& input,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
);

} // namespace cpu
} // namespace involution
