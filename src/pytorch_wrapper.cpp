#include <torch/script.h>
#include "involution2d_wrapper.h"

TORCH_LIBRARY(involution, m) {
    m.def("involution2d(Tensor input, Tensor weight, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
    m.def("_involution2d_backward_grad_input(Tensor grad, Tensor weight, int[] input_shape, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
    m.def("_involution2d_backward_grad_weight(Tensor grad, Tensor input, int[] weight_shape, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
    m.def("_involution2d_backward(Tensor grad, Tensor weight, Tensor input, int[] kernel_size, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(involution, CPU, m) {
    m.impl("involution2d", involution::cpu::involution2d_forward);
    m.impl("_involution2d_backward_grad_input", involution::cpu::involution2d_backward_grad_input);
    m.impl("_involution2d_backward_grad_weight", involution::cpu::involution2d_backward_grad_weight);
    m.impl("_involution2d_backward", involution::cpu::involution2d_backward);
}

TORCH_LIBRARY_IMPL(involution, CUDA, m) {
    m.impl("involution2d", involution::cuda::involution2d_forward);
    m.impl("_involution2d_backward_grad_input", involution::cuda::involution2d_backward_grad_input);
    m.impl("_involution2d_backward_grad_weight", involution::cuda::involution2d_backward_grad_weight);
    m.impl("_involution2d_backward", involution::cuda::involution2d_backward);
}

// TORCH_LIBRARY_IMPL(involution, Autocast, m) {
//     m.impl("involution2d", involution2d_autocast);
// }

TORCH_LIBRARY_IMPL(involution, AutogradCPU, m) {
    m.impl("involution2d", involution::cpu::involution2d_autograd);
}

TORCH_LIBRARY_IMPL(involution, AutogradCUDA, m) {
    m.impl("involution2d", involution::cuda::involution2d_autograd);
}
