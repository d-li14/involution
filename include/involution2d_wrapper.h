#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/custom_function.h>

#include "autocast.h"
#include "involution2d_cpu.h"

#ifdef USE_CUDA
#   include "involution2d_cuda.cuh"
#endif

namespace involution {

at::Tensor involution2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation
) {
    static auto op = at::Dispatcher::singleton()
        .findSchemaOrThrow("involution::involution2d", "")
        .typed<decltype(involution2d)>();

    return op.call(input, weight, stride, padding, dilation);
}

at::Tensor involution2d_autocast(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
    return involution2d(autocast::_cast(at::kFloat, input), autocast::_cast(at::kFloat, weight), stride, padding, dilation)
        .to(input.scalar_type());
}

at::Tensor _involution2d_backward_grad_input(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::IntArrayRef& input_shape,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation
) {
    static auto op = at::Dispatcher::singleton()
        .findSchemaOrThrow("involution2d::_involution2d_backward_grad_input", "")
        .typed<decltype(_involution2d_backward_grad_input)>();

    return op.call(grad, weight, input_shape, stride, padding, dilation);
}

at::Tensor _involution2d_backward_grad_weight(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::IntArrayRef& weight_shape,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation
) {
    static auto op = at::Dispatcher::singleton()
        .findSchemaOrThrow("involution2d::_involution2d_backward_grad_weight", "")
        .typed<decltype(_involution2d_backward_grad_weight)>();

    return op.call(grad, input, weight_shape, stride, padding, dilation);
}

namespace cpu {

class Involution2dFunctionCPU : public torch::autograd::Function<Involution2dFunctionCPU>
{
    public:

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::Variable& input,
        const torch::autograd::Variable& weight,
        const at::IntArrayRef& kernel_size,
        const at::IntArrayRef& stride,
        const at::IntArrayRef& padding,
        const at::IntArrayRef& dilation,
        const int64_t groups
    ) {
        ctx->saved_data["kernel_size"] = kernel_size;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->save_for_backward({input, weight});

        auto output = involution2d_forward(input, weight, kernel_size, stride, padding, dilation, groups);

        return {output};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list grad_output
    ) {
        torch::autograd::variable_list saved = ctx->get_saved_variables();
        torch::autograd::Variable input = saved[0];
        torch::autograd::Variable weight = saved[1];

        auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        auto groups = ctx->saved_data["groups"].toInt();

        auto grads = involution2d_backward(grad_output[0], weight, input, kernel_size, stride, padding, dilation, groups);

        return {
            grads[0],
            grads[1],
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
        };
    }
};

at::Tensor involution2d_autograd(
    const torch::autograd::Variable& input,
    const torch::autograd::Variable& weight,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    return Involution2dFunctionCPU::apply(input, weight, kernel_size, stride, padding, dilation, groups)[0];
}

} // namespace cpu

#ifdef USE_CUDA

namespace cuda {

class Involution2dFunctionCUDA : public torch::autograd::Function<Involution2dFunctionCUDA>
{
    public:

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::Variable& input,
        const torch::autograd::Variable& weight,
        const at::IntArrayRef& kernel_size,
        const at::IntArrayRef& stride,
        const at::IntArrayRef& padding,
        const at::IntArrayRef& dilation,
        const int64_t groups
    ) {
        ctx->saved_data["kernel_size"] = kernel_size;
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["dilation"] = dilation;
        ctx->saved_data["groups"] = groups;
        ctx->save_for_backward({input, weight});

        auto output = involution2d_forward(input, weight, kernel_size, stride, padding, dilation, groups);

        return {output};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list grad_output
    ) {
        torch::autograd::variable_list saved = ctx->get_saved_variables();
        torch::autograd::Variable input = saved[0];
        torch::autograd::Variable weight = saved[1];

        auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        auto groups = ctx->saved_data["groups"].toInt();

        auto grads = involution2d_backward(grad_output[0], weight, input, kernel_size, stride, padding, dilation, groups);

        return {
            grads[0],
            grads[1],
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable(),
            torch::autograd::Variable()
        };
    }
};

at::Tensor involution2d_autograd(
    const torch::autograd::Variable& input,
    const torch::autograd::Variable& weight,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    return Involution2dFunctionCUDA::apply(input, weight, kernel_size, stride, padding, dilation, groups)[0];
}

} // namespace cuda

#endif

} // namespace involution
