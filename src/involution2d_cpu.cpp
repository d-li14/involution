#include "involution2d_cpu.h"

namespace involution {
namespace cpu {

template <typename scalar_t>
static void involution2d_forward_frame(
    const at::Tensor& in_data,
    const at::Tensor& weight_data,
    at::Tensor& out_data,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& dilation
) {
    auto num_elements = out_data.numel();
    const auto groups = weight_data.size(1);
    const auto channels = in_data.size(1);
    const auto in_height = in_data.size(2);
    const auto in_width = in_data.size(3);
    const auto out_height = out_data.size(2);
    const auto out_width = out_data.size(3);

    auto in_data_a = in_data.accessor<scalar_t, 4>();
    auto weight_data_a = weight_data.accessor<scalar_t, 6>();
    auto* out_data_p = out_data.data_ptr<scalar_t>();

    #pragma omp parallel for private(idx)
    for (int64_t idx = 0l; idx < num_elements; idx++) {
        const int64_t w = idx % out_width;
        const int64_t h = (idx / out_width) % out_height;
        int64_t divisor = out_width * out_height;
        const int64_t c = (idx / divisor) % channels;
        divisor *= channels;
        const int64_t n = idx / divisor;
        const int64_t g = c / (channels / groups);

        scalar_t value = 0;

        for (int64_t kh = 0l; kh < kernel_size[0]; kh++) {
            const int64_t h_in = h * stride[0] + kh * dilation[0] - padding[0];

            if ((0l <= h_in) && (h_in < in_height)) {
                for (int64_t kw = 0l; kw < kernel_size[1]; kw++) {
                    const int64_t w_in = w * stride[1] + kw * dilation[1] - padding[1];

                    if ((0l <= w_in) && (w_in < in_width)) {
                        value += weight_data_a[n][g][kh][kw][h][w] * in_data_a[n][c][h_in][w_in];
                    }
                }
            }
        }
        out_data_p[idx] = value;
    }
}

at::Tensor involution2d_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    AT_ASSERTM(input.device().is_cpu(), "\"input\" must be a CPU tensor.");
    AT_ASSERTM(weight.device().is_cpu(), "\"weight\" must be a CPU tensor.");

    at::TensorArg input_t{input, "input", 1}, weight_t{weight, "weight", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameType(c, {input_t, weight_t});

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto weight_height = weight.size(2);
    const auto weight_width = weight.size(3);

    const at::Tensor weight_ = weight.view({batch_size, groups, kernel_size[0], kernel_size[1], weight_height, weight_width});

    const auto out_height = (in_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    const auto out_width = (in_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;

    at::Tensor output = at::zeros({batch_size, channels, out_height, out_width}, input.options());

    if (output.numel() == 0) {
        return output;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "involution2d_forward_kernel", [&] {
            involution2d_forward_frame<scalar_t>(
                input,
                weight_,
                output,
                kernel_size,
                padding,
                stride,
                dilation
            );
        }
    );
    return output;
}

template <typename scalar_t>
static void involution2d_backward_grad_input_frame(
    const at::Tensor& out_diff,
    const at::Tensor& weight_data,
    at::Tensor& in_diff,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& dilation
) {
    auto num_elements = in_diff.numel();
    const auto groups = weight_data.size(1);
    const auto channels = in_diff.size(1);
    const auto in_height = in_diff.size(2);
    const auto in_width = in_diff.size(3);
    const auto out_height = out_diff.size(2);
    const auto out_width = out_diff.size(3);

    auto out_diff_a = out_diff.accessor<scalar_t, 4>();
    auto weight_data_a = weight_data.accessor<scalar_t, 6>();
    auto* in_diff_p = in_diff.data_ptr<scalar_t>();

    #pragma omp parallel for private(idx)
    for (int64_t idx = 0l; idx < num_elements; idx++) {
        const int64_t w = idx % in_width;
        const int64_t h = (idx / in_width) % in_height;
        int64_t divisor = in_width * in_height;
        const int64_t c = (idx / divisor) % channels;
        divisor *= channels;
        const int64_t n = idx / divisor;
        const int64_t g = c / (channels / groups);

        scalar_t value = 0;

        for (int64_t kh = 0l; kh < kernel_size[0]; kh++) {
            const int64_t h_out_s = h + padding[0] - kh * dilation[0];

            for (int64_t kw = 0l; kw < kernel_size[1]; kw++) {
                const int64_t w_out_s = w + padding[1] - kw * dilation[1];

                if (((h_out_s % stride[0]) == 0) && ((w_out_s % stride[1]) == 0)) {
                    const int64_t h_out = h_out_s / stride[0];
                    const int64_t w_out = h_out_s / stride[1];

                    if ((0l <= h_out) && (h_out < out_height) && (0l <= w_out) && (w_out < out_width)) {
                        value += weight_data_a[n][g][kh][kw][h_out][w_out] * out_diff_a[n][c][h_out][w_out];
                    }
                }
            }
        }
        in_diff_p[idx] = value;
    }
}

at::Tensor involution2d_backward_grad_input(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::IntArrayRef& input_shape,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    AT_ASSERTM(grad.device().is_cpu(), "\"grad\" must be a CPU tensor.");
    AT_ASSERTM(weight.device().is_cpu(), "\"weight\" must be a CPU tensor.");

    at::TensorArg grad_t{grad, "grad", 1}, weight_t{weight, "weight", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameType(c, {grad_t, weight_t});

    const auto batch_size = input_shape[0];

    const auto weight_height = weight.size(2);
    const auto weight_width = weight.size(3);

    const at::Tensor weight_ = weight.view({batch_size, groups, kernel_size[0], kernel_size[1], weight_height, weight_width});

    at::Tensor grad_input = at::zeros(input_shape, grad.options());

    if (grad_input.numel() == 0) {
        return grad_input;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad.scalar_type(), "involution2d_backward_grad_input_frame", [&] {
        involution2d_backward_grad_input_frame<scalar_t>(
            grad,
            weight_,
            grad_input,
            kernel_size,
            padding,
            stride,
            dilation
        );
    });

    return grad_input;
}

template <typename scalar_t>
static void involution2d_backward_grad_weight_frame(
    const at::Tensor& out_diff,
    const at::Tensor& in_data,
    at::Tensor& weight_diff,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& dilation
) {
    auto num_elements = weight_diff.numel();
    const auto groups = weight_diff.size(1);
    const auto batch_size = in_data.size(0);
    const auto channels = in_data.size(1);
    const auto in_height = in_data.size(2);
    const auto in_width = in_data.size(3);
    const auto out_height = out_diff.size(2);
    const auto out_width = out_diff.size(3);
    const auto channels_per_group = channels / groups;

    auto out_diff_a = out_diff.accessor<scalar_t, 4>();
    auto in_data_a = in_data.accessor<scalar_t, 4>();
    auto* weight_diff_p = weight_diff.data_ptr<scalar_t>();

    #pragma omp parallel for private(idx)
    for (int64_t idx = 0l; idx < num_elements; idx++) {
        const int64_t w = idx % out_width;
        const int64_t h = (idx / out_width) % out_height;
        int64_t divisor = out_width * out_height;
        const int64_t kw = (idx / divisor) % kernel_size[1];
        divisor *= kernel_size[1];
        const int64_t kh = (idx / divisor) % kernel_size[0];

        const int64_t h_in = h * stride[0] + kh * dilation[0] - padding[0];
        const int64_t w_in = w * stride[1] + kw * dilation[1] - padding[1];

        if ((0l <= h_in) && (h_in < in_height) && (0l <= w_in) && (w_in < in_width)) {
            divisor *= kernel_size[0];
            const int64_t g = (idx / divisor) % groups;
            divisor *= groups;
            const int64_t n = (idx / divisor) % batch_size;

            scalar_t value = 0;

            for (int64_t c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
                value += out_diff_a[n][c][h][w] * in_data_a[n][c][h_in][w_in];
            }
            weight_diff_p[idx] = value;
        }
        else {
            weight_diff_p[idx] = 0;
        }
    }
}

at::Tensor involution2d_backward_grad_weight(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::IntArrayRef& weight_shape,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    AT_ASSERTM(grad.device().is_cpu(), "\"grad\" must be a CPU tensor.");
    AT_ASSERTM(input.device().is_cpu(), "\"input\" must be a CPU tensor.");

    at::TensorArg grad_t{grad, "grad", 1}, input_t{input, "input", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameType(c, {grad_t, input_t});

    const auto batch_size = input.size(0);

    at::Tensor grad_weight = at::zeros({batch_size, groups, kernel_size[0], kernel_size[1], weight_shape[2], weight_shape[3]}, grad.options());

    if (grad_weight.numel() == 0) {
        return grad_weight.view(weight_shape);
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad.scalar_type(),
        "involution2d_backward_grad_weight_kernel", [&] {
            involution2d_backward_grad_weight_frame<scalar_t>(
                grad,
                input,
                grad_weight,
                kernel_size,
                padding,
                stride,
                dilation
            );
        }
    );
    return grad_weight.view(weight_shape);
}

std::vector<at::Tensor> involution2d_backward(
    const at::Tensor& grad,
    const at::Tensor& weight,
    const at::Tensor& input,
    const at::IntArrayRef& kernel_size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& padding,
    const at::IntArrayRef& dilation,
    const int64_t groups
) {
    auto grad_input = involution2d_backward_grad_input(
        grad,
        weight,
        input.sizes(),
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    auto grad_weight = involution2d_backward_grad_weight(
        grad,
        input,
        weight.sizes(),
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );

    return {grad_input, grad_weight};
}

} // namespace cpu
} // namespace involution
