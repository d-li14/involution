#include <involution2d_cuda.cuh>

namespace involution {
namespace cuda {

static u_int32_t ceildiv(u_int32_t num_elements, u_int32_t threads) {
    return (num_elements + threads - 1) / threads;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS_PER_BLOCK)
__global__ static void involution2d_forward_kernel(
    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, int64_t> in_data,
    const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, int64_t> weight_data,
    scalar_t* const out_data,
    const int64_t num_elements,
    const int64_t channels,
    const int64_t groups,
    const int64_t in_height,        const int64_t in_width,
    const int64_t out_height,       const int64_t out_width,
    const int64_t kernel_height,    const int64_t kernel_width,
    const int64_t pad_h,            const int64_t pad_w,
    const int64_t stride_h,         const int64_t stride_w,
    const int64_t dilation_h,       const int64_t dilation_w
) {
    CUDA_KERNEL_LOOP(idx, num_elements) {
        const int64_t w = idx % out_width;
        const int64_t h = (idx / out_width) % out_height;
        int64_t divisor = out_width * out_height;
        const int64_t c = (idx / divisor) % channels;
        divisor *= channels;
        const int64_t n = idx / divisor;
        const int64_t g = c / (channels / groups);

        scalar_t value = 0;

        for (int64_t kh = 0l; kh < kernel_height; kh++) {
            const int64_t h_in = h * stride_h + kh * dilation_h - pad_h;

            if ((0l <= h_in) && (h_in < in_height)) {
                for (int64_t kw = 0l; kw < kernel_width; kw++) {
                    const int64_t w_in = w * stride_w + kw * dilation_w - pad_w;

                    if ((0l <= w_in) && (w_in < in_width)) {
                        value += weight_data[n][g][kh][kw][h][w] * in_data[n][c][h_in][w_in];
                    }
                }
            }
        }

        out_data[idx] = value;
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
    AT_ASSERTM(input.device().is_cuda(), "\"input\" must be a CUDA tensor.");
    AT_ASSERTM(weight.device().is_cuda(), "\"weight\" must be a CUDA tensor.");

    at::TensorArg input_t{input, "input", 1}, weight_t{weight, "weight", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameGPU(c, {input_t, weight_t});
    at::checkAllSameType(c, {input_t, weight_t});

    at::cuda::CUDAGuard device_guard(input.device());

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
    const auto num_elements = output.numel();

    if (num_elements == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return output;
    }

    const auto threads = std::min(static_cast<u_int32_t>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock), CUDA_MAX_THREADS_PER_BLOCK);
    const dim3 num_blocks(ceildiv(num_elements, threads), 1u, 1u);
    const dim3 threads_per_block(threads, 1u, 1u);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "involution2d_forward_kernel", [&] {
            involution2d_forward_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int64_t>(),
                weight_.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, int64_t>(),
                output.data_ptr<scalar_t>(),
                num_elements,
                channels,
                groups,
                in_height, in_width,
                out_height, out_width,
                kernel_size[0], kernel_size[1],
                padding[0], padding[1],
                stride[0], stride[1],
                dilation[0], dilation[1]
            );
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS_PER_BLOCK)
__global__ static void involution2d_backward_grad_input_kernel(
    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, int64_t> out_diff,
    const at::GenericPackedTensorAccessor<scalar_t, 6, at::RestrictPtrTraits, int64_t> weight_data,
    scalar_t* const in_diff,
    const int64_t num_elements,
    const int64_t channels,
    const int64_t groups,
    const int64_t in_height,        const int64_t in_width,
    const int64_t out_height,       const int64_t out_width,
    const int64_t kernel_height,    const int64_t kernel_width,
    const int64_t pad_h,            const int64_t pad_w,
    const int64_t stride_h,         const int64_t stride_w,
    const int64_t dilation_h,       const int64_t dilation_w
) {
    CUDA_KERNEL_LOOP(idx, num_elements) {
        const int64_t w = idx % in_width;
        const int64_t h = (idx / in_width) % in_height;
        int64_t divisor = in_width * in_height;
        const int64_t c = (idx / divisor) % channels;
        divisor *= channels;
        const int64_t n = idx / divisor;
        const int64_t g = c / (channels / groups);

        scalar_t value = 0;

        for (int64_t kh = 0l; kh < kernel_height; kh++) {
            const int64_t h_out_s = h + pad_h - kh * dilation_h;

            for (int64_t kw = 0l; kw < kernel_width; kw++) {
                const int64_t w_out_s = w + pad_w - kw * dilation_w;

                if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
                    const int64_t h_out = h_out_s / stride_h;
                    const int64_t w_out = h_out_s / stride_w;

                    if ((0l <= h_out) && (h_out < out_height) && (0l <= w_out) && (w_out < out_width)) {
                        value += weight_data[n][g][kh][kw][h_out][w_out] * out_diff[n][c][h_out][w_out];
                    }
                }
            }
        }
        in_diff[idx] = value;
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
    AT_ASSERTM(grad.device().is_cuda(), "\"grad\" must be a CUDA tensor.");
    AT_ASSERTM(weight.device().is_cuda(), "\"weight\" must be a CUDA tensor.");

    at::TensorArg grad_t{grad, "grad", 1}, weight_t{weight, "weight", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameGPU(c, {grad_t, weight_t});
    at::checkAllSameType(c, {grad_t, weight_t});

    at::cuda::CUDAGuard device_guard(grad.device());

    const auto batch_size = input_shape[0];
    const auto channels = input_shape[1];
    const auto in_height = input_shape[2];
    const auto in_width = input_shape[3];

    const auto weight_height = weight.size(2);
    const auto weight_width = weight.size(3);

    const at::Tensor weight_ = weight.view({batch_size, groups, kernel_size[0], kernel_size[1], weight_height, weight_width});

    const auto out_height = grad.size(2);
    const auto out_width = grad.size(3);

    at::Tensor grad_input = at::zeros(input_shape, grad.options());
    const auto num_elements = grad_input.numel();

    if (num_elements == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_input;
    }

    const auto threads = std::min(static_cast<u_int32_t>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock), CUDA_MAX_THREADS_PER_BLOCK);
    const dim3 num_blocks(ceildiv(num_elements, threads), 1u, 1u);
    const dim3 threads_per_block(threads, 1u, 1u);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad.scalar_type(),
        "involution2d_backward_grad_input_kernel", [&] {
            involution2d_backward_grad_input_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
                grad.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int64_t>(),
                weight_.generic_packed_accessor<scalar_t, 6, at::RestrictPtrTraits, int64_t>(),
                grad_input.data_ptr<scalar_t>(),
                num_elements,
                channels,
                groups,
                in_height, in_width,
                out_height, out_width,
                kernel_size[0], kernel_size[1],
                padding[0], padding[1],
                stride[0], stride[1],
                dilation[0], dilation[1]
            );
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(CUDA_MAX_THREADS_PER_BLOCK)
__global__ static void involution2d_backward_grad_weight_kernel(
    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, int64_t> out_diff,
    const at::GenericPackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, int64_t> in_data,
    scalar_t* const weight_diff,
    const int64_t num_elements,
    const int64_t batch_size,
    const int64_t channels_per_group,
    const int64_t groups,
    const int64_t in_height,        const int64_t in_width,
    const int64_t out_height,       const int64_t out_width,
    const int64_t kernel_height,    const int64_t kernel_width,
    const int64_t pad_h,            const int64_t pad_w,
    const int64_t stride_h,         const int64_t stride_w,
    const int64_t dilation_h,       const int64_t dilation_w
) {
    CUDA_KERNEL_LOOP(idx, num_elements) {
        const int64_t w = idx % out_width;
        const int64_t h = (idx / out_width) % out_height;
        int64_t divisor = out_width * out_height;
        const int64_t kw = (idx / divisor) % kernel_width;
        divisor *= kernel_width;
        const int64_t kh = (idx / divisor) % kernel_height;

        const int64_t h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int64_t w_in = -pad_w + w * stride_w + kw * dilation_w;

        if ((0l <= h_in) && (h_in < in_height) && (0l <= w_in) && (w_in < in_width)) {
            divisor *= kernel_height;
            const int64_t g = (idx / divisor) % groups;
            divisor *= groups;
            const int64_t n = (idx / divisor) % batch_size;

            scalar_t value = 0;

            for (int64_t c = g * channels_per_group; c < (g + 1) * channels_per_group; c++) {
                value += out_diff[n][c][h][w] * in_data[n][c][h_in][w_in];
            }
            weight_diff[idx] = value;
        }
        else {
            weight_diff[idx] = 0;
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
    AT_ASSERTM(grad.device().is_cuda(), "\"grad\" must be a CUDA tensor.");
    AT_ASSERTM(input.device().is_cuda(), "\"input\" must be a CUDA tensor.");

    at::TensorArg grad_t{grad, "grad", 1}, input_t{input, "input", 2};

    at::CheckedFrom c = __func__;
    at::checkAllSameGPU(c, {grad_t, input_t});
    at::checkAllSameType(c, {grad_t, input_t});

    at::cuda::CUDAGuard device_guard(grad.device());

    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);

    const auto out_height = grad.size(2);
    const auto out_width = grad.size(3);

    at::Tensor grad_weight = at::zeros({batch_size, groups, kernel_size[0], kernel_size[1], weight_shape[2], weight_shape[3]}, grad.options());
    const auto num_elements = grad_weight.numel();

    if (num_elements == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_weight.view(weight_shape);
    }

    const auto threads = std::min(static_cast<u_int32_t>(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock), CUDA_MAX_THREADS_PER_BLOCK);
    const dim3 num_blocks(ceildiv(num_elements, threads), 1u, 1u);
    const dim3 threads_per_block(threads, 1u, 1u);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad.scalar_type(),
        "involution2d_backward_grad_weight_kernel", [&] {
            involution2d_backward_grad_weight_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
                grad.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int64_t>(),
                input.generic_packed_accessor<scalar_t, 4, at::RestrictPtrTraits, int64_t>(),
                grad_weight.data_ptr<scalar_t>(),
                num_elements,
                batch_size,
                channels / groups,
                groups,
                in_height, in_width,
                out_height, out_width,
                kernel_size[0], kernel_size[1],
                padding[0], padding[1],
                stride[0], stride[1],
                dilation[0], dilation[1]
            );
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
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

} // namespace cuda
} // namespace involution
