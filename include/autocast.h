#pragma once

#include <ATen/ATen.h>

namespace autocast {

inline bool is_eligible(const at::Tensor& arg) {
    return (
        arg.device().is_cuda() && arg.is_floating_point() && (arg.scalar_type() != at::kDouble)
    );
}

inline at::Tensor _cast(at::ScalarType& to_type, const at::Tensor& arg) {
    if (is_eligible(arg) && (arg.scalar_type() != to_type)) {
        return arg.to(to_type);
    }
    else {
        return arg;
    }
}

template <typename scalar_t>
inline scalar_t _cast(at::ScalarType to_type, scalar_t& arg) {
    return arg;
}

}   //  namespace autocast
