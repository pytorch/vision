#pragma once

#ifdef WITH_CUDA
namespace autocast {

inline bool is_eligible(const at::Tensor& arg) {
  return (
      arg.is_cuda() && arg.is_floating_point() &&
      (arg.scalar_type() != at::kDouble));
}

// Overload to catch Tensor args
inline at::Tensor _cast(at::ScalarType to_type, const at::Tensor& arg) {
  if (is_eligible(arg) && (arg.scalar_type() != to_type)) {
    return arg.to(to_type);
  } else {
    return arg;
  }
}

// Template to catch non-Tensor args
template <typename T>
inline T _cast(at::ScalarType to_type, T arg) {
  return arg;
}

} // namespace autocast
#endif
