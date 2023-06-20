constexpr int threadsPerBlock = 512;

template <typename T>
constexpr inline T ceil_div(T n, T m) {
  return (n + m - 1) / m;
}
