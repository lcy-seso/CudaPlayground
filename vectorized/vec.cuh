#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>

typedef __nv_bfloat16 __bfloat16;
typedef __nv_bfloat162 __bfloat162;

#define HOST_DEVICE __forceinline__ __host__ __device__

// This struct represents a 64-bit vector containing 4 bfloat16 values.
// It is designed to efficiently utilize GPU's vectorization capabilities
// beyond the default 32-bit bfloat162 operations.
// packed 64-bit data, sizeof(bfloat164) = 8
struct __align__(8) __bfloat164 {
  __bfloat162 x, y;

  HOST_DEVICE __bfloat164() : x(), y() {}

  HOST_DEVICE __bfloat164(__bfloat162 x_, __bfloat162 y_) : x(x_), y(y_) {}

  HOST_DEVICE __bfloat164(__bfloat16 a, __bfloat16 b, __bfloat16 c,
                          __bfloat16 d)
      : x(make_bfloat162(a, b)), y(make_bfloat162(c, d)) {}

  // broadcast to all elements
  HOST_DEVICE __bfloat164(__bfloat16 v)
      : x(make_bfloat162(v, v)), y(make_bfloat162(v, v)) {}

  // broadcast to all elements
  HOST_DEVICE __bfloat164(__nv_bfloat162 v) : x(v), y(v) {}

  HOST_DEVICE __bfloat164& operator+=(const __bfloat164& b) {
    x = __hadd2(x, b.x);
    y = __hadd2(y, b.y);
    return *this;
  }

  HOST_DEVICE __bfloat164& operator=(const __bfloat16& b) {
    __bfloat162 v = make_bfloat162(b, b);
    x = v;
    y = v;
    return *this;
  }

  HOST_DEVICE __bfloat164& operator=(const __bfloat162& b) {
    x = b;
    y = b;
    return *this;
  }

  HOST_DEVICE __bfloat164& operator=(const __bfloat164& b) {
    x = b.x;
    y = b.y;
    return *this;
  }
};

HOST_DEVICE void print_bfloat164(const __bfloat164& vec) {
  printf("%.2f, %.2f, %.2f, %.2f\n", __bfloat162float(vec.x.x),
         __bfloat162float(vec.x.y), __bfloat162float(vec.y.x),
         __bfloat162float(vec.y.y));
}

HOST_DEVICE __bfloat164 operator+(const __bfloat164& a, const __bfloat164& b) {
  __bfloat164 res;
  res.x = __hadd2(a.x, b.x);
  res.y = __hadd2(a.y, b.y);
  return res;
}

HOST_DEVICE __bfloat164 operator*(const __bfloat164& a, const __bfloat164& b) {
  return __bfloat164(__hmul2(a.x, b.x), __hmul2(a.y, b.y));
}

// This struct represents a 128-bit vector containing 8 bfloat16 values.
// It is designed to efficiently utilize GPU's vectorization capabilities
// beyond the default 32-bit bfloat162 operations.
// packed 128-bit data, sizeof(bfloat168) = 16
struct __align__(16) __bfloat168 {
  __bfloat164 x, y;

  HOST_DEVICE __bfloat168() : x(), y() {}

  HOST_DEVICE __bfloat168(__bfloat164 x_, __bfloat164 y_) : x(x_), y(y_) {}

  HOST_DEVICE __bfloat168(__bfloat16 a, __bfloat16 b, __bfloat16 c,
                          __bfloat16 d, __bfloat16 e, __bfloat16 f,
                          __bfloat16 g, __bfloat16 h)
      : x(__bfloat164(a, b, c, d)), y(__bfloat164(e, f, g, h)) {}

  HOST_DEVICE __bfloat168(__nv_bfloat162 v)
      : x(__bfloat164(v)), y(__bfloat164(v)) {}

  HOST_DEVICE __bfloat168& operator+=(const __bfloat168& b) {
    x += b.x;
    y += b.y;
    return *this;
  }

  // Assignment operator for single bfloat16 value (broadcast)
  HOST_DEVICE __bfloat168& operator=(const __bfloat16& b) {
    __bfloat164 v(b);
    x = v;
    y = v;
    return *this;
  }

  // Assignment operator for bfloat162 value (broadcast)
  HOST_DEVICE __bfloat168& operator=(const __bfloat162& b) {
    __bfloat164 v(b);
    x = v;
    y = v;
    return *this;
  }

  // Assignment operator for bfloat164 value (broadcast)
  HOST_DEVICE __bfloat168& operator=(const __bfloat164& b) {
    x = b;
    y = b;
    return *this;
  }

  // Assignment operator for bfloat168 value
  HOST_DEVICE __bfloat168& operator=(const __bfloat168& b) {
    x = b.x;
    y = b.y;
    return *this;
  }
};

HOST_DEVICE __bfloat168 operator+(const __bfloat168& a, const __bfloat168& b) {
  __bfloat168 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

HOST_DEVICE __bfloat168 operator*(const __bfloat168& a, const __bfloat168& b) {
  return __bfloat168(a.x * b.x, a.y * b.y);
}

HOST_DEVICE void print_bfloat168(const __bfloat168& vec) {
  printf("%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n",
         __bfloat162float(vec.x.x.x), __bfloat162float(vec.x.x.y),
         __bfloat162float(vec.x.y.x), __bfloat162float(vec.x.y.y),
         __bfloat162float(vec.y.x.x), __bfloat162float(vec.y.x.y),
         __bfloat162float(vec.y.y.x), __bfloat162float(vec.y.y.y));
}
