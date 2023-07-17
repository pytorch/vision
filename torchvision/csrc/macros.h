#pragma once

#if defined(_WIN32) && !defined(TORCHVISION_BUILD_STATIC_LIBS)
#if defined(torchvision_EXPORTS)
#define VISION_API __declspec(dllexport)
#else
#define VISION_API __declspec(dllimport)
#endif
#else
#define VISION_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define VISION_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define VISION_INLINE_VARIABLE __declspec(selectany)
#define HINT_MSVC_LINKER_INCLUDE_SYMBOL
#else
#define VISION_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
