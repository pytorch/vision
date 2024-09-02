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
