#ifndef VISION_GENERAL_H
#define VISION_GENERAL_H

#ifdef _WIN32
#if defined(torchvision_EXPORTS)
#define VISION_API __declspec(dllexport)
#else
#define VISION_API __declspec(dllimport)
#endif
#else
#define VISION_API
#endif

#endif // VISION_GENERAL_H