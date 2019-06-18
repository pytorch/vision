#ifndef VISION_GENERAL_H
#define VISION_GENERAL_H

namespace vision {
namespace models {

#ifdef _WIN32
#define VISION_API __declspec(dllexport)
#else
#define VISION_API
#endif

} // namespace models
} // namespace vision

#endif // VISION_GENERAL_H