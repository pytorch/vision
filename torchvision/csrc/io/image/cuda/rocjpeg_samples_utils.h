/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef ROC_JPEG_SAMPLES_COMMON
#define ROC_JPEG_SAMPLES_COMMON
#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <functional>
#include <condition_variable>
#include <queue>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif
#include <chrono>
#include "rocjpeg/rocjpeg.h"

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                          \
    }                                                                     \
}

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cout << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                      \
    }                                                                 \
}

/**
 * @class RocJpegUtils
 * @brief Utility class for rocJPEG samples.
 *
 * This class provides utility functions for rocJPEG samples, such as parsing command line arguments,
 * getting file paths, initializing HIP device, getting chroma subsampling string, getting channel pitch and sizes,
 * getting output file extension, and saving images.
 */
class RocJpegUtils {
public:
    /**
     * @brief Parses the command line arguments.
     *
     * This function parses the command line arguments and sets the corresponding variables.
     *
     * @param input_path The input path.
     * @param output_file_path The output file path.
     * @param save_images Flag indicating whether to save images.
     * @param device_id The device ID.
     * @param rocjpeg_backend The rocJPEG backend.
     * @param decode_params The rocJPEG decode parameters.
     * @param num_threads The number of threads.
     * @param crop The crop rectangle.
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    static void ParseCommandLine(std::string &input_path, std::string &output_file_path, bool &save_images, int &device_id,
                                 RocJpegBackend &rocjpeg_backend, RocJpegDecodeParams &decode_params, int *num_threads, int *batch_size, int argc, char *argv[]) {
        if(argc <= 1) {
            ShowHelpAndExit("", num_threads != nullptr, batch_size != nullptr);
        }
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-h")) {
                ShowHelpAndExit("", num_threads != nullptr, batch_size != nullptr);
            }
            if (!strcmp(argv[i], "-i")) {
                if (++i == argc) {
                    ShowHelpAndExit("-i", num_threads != nullptr, batch_size != nullptr);
                }
                input_path = argv[i];
                continue;
            }
            if (!strcmp(argv[i], "-o")) {
                if (++i == argc) {
                    ShowHelpAndExit("-o", num_threads != nullptr, batch_size != nullptr);
                }
                output_file_path = argv[i];
                save_images = true;
                continue;
            }
            if (!strcmp(argv[i], "-d")) {
                if (++i == argc) {
                    ShowHelpAndExit("-d", num_threads != nullptr, batch_size != nullptr);
                }
                device_id = atoi(argv[i]);
                continue;
            }
            if (!strcmp(argv[i], "-be")) {
                if (++i == argc) {
                    ShowHelpAndExit("-be", num_threads != nullptr, batch_size != nullptr);
                }
                rocjpeg_backend = static_cast<RocJpegBackend>(atoi(argv[i]));
                continue;
            }
            if (!strcmp(argv[i], "-fmt")) {
                if (++i == argc) {
                    ShowHelpAndExit("-fmt", num_threads != nullptr, batch_size != nullptr);
                }
                std::string selected_output_format = argv[i];
                if (selected_output_format == "native") {
                    decode_params.output_format = ROCJPEG_OUTPUT_NATIVE;
                } else if (selected_output_format == "yuv_planar") {
                    decode_params.output_format = ROCJPEG_OUTPUT_YUV_PLANAR;
                } else if (selected_output_format == "y") {
                    decode_params.output_format = ROCJPEG_OUTPUT_Y;
                } else if (selected_output_format == "rgb") {
                    decode_params.output_format = ROCJPEG_OUTPUT_RGB;
                } else if (selected_output_format == "rgb_planar") {
                    decode_params.output_format = ROCJPEG_OUTPUT_RGB_PLANAR;
                } else {
                    ShowHelpAndExit(argv[i], num_threads != nullptr);
                }
                continue;
            }
            if (!strcmp(argv[i], "-t")) {
                if (++i == argc) {
                    ShowHelpAndExit("-t", num_threads != nullptr, batch_size != nullptr);
                }
                if (num_threads != nullptr) {
                    *num_threads = atoi(argv[i]);
                    if (*num_threads <= 0 || *num_threads > 32) {
                        ShowHelpAndExit(argv[i], num_threads != nullptr, batch_size != nullptr);
                    }
                }
                continue;
            }
            if (!strcmp(argv[i], "-b")) {
                if (++i == argc) {
                    ShowHelpAndExit("-b", num_threads != nullptr, batch_size != nullptr);
                }
                if (batch_size != nullptr)
                    *batch_size = atoi(argv[i]);
                continue;
            }
            if (!strcmp(argv[i], "-crop")) {
                if (++i == argc || 4 != sscanf(argv[i], "%hd,%hd,%hd,%hd", &decode_params.crop_rectangle.left, &decode_params.crop_rectangle.top, &decode_params.crop_rectangle.right, &decode_params.crop_rectangle.bottom)) {
                    ShowHelpAndExit("-crop");
                }
                if ((&decode_params.crop_rectangle.right - &decode_params.crop_rectangle.left) % 2 == 1 || (&decode_params.crop_rectangle.bottom - &decode_params.crop_rectangle.top) % 2 == 1) {
                    std::cout << "output crop rectangle must have width and height of even numbers" << std::endl;
                    exit(1);
                }
                continue;
            }
            ShowHelpAndExit(argv[i], num_threads != nullptr, batch_size != nullptr);
        }
    }

    /**
     * Checks if a file is a JPEG file.
     *
     * @param filePath The path to the file to be checked.
     * @return True if the file is a JPEG file, false otherwise.
     */
    static bool IsJPEG(const std::string& filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            return false;
        }

        unsigned char buffer[2];
        file.read(reinterpret_cast<char*>(buffer), 2);
        file.close();

        // The first two bytes of every JPEG stream are always 0xFFD8, which represents the Start of Image (SOI) marker.
        return buffer[0] == 0xFF && buffer[1] == 0xD8;
    }

    /**
     * @brief Gets the file paths.
     *
     * This function gets the file paths based on the input path and sets the corresponding variables.
     *
     * @param input_path The input path.
     * @param file_paths The vector to store the file paths.
     * @param is_dir Flag indicating whether the input path is a directory.
     * @param is_file Flag indicating whether the input path is a file.
     * @return True if successful, false otherwise.
     */
    static bool GetFilePaths(std::string &input_path, std::vector<std::string> &file_paths, bool &is_dir, bool &is_file) {
        std::cout << "Reading images from disk, please wait!" << std::endl;
        if (!fs::exists(input_path)) {
            std::cerr << "ERROR: the input path does not exist!" << std::endl;
            return false;
        }
        is_dir = fs::is_directory(input_path);
        is_file = fs::is_regular_file(input_path);
        if (is_dir) {
            for (const auto &entry : fs::recursive_directory_iterator(input_path)) {
                if (fs::is_regular_file(entry) && IsJPEG(entry.path().string())) {
                    file_paths.push_back(entry.path().string());
                }
            }
        } else if (is_file && IsJPEG(input_path)) {
            file_paths.push_back(input_path);
        } else {
            std::cerr << "ERROR: the input path does not contain JPEG files!" << std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief Initializes the HIP device.
     *
     * This function initializes the HIP device with the specified device ID.
     *
     * @param device_id The device ID.
     * @return True if successful, false otherwise.
     */
    static bool InitHipDevice(int device_id) {
        int num_devices;
        hipDeviceProp_t hip_dev_prop;
        CHECK_HIP(hipGetDeviceCount(&num_devices));
        if (num_devices < 1) {
            std::cerr << "ERROR: didn't find any GPU!" << std::endl;
            return false;
        }
        if (device_id >= num_devices) {
            std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
            return false;
        }
        CHECK_HIP(hipSetDevice(device_id));
        CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));

        std::cout << "Using GPU device " << device_id << ": " << hip_dev_prop.name << "[" << hip_dev_prop.gcnArchName << "] on PCI bus " <<
        std::setfill('0') << std::setw(2) << std::right << std::hex << hip_dev_prop.pciBusID << ":" << std::setfill('0') << std::setw(2) <<
        std::right << std::hex << hip_dev_prop.pciDomainID << "." << hip_dev_prop.pciDeviceID << std::dec << std::endl;

        return true;
    }

    /**
     * @brief Gets the chroma subsampling string.
     *
     * This function gets the chroma subsampling string based on the specified subsampling value.
     *
     * @param subsampling The chroma subsampling value.
     * @param chroma_sub_sampling The string to store the chroma subsampling.
     */
    void GetChromaSubsamplingStr(RocJpegChromaSubsampling subsampling, std::string &chroma_sub_sampling) {
        switch (subsampling) {
            case ROCJPEG_CSS_444:
                chroma_sub_sampling = "YUV 4:4:4";
                break;
            case ROCJPEG_CSS_440:
                chroma_sub_sampling = "YUV 4:4:0";
                break;
            case ROCJPEG_CSS_422:
                chroma_sub_sampling = "YUV 4:2:2";
                break;
            case ROCJPEG_CSS_420:
                chroma_sub_sampling = "YUV 4:2:0";
                break;
            case ROCJPEG_CSS_411:
                chroma_sub_sampling = "YUV 4:1:1";
                break;
            case ROCJPEG_CSS_400:
                chroma_sub_sampling = "YUV 4:0:0";
                break;
            case ROCJPEG_CSS_UNKNOWN:
                chroma_sub_sampling = "UNKNOWN";
                break;
            default:
                chroma_sub_sampling = "";
                break;
        }
    }

    /**
     * @brief Gets the channel pitch and sizes.
     *
     * This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
     * output image, and channel sizes.
     *
     * @param decode_params The decode parameters that specify the output format and crop rectangle.
     * @param subsampling The chroma subsampling.
     * @param widths The array to store the channel widths.
     * @param heights The array to store the channel heights.
     * @param num_channels The number of channels.
     * @param output_image The output image.
     * @param channel_sizes The array to store the channel sizes.
     * @return The channel pitch.
     */
    int GetChannelPitchAndSizes(RocJpegDecodeParams decode_params, RocJpegChromaSubsampling subsampling, uint32_t *widths, uint32_t *heights,
                                uint32_t &num_channels, RocJpegImage &output_image, uint32_t *channel_sizes) {
        
        bool is_roi_valid = false;
        uint32_t roi_width;
        uint32_t roi_height;
        roi_width = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
        roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;
        if (roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0]) {
            is_roi_valid = true; 
        }
        switch (decode_params.output_format) {
            case ROCJPEG_OUTPUT_NATIVE:
                switch (subsampling) {
                    case ROCJPEG_CSS_444:
                        num_channels = 3;
                        output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                        channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                        break;
                    case ROCJPEG_CSS_440:
                        num_channels = 3;
                        output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                        channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                        channel_sizes[2] = channel_sizes[1] = output_image.pitch[0] * (is_roi_valid ? align(roi_height >> 1, mem_alignment) : align(heights[0] >> 1, mem_alignment));
                        break;
                    case ROCJPEG_CSS_422:
                        num_channels = 1;
                        output_image.pitch[0] = (is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment)) * 2;
                        channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                        break;
                    case ROCJPEG_CSS_420:
                        num_channels = 2;
                        output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                        channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                        channel_sizes[1] = output_image.pitch[1] * (is_roi_valid ? align(roi_height >> 1, mem_alignment) : align(heights[0] >> 1, mem_alignment));
                        break;
                    case ROCJPEG_CSS_400:
                        num_channels = 1;
                        output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                        channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                        break;
                    default:
                        std::cout << "Unknown chroma subsampling!" << std::endl;
                        return EXIT_FAILURE;
                }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                if (subsampling == ROCJPEG_CSS_400) {
                    num_channels = 1;
                    output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                } else {
                    num_channels = 3;
                    output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                    output_image.pitch[1] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[1], mem_alignment);
                    output_image.pitch[2] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[2], mem_alignment);
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                    channel_sizes[1] = output_image.pitch[1] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[1], mem_alignment));
                    channel_sizes[2] = output_image.pitch[2] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[2], mem_alignment));
                }
                break;
            case ROCJPEG_OUTPUT_Y:
                num_channels = 1;
                output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                break;
            case ROCJPEG_OUTPUT_RGB:
                num_channels = 1;
                output_image.pitch[0] = (is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment)) * 3;
                channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                num_channels = 3;
                output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? align(roi_width, mem_alignment) : align(widths[0], mem_alignment);
                channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? align(roi_height, mem_alignment) : align(heights[0], mem_alignment));
                break;
            default:
                std::cout << "Unknown output format!" << std::endl;
                return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    /**
     * @brief Gets the output file extension.
     *
     * This function gets the output file extension based on the specified output format, base file name,
     * image width, image height, and file name for saving.
     *
     * @param output_format The output format.
     * @param base_file_name The base file name.
     * @param image_width The image width.
     * @param image_height The image height.
     * @param file_name_for_saving The string to store the file name for saving.
     */
    void GetOutputFileExt(RocJpegOutputFormat output_format, std::string &base_file_name, uint32_t image_width, uint32_t image_height, RocJpegChromaSubsampling subsampling, std::string &file_name_for_saving) {
        std::string file_extension;
        std::string::size_type const p(base_file_name.find_last_of('.'));
        std::string file_name_no_ext = base_file_name.substr(0, p);
        std::string format_description = "";
        switch (output_format) {
            case ROCJPEG_OUTPUT_NATIVE:
                file_extension = "yuv";
                switch (subsampling) {
                    case ROCJPEG_CSS_444:
                        format_description = "444";
                        break;
                    case ROCJPEG_CSS_440:
                        format_description = "440";
                        break;
                    case ROCJPEG_CSS_422:
                        format_description = "422_yuyv";
                        break;
                    case ROCJPEG_CSS_420:
                        format_description = "nv12";
                        break;
                    case ROCJPEG_CSS_400:
                        format_description = "400";
                        break;
                    default:
                        std::cout << "Unknown chroma subsampling!" << std::endl;
                        return;
                }
                break;
            case ROCJPEG_OUTPUT_YUV_PLANAR:
                file_extension = "yuv";
                format_description = "planar";
                break;
            case ROCJPEG_OUTPUT_Y:
                file_extension = "yuv";
                format_description = "400";
                break;
            case ROCJPEG_OUTPUT_RGB:
                file_extension = "rgb";
                format_description = "packed";
                break;
            case ROCJPEG_OUTPUT_RGB_PLANAR:
                file_extension = "rgb";
                format_description = "planar";
                break;
            default:
                file_extension = "";
                break;
        }
        file_name_for_saving += "//" + file_name_no_ext + "_" + std::to_string(image_width) + "x"
            + std::to_string(image_height) + "_" + format_description + "." + file_extension;
    }

private:
    static const int mem_alignment = 16;
    /**
     * @brief Shows the help message and exits.
     *
     * This function shows the help message and exits the program.
     *
     * @param option The option to display in the help message (optional).
     * @param show_threads Flag indicating whether to show the number of threads in the help message.
     */
    static void ShowHelpAndExit(const char *option = nullptr, bool show_threads = false, bool show_batch_size = false) {
        std::cout  << "Options:\n"
        "-i     [input path] - input path to a single JPEG image or a directory containing JPEG images - [required]\n"
        "-be    [backend] - select rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN,\n"
        "                                           1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)) [optional - default: 0]\n"
        "-fmt   [output format] - select rocJPEG output format for decoding, one of the [native, yuv_planar, y, rgb, rgb_planar] - [optional - default: native]\n"
        "-o     [output path] - path to an output file or a path to an existing directory - write decoded images to a file or an existing directory based on selected output format - [optional]\n"
        "-crop  [crop rectangle] - crop rectangle for output in a comma-separated format: left,top,right,bottom - [optional]\n"
        "-d     [device id] - specify the GPU device id for the desired device (use 0 for the first device, 1 for the second device, and so on) [optional - default: 0]\n";
        if (show_threads) {
            std::cout << "-t     [threads] - number of threads (<= 32) for parallel JPEG decoding - [optional - default: 1]\n";
        }
        if (show_batch_size) {
            std::cout << "-b     [batch_size] - decode images from input by batches of a specified size - [optional - default: 1]\n";
        }
        exit(0);
    }
    /**
     * @brief Aligns a value to a specified alignment.
     *
     * This function takes a value and aligns it to the specified alignment. It returns the aligned value.
     *
     * @param value The value to be aligned.
     * @param alignment The alignment value.
     * @return The aligned value.
     */
    static inline int align(int value, int alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }
};

class ThreadPool {
    public:
        ThreadPool(int nthreads) : shutdown_(false) {
            // Create the specified number of threads
            threads_.reserve(nthreads);
            for (int i = 0; i < nthreads; ++i)
                threads_.emplace_back(std::bind(&ThreadPool::ThreadEntry, this, i));
        }

        ~ThreadPool() {}

        void JoinThreads() {
            {
                // Unblock any threads and tell them to stop
                std::unique_lock<std::mutex> lock(mutex_);
                shutdown_ = true;
                cond_var_.notify_all();
            }

            // Wait for all threads to stop
            for (auto& thread : threads_)
                thread.join();
        }

        void ExecuteJob(std::function<void()> func) {
            // Place a job on the queue and unblock a thread
            std::unique_lock<std::mutex> lock(mutex_);
            decode_jobs_queue_.emplace(std::move(func));
            cond_var_.notify_one();
        }

    protected:
        void ThreadEntry(int i) {
            std::function<void()> execute_decode_job;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cond_var_.wait(lock, [&] {return shutdown_ || !decode_jobs_queue_.empty();});
                    if (decode_jobs_queue_.empty()) {
                        // No jobs to do; shutting down
                        return;
                    }

                    execute_decode_job = std::move(decode_jobs_queue_.front());
                    decode_jobs_queue_.pop();
                }

                // Execute the decode job without holding any locks
                execute_decode_job();
            }
        }

        std::mutex mutex_;
        std::condition_variable cond_var_;
        bool shutdown_;
        std::queue<std::function<void()>> decode_jobs_queue_;
        std::vector<std::thread> threads_;
};

#endif //ROC_JPEG_SAMPLES_COMMON
