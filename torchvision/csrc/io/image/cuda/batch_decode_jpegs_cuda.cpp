#include "batched_decode_jpeg_cuda.h"

int decode_images(const std::vector<torch::Tensor>& encoded_images, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  CHECK_CUDA(cudaStreamSynchronize(params.stream));

  // baseline JPEGs can be batch decoded with hardware support on A100+ GPUs
  // ultra fast!
  std::vector<const unsigned char*> hw_input_buffer;
  std::vector<size_t> hw_input_buffer_size;
  std::vector<nvjpegImage_t>  hw_output_buffer;

  // other JPEG types such as progressive JPEGs can be decoded one-by-one in software
  // slow :(
  std::vector<const unsigned char*> sw_input_buffer;
  std::vector<size_t> sw_input_buffer_size;
  std::vector<nvjpegImage_t> sw_output_buffer;

  if(params.hw_decode_available){
    for(int i = 0; i < encoded_images.size(); ++i){
      // extract bitstream meta data to figure out whether a bit-stream can be decoded
      nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], params.jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported);

      if(isSupported == 0){
        hw_input_buffer.push_back((const unsigned char *)img_data[i].data());
        hw_input_buffer_size.push_back(img_len[i]);
        hw_output_buffer.push_back(out[i]);
      } else {
        sw_input_buffer.push_back((const unsigned char *)img_data[i].data());
        sw_input_buffer_size.push_back(img_len[i]);
        sw_output_buffer.push_back(out[i]);
      }
    }
  } else {
    for(int i = 0; i < encoded_images.size(); ++i) {
      sw_input_buffer.push_back((const unsigned char *)img_data[i].data());
      sw_input_buffer_size.push_back(img_len[i]);
      otherdecode_output.push_back(out[i]);
    }
  }

    if(hw_input_buffer.size() > 0)
     {
          CHECK_NVJPEG(
               nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                            hw_input_buffer.size(), 1, params.output_format));

         CHECK_NVJPEG(nvjpegDecodeBatched(
             params.nvjpeg_handle, params.nvjpeg_state, hw_input_buffer.data(),
             hw_input_buffer_size.data(), hw_output_buffer.data(), params.stream));
     }

    if(sw_input_buffer.size() > 0)
    {
          CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
          int buffer_index = 0;
          CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.output_format));
          for (int i = 0; i < encoded_images.size(); ++i) {
              CHECK_NVJPEG(
                  nvjpegJpegStreamParse(params.nvjpeg_handle, sw_input_buffer[i], sw_input_buffer_size[i],
                  0, 0, params.jpeg_streams[buffer_index]));

              CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                  params.pinned_buffers[buffer_index]));

              CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

              CHECK_CUDA(cudaStreamSynchronize(params.stream));

              CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.jpeg_streams[buffer_index], params.stream));

              buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

              CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  &sw_output_buffer[i], params.stream));

          }
    }

  return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].rfind("/");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "/" + sFileName + ".bmp");

    int err;
    if (params.output_format == NVJPEG_OUTPUT_RGB || params.output_format == NVJPEG_OUTPUT_BGR) {
      err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                     iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                     iout[i].pitch[2], widths[i], heights[i]);
    } else if (params.output_format == NVJPEG_OUTPUT_RGBI ||
               params.output_format == NVJPEG_OUTPUT_BGRI) {
      // Write BMP from interleaved data
      err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                      widths[i], heights[i]);
    }
    if (err) {
      std::cout << "Cannot write output file: " << fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done writing decoded image to file: " << fname << std::endl;
  }
  return EXIT_SUCCESS;
}

std::vector<torch::Tensor> batch_decode_jpegs_cuda(const std::vector<torch::Tensor>& encoded_images, ImageReadMode mode,
    torch::Device device) {

    C10_LOG_API_USAGE_ONCE(
      "torchvision.csrc.io.image.cuda.batched_decode_jpeg_cuda.batchedDecodeJpegCuda");
  for (auto &encoded_image : encoded_images) {
    TORCH_CHECK(encoded_image.dtype() == torch::kU8, "Expected a torch.uint8 tensor");

  TORCH_CHECK(
      !encoded_image.is_cuda(),
      "The input tensor must be on CPU when decoding with nvjpeg")

  TORCH_CHECK(
      encoded_image.dim() == 1 && encoded_image.numel() > 0,
      "Expected a non empty 1-dimensional tensor");

  }

  TORCH_CHECK(device.is_cuda(), "Expected a cuda device");

  nvjpegOutputFormat_t ouput_format;
  int num_channels_output;

  switch (mode) {
    case IMAGE_READ_MODE_UNCHANGED:
      num_channels_output = num_channels;
      // For some reason, setting output_format to NVJPEG_OUTPUT_UNCHANGED will
      // not properly decode RGB images (it's fine for grayscale), so we set
      // output_format manually here
      if (num_channels == 1) {
        params.output_format = NVJPEG_OUTPUT_Y;
      } else if (num_channels == 3) {
        params.output_format = NVJPEG_OUTPUT_RGB;
      } else {
        nvjpegJpegStateDestroy(jpeg_state);
        TORCH_CHECK(
            false,
            "When mode is UNCHANGED, only 1 or 3 input channels are allowed.");
      }
      break;
    case IMAGE_READ_MODE_GRAY:
      params.output_format = NVJPEG_OUTPUT_Y;
      num_channels_output = 1;
      break;
    case IMAGE_READ_MODE_RGB:
      params.output_format = NVJPEG_OUTPUT_RGB;
      num_channels_output = 3;
      break;
    default:
      nvjpegJpegStateDestroy(jpeg_state);
      TORCH_CHECK(
          false, "The provided mode is not supported for JPEG decoding on GPU");
  }

  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};

  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator,
                                &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &params.nvjpeg_handle);
  params.hw_decode_available = true;
  if( status == NVJPEG_STATUS_ARCH_MISMATCH) {
    std::cout<<"Hardware Decoder not supported. Falling back to default backend"<<std::endl;
    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                              &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &params.nvjpeg_handle));
    params.hw_decode_available = false;
  } else {
    CHECK_NVJPEG(status);
  }

  CHECK_NVJPEG(
      nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state));

  create_decoupled_api_handles(params);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);

  // stream for decoding
  CHECK_CUDA(
      cudaStreamCreateWithFlags(&params.stream, cudaStreamNonBlocking));

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }
    if (prepare_buffers(file_data, widths, heights, iout, isz,
                        current_names, params))
      return EXIT_FAILURE;

    if (decode_images(file_data, file_len, iout, params, time))
      return EXIT_FAILURE;



    if (params.write_decoded)
      write_images(iout, widths, heights, params, current_names);

  release_buffers(iout);

  CHECK_CUDA(cudaStreamDestroy(params.stream));

  destroy_decoupled_api_handles(params);

  CHECK_NVJPEG(nvjpegJpegStateDestroy(params.nvjpeg_state));
  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}
