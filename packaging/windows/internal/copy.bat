copy "%CUDA_PATH%\bin\cusparse64_%CUDA_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cublas64_%CUDA_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cudart64_%CUDA_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\curand64_%CUDA_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cufft64_%CUDA_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\cufftw64_%CUDA_VERSION%.dll*" pytorch\torch\lib

copy "%CUDA_PATH%\bin\cudnn64_%CUDNN_VERSION%.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\nvrtc64_%CUDA_VERSION%*.dll*" pytorch\torch\lib
copy "%CUDA_PATH%\bin\nvrtc-builtins64_%CUDA_VERSION%.dll*" pytorch\torch\lib

copy "C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64\nvToolsExt64_1.dll*" pytorch\torch\lib
copy "%CONDA_LIB_PATH%\libiomp*5md.dll" pytorch\torch\lib
