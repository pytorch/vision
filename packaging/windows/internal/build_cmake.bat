@echo on
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" "-p:MultiProcessorCompilation=true" "-p:CL_MPCount=%1" torchvision.vcxproj -maxcpucount:%1
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" "-p:MultiProcessorCompilation=true" "-p:CL_MPCount=%1" INSTALL.vcxproj -maxcpucount:%1
