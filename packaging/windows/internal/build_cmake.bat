@echo on
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" torchvision.vcxproj %1
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" INSTALL.vcxproj %1
