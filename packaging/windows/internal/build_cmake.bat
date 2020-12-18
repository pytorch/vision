@echo on
msbuild "-p:Configuration=Release" torchvision.vcxproj %1
msbuild "-p:Configuration=Release" INSTALL.vcxproj %1
