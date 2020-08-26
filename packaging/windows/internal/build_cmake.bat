@echo on
msbuild "-p:Configuration=Release" torchvision.vcxproj
msbuild "-p:Configuration=Release" INSTALL.vcxproj
