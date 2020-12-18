@echo on
set CL=/I"C:\Program Files (x86)\torchvision\include"
msbuild "-p:Configuration=Release" "-p:BuildInParallel=true" hello-world.vcxproj %1
