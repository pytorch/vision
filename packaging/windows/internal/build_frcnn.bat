@echo on
set CL=/I"C:\Program Files (x86)\torchvision\include"
msbuild "-p:Configuration=Release" test_frcnn_tracing.vcxproj
