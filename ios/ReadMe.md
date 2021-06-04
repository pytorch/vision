## Torchvision ops for iOS

### Cocoapods Developers

Torchvision ops is available through Cocoapods

```ruby
pod 'LibTorch'
pod 'LibTorch_vision_ops'
```

### Import the library

For Objective-C developers, simply import the umbrella header

```
#import <LibTorch/LibTorch.h>
```

### To generate cocoapods package
Simply run `build_podspec.sh`
It will do:
1. Download latest `LibTorch`
2. Compile torchvision for `x86_64` and `arm64`.
3. Combine libraries (`x86_64` and `arm64`) into one fat binary.
4. Create zip file containing, `libtorchvision_ops.a` and `LICENSE`.

### To push Cocoapods library
1. Specify new version of library
2. Upload `libtorchvision_ios.zip` to release as `libtorchvision_ios_{version}.zip`, version and release version should be the same, for `0.9.0` version upload archive to tag `v0.9.0`.
3. `pod trunk push LibTorch_vision_ops.podspec` to publish.