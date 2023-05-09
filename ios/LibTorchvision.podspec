pytorch_version = '2.0.0'

Pod::Spec.new do |s|
    s.name             = 'LibTorchvision'
    s.version          = '0.15.1'
    s.authors          = 'PyTorch Team'
    s.license          = { :type => 'BSD' }
    s.homepage         = 'https://github.com/pytorch/vision'
    s.source           = { :http => "https://ossci-ios.s3.amazonaws.com/libtorchvision_ops_ios_#{s.version}.zip" }
    s.summary          = '"The C++ library of TorchVision ops for iOS'
    s.description      = <<-DESC
        The C++ library of TorchVision ops for iOS.
        This version (#{s.version}) requires the installation of LibTorch #{pytorch_version} or LibTorch-Lite #{pytorch_version}.
    DESC
    s.ios.deployment_target = '12.0'
    s.vendored_libraries = 'install/lib/*.a'
    s.user_target_xcconfig = {
        'VALID_ARCHS' => 'x86_64 arm64',
        'OTHER_LDFLAGS' => '$(inherited) -force_load "$(PODS_ROOT)/LibTorchvision/install/lib/libtorchvision_ops.a"',
        'CLANG_CXX_LANGUAGE_STANDARD' => 'c++14',
        'CLANG_CXX_LIBRARY' => 'libc++'
    }
    s.library = ['c++', 'stdc++']
end
