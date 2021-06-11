#
# Be sure to run `pod lib lint LibTorch_vision_ops.podspec' to ensure this is a
# valid spec before submitting.
#
# Any lines starting with a # are optional, but their use is encouraged
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html
#

Pod::Spec.new do |s|
  s.name             = 'LibTorch_vision_ops'
  s.version          = '0.9.0'
  s.summary          = 'libtorchvision_ops library for LibTorch'

# This description is used to generate tags and improve search results.
#   * Think: What does it do? Why did you write it? What is the focus?
#   * Try to keep it short, snappy and to the point.
#   * Write the description between the DESC delimiters below.
#   * Finally, don't worry about the indent, CocoaPods strips it!

  s.description      = <<-DESC
Additional TorchVision operations for LibTorch
                       DESC

  s.homepage         = 'https://github.com/pytorch/vision'
  # s.screenshots     = 'www.example.com/screenshots_1', 'www.example.com/screenshots_2'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = 'Torchvision Team'
  s.source           = { :http => "https://github.com/pytorch/vision/releases/download/v#{s.version}/libtorchvision_ops_ios_#{s.version}.zip" }
  # s.social_media_url = 'https://twitter.com/<TWITTER_USERNAME>'

  s.ios.deployment_target = '12.0'

  s.vendored_libraries = 'lib/*.a'
  s.static_framework = true
  
  s.dependency 'LibTorch', '1.8.0'
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'arm64',
    'OTHER_LDFLAGS' => '-force_load "$(PODS_ROOT)/LibTorch_vision_ops/lib/libtorchvision_ops.a"',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++14',
    'CLANG_CXX_LIBRARY' => 'libc++'
  }
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386 arm64',
    'VALID_ARCHS' => 'x86_64 arm64'
  }
  s.library = ['c++', 'stdc++']
   
end
