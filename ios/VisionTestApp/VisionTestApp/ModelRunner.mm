
#import "ModelRunner.h"
#include <string>
#include <vector>
#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/serialization/import.h"
#include "torch/script.h"

static NSString *model_name = @"frcnn_mnetv3";
static NSString *model_suffix = @"pt";
static NSString *model_path = nil;
static int warmup = 5;
static int iter = 20;

@implementation ModelRunner

+ (NSString *)run {
  std::vector<std::string> logs;
#define UI_LOG(fmt, ...)                                          \
  {                                                               \
    NSString* log = [NSString stringWithFormat:fmt, __VA_ARGS__]; \
    NSLog(@"%@", log);                                            \
    logs.push_back(log.UTF8String);                               \
  }
  
  auto module = torch::jit::load(std::string(model_path.UTF8String));
  module.eval();
  
  std::vector<c10::IValue> inputs;
  auto img_tensor = torch::ones({3, 224, 224}, at::ScalarType::Float);
  inputs.push_back(c10::List<at::Tensor>(img_tensor));
  torch::autograd::AutoGradMode guard(false);
  at::InferenceMode nonVarTypeModeGuard(true);
  
  UI_LOG(@"Running warmup runs...", nil);
  for (int i = 0; i < warmup; ++i) {
    module.forward(inputs);
  }
  UI_LOG(@"Warmup runs finished.\nMain runs...", nil);
  caffe2::Timer timer;
  auto millis = timer.MilliSeconds();
  for (int i = 0; i < iter; ++i) {
    module.forward(inputs);
  }
  millis = timer.MilliSeconds();
  UI_LOG(@"Main run finished. \nMilliseconds per iter: %.3f", millis / iter, nil);
  UI_LOG(@"Iters per second: : %.3f", 1000.0 * iter / millis, nil);
  UI_LOG(@"Done.", nil);
  
  std::cout << module.forward(inputs) << std::endl;
  
  NSString* log_text = @"";
  for (auto& msg : logs) {
    log_text = [log_text stringByAppendingString:[NSString stringWithUTF8String:msg.c_str()]];
    log_text = [log_text stringByAppendingString:@"\n"];
  }
  return log_text;
}

+ (BOOL)setUp {
  model_path = [[NSBundle mainBundle] pathForResource:model_name ofType:model_suffix];
  if (![[NSFileManager defaultManager] fileExistsAtPath:model_path]) {
    NSLog(@"Invalid model path!");
    model_path = nil;
    return NO;
  }
  return YES;
}

@end
