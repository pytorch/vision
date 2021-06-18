
#import "ViewController.h"
#include <torch/script.h>
#import "ModelRunner.h"

@interface ViewController ()
@property (weak, nonatomic) IBOutlet UITextView *textView;
@end

static NSString const *config_error_msg = @"Wrong model configurations... Please fix and click \"Redo\"";

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  if ([ModelRunner setUp]) {
    [self testModel];
  } else {
    self.textView.text = [config_error_msg copy];
  }
}


- (IBAction)rerun:(id)sender {
  self.textView.text = @"";
  if (![ModelRunner setUp]) {
    self.textView.text = [config_error_msg copy];
    return;
  }
  dispatch_async(dispatch_get_main_queue(), ^{
    [self testModel];
  });
}

- (void)testModel {
  dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSString *text = [ModelRunner run];
    dispatch_async(dispatch_get_main_queue(), ^{
      self.textView.text = [self.textView.text stringByAppendingString:text];
    });
  });
}

@end
