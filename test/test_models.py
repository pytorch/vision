from collections import OrderedDict
from itertools import product
import torch
from torchvision import models
import unittest
import math



# debug
import inspect
import random



def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchHvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# model_name, expected to script without error
torchub_models = {
    "alexnet": True, # √ indicates updated test has been written
    "resnet18": True, # √
    "resnext50_32x4d": False, # √
    "vgg11": True, # √
    "squeezenet1_0": True, # √
    "inception_v3": False, # √
    "googlenet": False, # √
    "mobilenet_v2": True, # √
    "shufflenet_v2_x1_0": True, # √
    "densenet121": False, # √
    "deeplabv3_resnet101": False,
    "fcn_resnet101": False,
}

STANDARD_SEED = 1729 # https://fburl.com/3i5wkg9p
STANDARD_INPUT_SHAPE = (1, 3, 224, 224) # for ImageNet-trained models
EPSILON = 1e-4

class TorchVisionTester(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(1)

    TEST_INPUTS = {}

    # set random seed for whatever callable follows (if any)
    # can be called with no args to just set to standard random seed
    def _rand_sync(self, callable=None, **kwargs):
        torch.manual_seed(STANDARD_SEED)
        if callable is not None:
            return callable(**kwargs)

    # create a randomly-weighted model w/ synced RNG state
    def _get_test_model(self, callable, **kwargs):
        return self._rand_sync(callable, **kwargs).eval()

    # create random tensor with given shape using synced RNG state
    # caching because these tests take pretty long already (instantiating models and all)
    def _get_test_input(self, shape):
        # NOTE not thread-safe, but should give same results even if multi-threaded testing gave a race condition
        # giving consistent results is kind of the point of this helper method
        if shape not in self.TEST_INPUTS:
            self.TEST_INPUTS[shape] = self._rand_sync(lambda: torch.rand(shape))
        return self.TEST_INPUTS[shape]

    def _infer_for_test_with(self, model, x):
        return self._rand_sync(lambda: model(x)) # rand state force because dropout &c

    def _check_scriptable(self, model, should_be_scriptable):
        is_actually_scriptable = True
        try:
            torch.jit.script(model)
        except Exception:
            is_actually_scriptable = False
        self.assertEqual(should_be_scriptable, is_actually_scriptable)

    def _check_classification_output_shape(self, model, test_input, num_classes):
        out = model(test_input)
        self.assertEqual(out.shape, (1, num_classes))

    def _relative_error_within(self, expected, actual, tolerance):
        if expected == 0.:
            return actual == 0.
        abserr = abs(expected - actual)
        if not ((abserr / expected) < tolerance) and False: # debug
            print('expected {}'.format(expected))
            print('actual {}'.format(actual))
            print('tolerance {}'.format(tolerance))
            print('abserr {}'.format(abserr))
            print('relerr {}'.format(abserr / expected))
            print('output {}'.format((abserr / expected) < tolerance))
        return (abserr / expected) < tolerance

    def _check_model_correctness(self, model, x, expected_values, num_classes):
        y = self._infer_for_test_with(model, x) # because dropout &c
        self._check_classification_output_shape(model, x, num_classes)
        for k in expected_values:
            self.assertTrue(self._relative_error_within(expected_values[k], y[0][k].item(), EPSILON),
                            'output tensor value at {} should be {}, got {}'.format(k, expected_values[k], y[0][k].item()))

    # I'm using this to help build sets of expected values; input indices are generated randomly elsewhere
    def _build_correctness_check(self, model, input_shape, indices = None):
        if indices is None:
            indices = []
            for i in range(10):
                indices.append(random.randint(0,1000))
            indices.sort()

        # do inference with RNG in known state for determinacy
        y = self._infer_for_test_with(model, self._get_test_input(input_shape))
        print(y.shape) # output shape
        vals = []
        for i in indices:
            vals.append(y[0][i].item())
        print('{')
        for i in range(len(indices)):
            print('            {} : {},'.format(indices[i], '%E' % vals[i]))
        print('}')

class AlexnetTester(TorchVisionTester): # run time ~2s

    def test_classification_alexnet(self):
        # num_classes=1000
        model = self._get_test_model(models.alexnet)
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)

        self._check_scriptable(model, True)
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            17 : -1.784961E-02,
            85 : 4.111526E-03,
            260 : -7.347998E-03,
            422 : -5.389921E-03,
            546 : 4.218812E-04,
            583 : 6.394656E-04,
            648 : 1.458143E-02,
            652 : 3.941567E-03,
            666 : -1.270696E-02,
            805 : 1.309388E-02
        }
        self._check_model_correctness(model, test_input, expected_values, 1000)

class ResnetTester(TorchVisionTester): # run time ~130s

    # all resnet classes assumed to have default num_classes=1000
    # TODO might be worth testing args width_per_group, replace_stride_with_dilation, norm_layer, groups, zero_init_residual
    def _test_classification_resnet(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_resnet18(self):
        model = self._get_test_model(models.resnet18)
        self._check_scriptable(model, True)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            84 : 7.921292E-01,
            95 : -8.626075E-01,
            198 : -3.315417E-01,
            209 : -4.074208E-01,
            267 : -1.350757E+00,
            288 : 5.114775E-01,
            468 : -9.876035E-01,
            678 : 1.986771E-01,
            686 : -2.126825E-01,
            745 : 1.322185E-01
        }
        self._test_classification_resnet(model, expected_values)


    def test_classification_resnet34(self):
        # NOTE passes scriptability check, but none was previously specified
        model = self._get_test_model(models.resnet34)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            78 : 1.051084E+01,
            128 : -3.450469E+00,
            426 : 4.292104E+00,
            456 : 2.427412E+01,
            480 : -4.657684E+00,
            557 : 9.758177E+00,
            646 : -1.581276E+01,
            749 : 2.536623E+00,
            893 : 2.102486E+01,
            901 : 8.516801E-01
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet50(self):
        # NOTE fails scriptability check, but none was previously specified
        model = self._get_test_model(models.resnet50)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            90 : -1.128571E+01,
            288 : 8.061395E+00,
            367 : 8.420571E+00,
            462 : -2.997678E+00,
            474 : 5.429098E+00,
            575 : -1.098446E+01,
            635 : -7.044466E+00,
            706 : 1.058293E+01,
            863 : -5.997756E+00,
            913 : -4.179638E+00
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet101(self):
        # NOTE fails scriptability check, but none was previously specified
        # NOTE weird, high output values compared to some others in resnet family
        model = self._get_test_model(models.resnet101)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            223 : 4.355917E+03,
            250 : -4.877168E+03,
            262 : 1.647226E+03,
            436 : -1.239408E+04,
            531 : 4.820865E+03,
            559 : 1.529453E+03,
            702 : -1.411818E+04,
            769 : -2.623067E+03,
            990 : -2.992360E+03,
            997 : -6.060187E+03
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet152(self):
        # NOTE fails scriptability check, but none was previously specified
        # NOTE weird, high output values compared to some others in resnet family
        model = self._get_test_model(models.resnet152)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            202 : 3.760286E+06,
            366 : 3.434094E+07,
            461 : 2.416596E+07,
            463 : -8.811064E+06,
            508 : 3.172601E+06,
            624 : 4.106310E+05,
            770 : 1.020243E+07,
            826 : 1.066852E+07,
            854 : -2.702246E+07,
            868 : 1.479076E+07
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnext50_32x4d(self):
        model = self._get_test_model(models.resnext50_32x4d)
        # self._check_scriptable(model, False)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            8 : 1.195142E-02,
            232 : -3.794406E-02,
            366 : 2.968006E-02,
            375 : 4.636388E-02,
            393 : -8.227661E-03,
            836 : 6.085198E-02,
            854 : 2.434794E-02,
            943 : -5.337046E-03,
            955 : 1.393929E-03,
            974 : -6.908817E-03
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnext101_32x8d(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.resnext101_32x8d)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            6 : -5.437661E-02,
            108 : 1.919848E-02,
            325 : -4.192499E-03,
            327 : -9.156425E-02,
            361 : -5.895464E-02,
            510 : -4.544296E-02,
            568 : -2.835527E-02,
            679 : -4.588182E-02,
            692 : 2.861079E-02,
            820 : 5.712935E-02
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_wide_resnet50_2(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.wide_resnet50_2)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            0 : 2.909388E+00,
            28 : 1.198006E+01,
            247 : -9.703583E+00,
            272 : 1.206837E+00,
            371 : 1.662452E-02,
            407 : -3.991163E+00,
            416 : -5.484074E+00,
            600 : 4.748647E+00,
            758 : 1.355240E+01,
            804 : 9.518572E+00
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_wide_resnet101_2(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.wide_resnet101_2)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            116 : 9.204982E+03,
            144 : 4.886913E+03,
            213 : 8.238505E+03,
            391 : 1.011332E+04,
            412 : -2.144386E+03,
            449 : -1.285868E+03,
            546 : 4.408850E+03,
            560 : 1.586058E+04,
            754 : 7.677771E+03,
            955 : 4.844999E+03
        }
        self._test_classification_resnet(model, expected_values)

class VGGTester(TorchVisionTester): # run time ~140s
    # num_classes=1000
    # TODO test with init_weights?
    def _test_classification_vgg(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_vgg11(self):
        model = self._get_test_model(models.vgg11)
        self._check_scriptable(model, True)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE)
        expected_values = { # known good values for this model with rand seeded to standard
            19 : 3.458232E-03,
            88 : 1.335686E-02,
            140 : -3.639125E-02,
            345 : -7.817472E-03,
            590 : -3.509789E-02,
            711 : -6.121674E-03,
            730 : 3.107173E-02,
            736 : -1.727812E-02,
            786 : 1.223841E-02,
            996 : 2.816231E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg11_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg11_bn)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [9, 16, 66, 212, 218, 254, 270, 790, 942, 959])
        expected_values = { # known good values for this model with rand seeded to standard
            9 : -2.787093E-03,
            16 : 3.576707E-02,
            66 : -2.639397E-02,
            212 : -5.736205E-03,
            218 : -7.492830E-03,
            254 : -7.046573E-04,
            270 : -1.804215E-02,
            790 : -1.000134E-02,
            942 : 1.227958E-02,
            959 : -2.625632E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg13(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg13)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [129, 205, 272, 336, 401, 432, 812, 860, 880, 964])
        expected_values = { # known good values for this model with rand seeded to standard
            129 : -5.939748E-03,
            205 : 5.725319E-03,
            272 : -1.597922E-02,
            336 : -4.327245E-02,
            401 : -7.134268E-02,
            432 : 8.038238E-03,
            812 : 2.041768E-02,
            860 : -2.816942E-02,
            880 : -8.570410E-03,
            964 : 5.890184E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg13_bn(self):
        # NOTE no scriptability check specified
        # NOTE this also passes using the expected values from vgg13 - is that expected?
        #      not the case with vgg11[_bn] (though some values were very close, just not within EPSILON)
        model = self._get_test_model(models.vgg13_bn)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [243, 347, 407, 591, 665, 668, 827, 866, 916, 969])
        expected_values = { # known good values for this model with rand seeded to standard
            243 : -1.422615E-02,
            347 : -3.358474E-02,
            407 : -1.568302E-02,
            591 : -2.871177E-02,
            665 : -2.439759E-02,
            668 : -6.531857E-02,
            827 : -7.009151E-03,
            866 : 5.519774E-02,
            916 : 1.613093E-02,
            969 : -5.870605E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg16(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg16)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [249, 441, 514, 615, 620, 650, 794, 909, 929, 965])
        expected_values = { # known good values for this model with rand seeded to standard
            249 : -1.924329E-03,
            441 : -3.037640E-02,
            514 : 4.416623E-02,
            615 : 4.362534E-02,
            620 : -7.850133E-03,
            650 : 5.465816E-04,
            794 : -3.219938E-02,
            909 : -6.297868E-03,
            929 : -3.697080E-02,
            965 : -7.405278E-03
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg16_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg16_bn)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [159, 199, 234, 253, 333, 584, 620, 634, 743, 859])
        expected_values = { # known good values for this model with rand seeded to standard
            159 : 1.883547E-02,
            199 : 1.646904E-02,
            234 : -1.799851E-02,
            253 : 1.224894E-03,
            333 : -4.926957E-04,
            584 : -2.378101E-02,
            620 : -7.849582E-03,
            634 : -3.253389E-02,
            743 : 4.304915E-02,
            859 : -2.494348E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg19(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg19)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [54, 56, 93, 333, 439, 552, 562, 628, 785, 837])
        expected_values = { # known good values for this model with rand seeded to standard
            54 : 1.092859E-02,
            56 : -1.234227E-02,
            93 : 2.286904E-02,
            333 : 8.079894E-04,
            439 : -2.807981E-02,
            552 : 4.727743E-02,
            562 : 2.900094E-02,
            628 : -1.278550E-02,
            785 : 2.089911E-02,
            837 : -1.127857E-02
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg19_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg19_bn)
        # print(inspect.stack()[0][3])
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [118, 297, 390, 396, 475, 494, 515, 578, 725, 940])
        expected_values = { # known good values for this model with rand seeded to standard
            118 : 1.980081E-02,
            297 : 1.414855E-02,
            390 : 2.228560E-02,
            396 : 8.866569E-03,
            475 : 5.065463E-05,
            494 : -6.476291E-02,
            515 : -3.320419E-02,
            578 : -9.235823E-02,
            725 : 2.133613E-02,
            940 : -3.484163E-02
        }
        self._test_classification_vgg(model, expected_values)


class SqueezeNetTester(TorchVisionTester): # run time ~4s
    # num_classes=1000
    # TODO might be worth testing args width_per_group, replace_stride_with_dilation, norm_layer, groups, zero_init_residual
    def _test_classification_squeezenet(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_squeezenet1_0(self):
        # num_classes=1000
        # NOTE seeing straight-up zeros in the expected values surprised me
        model = self._get_test_model(models.squeezenet1_0)
        self._check_scriptable(model, True)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [89, 333, 337, 373, 520, 528, 571, 638, 707, 821])
        expected_values = { # known good values for this model with rand seeded to standard
            89 : 0.000000E+00,
            333 : 0.000000E+00,
            337 : 0.000000E+00,
            373 : 7.850152E-02,
            520 : 7.918072E-02,
            528 : 0.000000E+00,
            571 : 3.812587E-03,
            638 : 1.504575E-01,
            707 : 0.000000E+00,
            821 : 2.733751E-01
        }
        self._test_classification_squeezenet(model, expected_values)

    def test_classification_squeezenet1_1(self):
        # num_classes=1000
        # NOTE seeing straight-up zeros in the expected values surprised me
        # NOTE no scriptability check specified, but this passes
        model = self._get_test_model(models.squeezenet1_1)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [87, 102, 428, 494, 635, 723, 883, 900, 932, 970])
        expected_values = { # known good values for this model with rand seeded to standard
            87 : 5.559896E-03,
            102 : 2.049656E-01,
            428 : 2.592387E-02,
            494 : 7.646167E-02,
            635 : 5.003151E-01,
            723 : 1.846589E-01,
            883 : 1.804985E-03,
            900 : 3.174357E-01,
            932 : 4.590575E-02,
            970 : 1.557179E-01
        }
        self._test_classification_squeezenet(model, expected_values)

class InceptionTester(TorchVisionTester): # run time ~18s
    def test_classification_inception_v3(self):
        INCEPTION_INPUT_SHAPE = (1, 3, 299, 299)
        # num_classes=1000
        # NOTE should we test aux_logits=True, transform_input=False?
        model = self._get_test_model(models.inception_v3)
        model.eval()
        test_input = self._get_test_input(INCEPTION_INPUT_SHAPE)

        self._check_scriptable(model, False)

        # TODO for whatever reason, this was not running deterministically - will fix others and come back
        # NOTE values are also really huge, not the usual -1 < x < 1
        # NOTE The issue is not rand dropout. InceptionV3 *does* use F.dropout where everyone else uses nn.Dropout,
        #      But changing to nn or even removing the dropout layer doesn't make the run deterministic.
        # NOTE When I create the model in the Python REPL, and run the *same* tensor through it more than once,
        #      it looks like it's deterministic. Maybe it's the scipy PRNG? No idea.
        # NOTE I checked the test input, it's being generated deterministically every time.
        # NOTE This is interesting: The model is deterministic within a running process - i.e., if I execute the
        #      test multiple times within the same run, it works. It gives a consistent answer within a run,
        #      but it's a different answer for every run. I suspect the scipy PRNG is involved.
        #   
        # self._build_correctness_check(model, INCEPTION_INPUT_SHAPE, [253, 261, 318, 401, 480, 562, 675, 771, 842, 890])
        expected_values = { # known good values for this model with rand seeded to standard
            253 : 2.167112E+08,
            261 : 3.270227E+07,
            318 : -2.352469E+08,
            401 : 1.535243E+08,
            480 : 2.587064E+08,
            562 : 2.433380E+08,
            675 : 8.990869E+07,
            771 : 2.878284E+08,
            842 : -1.568332E+08,
            890 : -2.132923E+08
        }
        self._check_model_correctness(model, test_input, expected_values, 1000)

class GoogleNetTester(TorchVisionTester):
    def test_classification_googlenet(self):
        # num_classes=1000
        # NOTE should we test aux_logits=True, transform_input=False?
        model = self._get_test_model(models.googlenet)
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)

        self._check_scriptable(model, False)
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [153, 264, 378, 518, 562, 654, 684, 747, 823, 843])
        expected_values = { # known good values for this model with rand seeded to standard
            153 : -1.654747E-02,
            264 : -1.643153E-02,
            378 : 2.358393E-02,
            518 : 2.069115E-02,
            562 : 1.702593E-02,
            654 : 1.346865E-02,
            684 : -2.816701E-02,
            747 : -5.827621E-03,
            823 : 3.103274E-02,
            843 : 2.653003E-02
        }
        self._check_model_correctness(model, test_input, expected_values, 1000)


class MobileNetTester(TorchVisionTester):
    def test_classification_mobilenet_v2(self):
        # num_classes=1000
        # NOTE should we test width_mult=1.0, inverted_residual_setting=None, round_nearest=8?
        # NOTE something is up here - expected values all 0.0
        model = self._get_test_model(models.mobilenet_v2)
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)

        self._check_scriptable(model, True)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [2, 115, 211, 222, 416, 562, 757, 900, 918, 984])
        expected_values = { # known good values for this model with rand seeded to standard
            2 : 5.376910E-11,
            115 : 5.723687E-10,
            211 : 5.142162E-10,
            222 : 7.450356E-10,
            416 : 1.624237E-10,
            562 : -6.263705E-10,
            757 : 1.555139E-10,
            900 : 7.528030E-10,
            918 : 1.139322E-09,
            984 : -1.157643E-09
        }
        self._check_model_correctness(model, test_input, expected_values, 1000)


class MNASNetTester(TorchVisionTester):
    # num_classes=1000
    # NOTE should we test dropout=0.2?
    # NOTE no scriptability check specified
    # NOTE something is up here - expected values all 0.0
    def _test_classification_mnas(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_mnasnet0_5(self):
        model = self._get_test_model(models.mnasnet0_5)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [124, 249, 272, 287, 306, 409, 494, 505, 569, 959])
        expected_values = { # known good values for this model with rand seeded to standard
            124 : 2.703910E-07,
            249 : -1.089239E-07,
            272 : 2.899310E-07,
            287 : 1.193933E-07,
            306 : -2.121729E-07,
            409 : -3.381995E-08,
            494 : -6.292279E-08,
            505 : 2.337501E-07,
            569 : 2.391237E-07,
            959 : -4.019812E-08
        }
        self._test_classification_mnas(model, expected_values)


    def test_classification_mnasnet0_75(self):
        model = self._get_test_model(models.mnasnet0_75)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [56, 62, 304, 330, 380, 388, 434, 443, 550, 579])
        expected_values = { # known good values for this model with rand seeded to standard
            56 : 1.806811E-09,
            62 : -3.786029E-08,
            304 : 8.238560E-08,
            330 : -7.117174E-08,
            380 : -2.559868E-07,
            388 : 2.254940E-08,
            434 : 6.211397E-08,
            443 : -2.445332E-07,
            550 : -9.940627E-08,
            579 : 1.639630E-07
        }
        self._test_classification_mnas(model, expected_values)

    def test_classification_mnasnet1_0(self):
        model = self._get_test_model(models.mnasnet1_0)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [252, 287, 291, 297, 351, 384, 542, 653, 738, 829])
        expected_values = { # known good values for this model with rand seeded to standard
            252 : -2.228216E-08,
            287 : 9.844145E-08,
            291 : 6.898448E-08,
            297 : 1.076740E-07,
            351 : -2.434381E-08,
            384 : -1.197281E-07,
            542 : 2.228182E-08,
            653 : -5.830330E-09,
            738 : 9.718425E-09,
            829 : 6.893035E-08
        }
        self._test_classification_mnas(model, expected_values)

    def test_classification_mnasnet1_3(self):
        model = self._get_test_model(models.mnasnet1_3)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [96, 207, 397, 552, 654, 750, 751, 788, 822, 827])
        expected_values = { # known good values for this model with rand seeded to standard
            96 : -4.894060E-09,
            207 : 1.960604E-08,
            397 : 1.658925E-08,
            552 : -2.500858E-08,
            654 : 3.843252E-08,
            750 : -4.504147E-08,
            751 : 2.236427E-08,
            788 : -1.602543E-08,
            822 : 1.778895E-08,
            827 : 7.287996E-08
        }
        self._test_classification_mnas(model, expected_values)


class ShuffleNetTester(TorchVisionTester):
    def _test_classification_shufflenet(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_shufflenet_v2_x0_5(self):
        model = self._get_test_model(models.shufflenet_v2_x0_5)
        # NOTE no scriptability check specified

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [121, 279, 296, 589, 633, 667, 835, 921, 923, 990])
        expected_values = { # known good values for this model with rand seeded to standard
            121 : 6.365352E-03,
            279 : 2.777028E-02,
            296 : 1.771086E-02,
            589 : 7.464714E-03,
            633 : -4.323924E-03,
            667 : 2.060407E-02,
            835 : 2.895552E-02,
            921 : -4.661043E-03,
            923 : -1.933447E-02,
            990 : 2.745904E-02
        }
        self._test_classification_shufflenet(model, expected_values)

    def test_classification_shufflenet_v2_x1_0(self):
        model = self._get_test_model(models.shufflenet_v2_x1_0)
        self._check_scriptable(model, True) # Failing!

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [50, 187, 438, 501, 518, 579, 588, 640, 667, 968])
        expected_values = { # known good values for this model with rand seeded to standard
            50 : -2.583135E-03,
            187 : -5.233090E-03,
            438 : 2.091412E-02,
            501 : -5.121271E-03,
            518 : 1.609881E-02,
            579 : -8.247387E-03,
            588 : -1.574289E-02,
            640 : 2.507384E-02,
            667 : 1.039669E-02,
            968 : -9.486280E-03
        }
        self._test_classification_shufflenet(model, expected_values)

    def test_classification_shufflenet_v2_x1_5(self):
        model = self._get_test_model(models.shufflenet_v2_x1_5)
        # NOTE no scriptability check specified

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [75, 209, 239, 244, 285, 379, 511, 657, 744, 767])
        expected_values = { # known good values for this model with rand seeded to standard
            75 : 1.470629E-02,
            209 : -1.318278E-02,
            239 : 5.034821E-03,
            244 : 2.056844E-02,
            285 : -1.494422E-02,
            379 : 8.534319E-03,
            511 : -2.769079E-02,
            657 : -3.036056E-02,
            744 : 2.080933E-04,
            767 : 1.973816E-02
        }
        self._test_classification_shufflenet(model, expected_values)

    def test_classification_shufflenet_v2_x2_0(self):
        model = self._get_test_model(models.shufflenet_v2_x2_0)
        # NOTE no scriptability check specified

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [110, 364, 458, 497, 542, 663, 694, 812, 880, 971])
        expected_values = { # known good values for this model with rand seeded to standard
            110 : -1.394911E-02,
            364 : 9.127663E-03,
            458 : 9.575198E-03,
            497 : -5.239639E-03,
            542 : -4.049195E-03,
            663 : 5.255685E-03,
            694 : 2.597559E-03,
            812 : 1.349113E-02,
            880 : 2.012882E-02,
            971 : 5.573011E-03
        }
        self._test_classification_shufflenet(model, expected_values)


class DenseNetTester(TorchVisionTester):
    # num_classes = 1000
    # NOTE growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0 ?
    # TODO the memory efficient ctor flag is tested in a model yet to be moved to this object
    def _test_classification_densenet(self, model, expected_values):
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        self._check_model_correctness(model, test_input, expected_values, 1000)

    def test_classification_densenet121(self):
        model = self._get_test_model(models.densenet121)
        self._check_scriptable(model, False)

        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [11, 82, 325, 346, 423, 567, 575, 745, 963, 978])
        expected_values = { # known good values for this model with rand seeded to standard
            11 : 4.605587E-01,
            82 : 2.790979E-01,
            325 : -6.546915E-01,
            346 : 4.518642E-01,
            423 : -1.023913E-01,
            567 : -4.808005E-01,
            575 : 3.504707E-01,
            745 : -1.296859E-01,
            963 : -1.144047E-01,
            978 : -3.576259E-02
        }
        self._test_classification_densenet(model, expected_values)

    def test_classification_densenet161(self):
        model = self._get_test_model(models.densenet161)
        # NOTE no scriptability check specified
        
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [114, 137, 367, 389, 394, 434, 599, 669, 837, 950])
        expected_values = { # known good values for this model with rand seeded to standard
            114 : -5.713884E-01,
            137 : -4.074038E-01,
            367 : 4.238620E-02,
            389 : 5.561393E-01,
            394 : -3.722799E-02,
            434 : -5.974094E-01,
            599 : -1.717324E-01,
            669 : 1.266117E-01,
            837 : -2.939904E-01,
            950 : 4.458589E-01
        }
        self._test_classification_densenet(model, expected_values)

    def test_classification_densenet169(self):
        model = self._get_test_model(models.densenet169)
        # NOTE no scriptability check specified
        
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [37, 258, 319, 440, 479, 547, 829, 836, 946, 976])
        expected_values = { # known good values for this model with rand seeded to standard
            37 : -8.443325E-01,
            258 : -6.752821E-01,
            319 : -4.467736E-01,
            440 : 8.621871E-01,
            479 : -1.954675E-03,
            547 : 2.341446E-01,
            829 : -5.012124E-01,
            836 : 1.903470E-01,
            946 : -6.522099E-01,
            976 : 9.011360E-01
        }
        self._test_classification_densenet(model, expected_values)

    def test_classification_densenet201(self):
        model = self._get_test_model(models.densenet201)
        # NOTE no scriptability check specified
        
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [42, 93, 146, 170, 495, 512, 588, 783, 915, 952])
        expected_values = { # known good values for this model with rand seeded to standard
            42 : -1.090873E-01,
            93 : -3.196294E-01,
            146 : -2.306343E-01,
            170 : 3.833550E-01,
            495 : 6.323808E-01,
            512 : -2.975282E-01,
            588 : 4.006727E-01,
            783 : -6.624119E-01,
            915 : 4.383706E-01,
            952 : 2.573722E-01
        }
        self._test_classification_densenet(model, expected_values)

#################################################################
#################################################################
#################################################################

class YetToBeFixed:

    def test_memory_efficient_densenet(self):
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)

        for name in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
            model1 = models.__dict__[name](num_classes=50, memory_efficient=True)
            params = model1.state_dict()
            model1.eval()
            out1 = model1(x)
            out1.sum().backward()

            model2 = models.__dict__[name](num_classes=50, memory_efficient=False)
            model2.load_state_dict(params)
            model2.eval()
            out2 = model2(x)

            max_diff = (out1 - out2).abs().max()

            self.assertTrue(max_diff < 1e-5)

    def test_resnet_dilation(self):
        # TODO improve tests to also check that each layer has the right dimensionality
        for i in product([False, True], [False, True], [False, True]):
            model = models.__dict__["resnet50"](replace_stride_with_dilation=i)
            model = self._make_sliced_model(model, stop_layer="layer4")
            model.eval()
            x = torch.rand(1, 3, 224, 224)
            out = model(x)
            f = 2 ** sum(i)
            self.assertEqual(out.shape, (1, 2048, 7 * f, 7 * f))

    def test_mobilenetv2_residual_setting(self):
        model = models.__dict__["mobilenet_v2"](inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape[-1], 1000)

    ##
    # Old helpers that do some standard stuff, depending on the nature of the model
    #

    def _test_segmentation_model(self, name):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.segmentation.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(tuple(out["out"].shape), (1, 50, 300, 300))

    def _test_detection_model(self, name):
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])

    def _test_video_model(self, name):
        # the default input shape is
        # bs * num_channels * clip_len * h *w
        input_shape = (1, 3, 4, 112, 112)
        # test both basicblock and Bottleneck
        model = models.video.__dict__[name](num_classes=50)
        self.check_script(model, name)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)

    def _make_sliced_model(self, model, stop_layer):
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            if name == stop_layer:
                break
        new_model = torch.nn.Sequential(layers)
        return new_model


for model_name in []: # get_available_segmentation_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_segmentation_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name in []: # get_available_detection_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_detection_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name in []: # get_available_video_models():

    def do_test(self, model_name=model_name):
        self._test_video_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)

if __name__ == '__main__':
    unittest.main()
#    for model_name in get_available_classification_models():
#        print('    def test_classification_{}(self):'.format(model_name))
#        print('        self._test_classification_model({}, STANDARD_INPUT_SHAPE)'.format(model_name))
#        print('')
