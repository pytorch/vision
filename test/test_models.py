from collections import OrderedDict
from itertools import product
import torch
from torchvision import models
import unittest
import math



def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
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
        torch.random.manual_seed(STANDARD_SEED)
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
        abserr = abs(expected - actual)
        if not ((abserr / expected) < tolerance) and False:
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
    def _build_correctness_check(self, model, input_shape, indices):
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
        expected_values = { # known good values for this model with rand seeded to standard
            130 : 0.019345,
            257 : -0.002852,
            313 : 0.019647,
            361 : -0.006478,
            466 : 0.011666,
            525 : 0.009539,
            537 : 0.01841,
            606 : 0.003135,
            667 : 0.004638,
            945 : -0.014482
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
        expected_values = { # known good values for this model with rand seeded to standard
            65 : -0.115954,
            172 : 0.139294,
            195 : 1.248264,
            241 : -1.769466,
            319 : -0.237925,
            333 : -0.038517,
            538 : -0.346574,
            546 : 0.364637,
            763 : 0.43461,
            885 : -1.386981
        }
        self._test_classification_resnet(model, expected_values)


    def test_classification_resnet34(self):
        # NOTE passes scriptability check, but none was previously specified
        model = self._get_test_model(models.resnet34)
        expected_values = { # known good values for this model with rand seeded to standard
            69 : -0.073099,
            328 : 10.624151,
            387 : -31.548423,
            391 : -14.790843,
            560 : 3.853342,
            601 : 2.775564,
            631 : 44.730876,
            795 : 7.870284,
            875 : 12.352467,
            960 : -12.154144
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet50(self):
        # NOTE fails scriptability check, but none was previously specified
        model = self._get_test_model(models.resnet50)
        expected_values = { # known good values for this model with rand seeded to standard
            44 : 12.24592,
            91 : 9.113415,
            238 : -7.919643,
            260 : 1.651342,
            263 : 3.42577,
            391 : -1.186231,
            416 : 0.128767,
            648 : -8.874666,
            700 : 13.21073,
            883 : -10.211411
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet101(self):
        # NOTE fails scriptability check, but none was previously specified
        # NOTE weird, high output values compared to some others in resnet family
        model = self._get_test_model(models.resnet101)
        expected_values = { # known good values for this model with rand seeded to standard
            291 : -12148.84375,
            303 : -4350.053222,
            360 : 911.205444,
            429 : -4101.465332,
            509 : -10198.37207,
            558 : 2845.450683,
            743 : -604.379882,
            747 : 3814.401611,
            824 : 2433.574707,
            856 : 10094.262695
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnet152(self):
        # NOTE fails scriptability check, but none was previously specified
        # NOTE weird, high output values compared to some others in resnet family
        model = self._get_test_model(models.resnet152)
        expected_values = { # known good values for this model with rand seeded to standard
            37 : -6643550.0,
            125 : -26183394.0,
            240 : -17213840.0,
            555 : 1835685.0,
            573 : 21868084.0,
            644 : 16231895.0,
            761 : 19799108.0,
            794 : 13369855.0,
            805 : -7568018.0,
            853 : 7857503.5
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnext50_32x4d(self):
        model = self._get_test_model(models.resnext50_32x4d)
        self._check_scriptable(model, False)
        expected_values = { # known good values for this model with rand seeded to standard
            10 : 0.035699,
            409 : 0.013995,
            515 : -0.039913,
            548 : -0.049931,
            566 : 0.000606,
            589 : 0.037367,
            684 : -0.048179,
            767 : -0.004519,
            870 : -0.063937,
            877 : -0.036286
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_resnext101_32x8d(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.resnext101_32x8d)
        expected_values = { # known good values for this model with rand seeded to standard
            264 : 0.002714,
            274 : -0.061351,
            366 : -0.096291,
            378 : -0.05945,
            471 : -0.001634,
            534 : -0.021004,
            544 : 0.028956,
            584 : 0.058926,
            742 : 0.055629,
            822 : -0.02232
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_wide_resnet50_2(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.wide_resnet50_2)
        expected_values = { # known good values for this model with rand seeded to standard
            18 : 3.126974,
            39 : -3.925597,
            132 : -4.264346,
            174 : -7.149744,
            307 : 19.028724,
            514 : 0.139877,
            530 : -1.253244,
            683 : -21.184637,
            702 : -3.710342,
            748 : 0.577609
        }
        self._test_classification_resnet(model, expected_values)

    def test_classification_wide_resnet101_2(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.wide_resnet101_2)
        expected_values = { # known good values for this model with rand seeded to standard
            63 : 2206.879882,
            147 : 2459.263916,
            238 : -27444.535156,
            408 : 2603.14624,
            478 : 1474.905883,
            756 : 94.388496,
            773 : 8560.335937,
            927 : -13348.708984,
            981 : 4570.083496,
            994 : -924.104736
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
        expected_values = { # known good values for this model with rand seeded to standard
            12 : -0.023673,
            150 : -0.030705,
            262 : 0.05549,
            262 : 0.05549,
            422 : -0.006336,
            501 : 0.008839,
            731 : 0.002569,
            750 : -0.020255,
            939 : 0.078099,
            942 : 0.012279
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg11_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg11_bn)
        expected_values = { # known good values for this model with rand seeded to standard
            88 : 0.013356,
            136 : -0.025611,
            323 : 0.08224,
            335 : -0.022072,
            343 : 0.024697,
            350 : 0.047756,
            384 : -0.004922,
            640 : 0.008104,
            687 : 7.1e-05,
            823 : 0.032322
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg13(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg13)
        expected_values = { # known good values for this model with rand seeded to standard
            29 : 0.019654,
            145 : -0.00043,
            254 : -6.7e-05,
            352 : 0.0138,
            400 : 0.024247,
            508 : 0.026169,
            845 : 0.016207,
            897 : -0.013479,
            928 : 0.002154,
            997 : 0.019514
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg13_bn(self):
        # NOTE no scriptability check specified
        # NOTE this also passes using the expected values from vgg13 - is that expected?
        #      not the case with vgg11[_bn] (though some values were very close, just not within EPSILON)
        model = self._get_test_model(models.vgg13_bn)
        expected_values = { # known good values for this model with rand seeded to standard
            317 : 0.040705,
            454 : 0.015814,
            522 : -0.007018,
            556 : 0.01947,
            609 : -0.039502,
            699 : -0.016666,
            807 : 0.033923,
            819 : 0.039569,
            834 : -0.06542,
            931 : -0.000928
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg16(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg16)
        expected_values = { # known good values for this model with rand seeded to standard
            1 : 0.010134,
            42 : -0.022099,
            221 : 0.092082,
            512 : 0.058425,
            552 : -0.024713,
            664 : 0.018183,
            689 : -0.000681,
            700 : 0.003135,
            909 : -0.006297,
            982 : -0.071333
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg16_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg16_bn)
        expected_values = { # known good values for this model with rand seeded to standard
            88 : -0.061056,
            136 : -0.059298,
            323 : 0.015986,
            335 : 0.020001,
            343 : -0.030813,
            350 : -0.024766,
            384 : -0.028537,
            640 : 0.043581,
            687 : 0.01704,
            823 : -0.00098
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg19(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg19)
        expected_values = { # known good values for this model with rand seeded to standard
            327 : -0.0021,
            367 : 0.006371,
            670 : 0.047663,
            763 : 0.009797,
            848 : -0.00922,
            864 : -0.03593,
            877 : -0.032121,
            902 : 0.030292,
            933 : -0.038277,
            970 : -0.033321
        }
        self._test_classification_vgg(model, expected_values)

    def test_classification_vgg19_bn(self):
        # NOTE no scriptability check specified
        model = self._get_test_model(models.vgg19_bn)
        # self._build_correctness_check(model, STANDARD_INPUT_SHAPE, [5, 202, 264, 307, 313, 317, 467, 543, 631, 721])
        expected_values = { # known good values for this model with rand seeded to standard
            5 : 2.132084E-03,
            202 : -1.154007E-02,
            264 : 6.224869E-03,
            307 : 2.161325E-02,
            313 : -4.450932E-02,
            317 : -4.523434E-02,
            467 : 1.715319E-02,
            543 : 2.472959E-02,
            631 : 4.078178E-02,
            721 : 6.985172E-02
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
        expected_values = { # known good values for this model with rand seeded to standard
            277 : 0.0,
            692 : 0.007344,
            721 : 0.0,
            851 : 0.000918,
            871 : 0.11461,
            880 : 0.00015,
            934 : 0.000121,
            937 : 0.010954,
            966 : 0.008219,
            991 : 0.0
        }
        self._test_classification_squeezenet(model, expected_values)

    def test_classification_squeezenet1_1(self):
        # num_classes=1000
        # NOTE seeing straight-up zeros in the expected values surprised me
        # NOTE no scriptability check specified, but this passes
        model = self._get_test_model(models.squeezenet1_1)

        expected_values = { # known good values for this model with rand seeded to standard
            42 : 0.005651,
            127 : 0.002385,
            140 : 0.402233,
            321 : 0.0,
            520 : 0.0,
            663 : 0.05538,
            687 : 0.000343,
            822 : 0.180019,
            957 : 0.0,
            967 : 0.0
        }
        self._test_classification_squeezenet(model, expected_values)

class InceptionTester(TorchVisionTester): # run time ~18s
    def test_classification_inception_v3(self):
        INCEPTION_INPUT_SHAPE = (1, 3, 299, 299)
        # num_classes=1000
        # NOTE should we test aux_logits=True, transform_input=False?
        model = self._get_test_model(models.inception_v3)
        test_input = self._get_test_input(INCEPTION_INPUT_SHAPE)

        self._check_scriptable(model, False)

        # TODO for whatever reason, this was not running deterministically - will fix others and come back
        # NOTE values are also really huge, not the usual -1 < x < 1
        # NOTE The issue is not rand dropout. InceptionV3 *does* use F.dropout where everyone else uses nn.Dropout,
        #      But changing to nn or even removing the dropout layer doesn't make the run deterministic.
        # self._build_correctness_check(model, INCEPTION_INPUT_SHAPE, [253, 261, 318, 401, 480, 562, 675, 771, 842, 890])
        expected_values = { # known good values for this model with rand seeded to standard
            253 : -1687277440.0,
            261 : 39273372.0,
            318 : 697125056.0,
            401 : 1686322432.0,
            480 : 435522368.0,
            562 : -2154610688.0,
            675 : -601827328.0,
            771 : -1603795072.0,
            842 : -1566286464.0,
            890 : -431068224.0
        }
        self._check_model_correctness(model, test_input, expected_values, 1000)

class GoogleNetTester(TorchVisionTester):
    def test_classification_googlenet(self):
        # num_classes=1000
        # NOTE should we test aux_logits=True, transform_input=False?
        model = self._get_test_model(models.googlenet)
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)

        self._check_scriptable(model, False)

        expected_values = { # known good values for this model with rand seeded to standard
            153 : -0.016547,
            264 : -0.016431,
            378 : 0.023583,
            518 : 0.020691,
            562 : 0.017025,
            654 : 0.013468,
            684 : -0.028167,
            747 : -0.005827,
            823 : 0.031032,
            843 : 0.02653
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
            2 : 1.0, # actual is 0.0, but i want to induce failure here until i understand the 0.0
            115 : 0.0,
            211 : 0.0,
            222 : 0.0,
            416 : 0.0,
            562 : 0.0,
            757 : 0.0,
            900 : 0.0,
            918 : 0.0,
            984 : 0.0
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
