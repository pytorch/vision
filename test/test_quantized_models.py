import torch
import unittest
import torchvision


class Tester(unittest.TestCase):
    def test_model_quantizable(self):
        image = torch.rand(1, 3, 224, 224, dtype=torch.float32)
        model_arch_list = ['mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d'
                          ]
        for eval in [True, False]:
            for arch in model_arch_list:
                model = torchvision.models.quantization.__dict__[arch](pretrained=False)
                print("Creating model", arch)
                if eval:
                    model.eval()
                    model.qconfig = torch.quantization.default_qconfig
                else:
                    model.train()
                    model.qconfig = torch.quantization.default_qat_qconfig

                model.fuse_model()
                if eval:
                    torch.quantization.prepare(model, inplace=True)
                else:
                    torch.quantization.prepare_qat(model, inplace=True)
                    model.eval()
                torch.quantization.convert(model, inplace=True)
                # Ensure that quantized model runs successfully
                model(image)

    def test_model_scriptability(self):
        model_arch_list = ['mobilenet_v2','resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
                          ]
        for arch in model_arch_list:
            model = torchvision.models.quantization.__dict__[arch](pretrained=False)
            print("Creating model", arch)
            model.eval()
            model.qconfig = torch.quantization.default_qconfig

            model.fuse_model()
            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)
            # Ensure that quantized model is scriptable
            torch.jit.script(model)


if __name__ == '__main__':
    unittest.main()
