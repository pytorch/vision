"""Run smoke tests"""

import os
from pathlib import Path
from sys import platform

import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

SCRIPT_DIR = Path(__file__).parent


def smoke_test_torchvision() -> None:
    print(
        "Is torchvision usable?",
        all(x is not None for x in [torch.ops.image.decode_png, torch.ops.torchvision.roi_align]),
    )


def smoke_test_torchvision_read_decode() -> None:
    img_jpg = read_image(str(SCRIPT_DIR / "assets" / "encode_jpeg" / "grace_hopper_517x606.jpg"))
    if img_jpg.ndim != 3 or img_jpg.numel() < 100:
        raise RuntimeError(f"Unexpected shape of img_jpg: {img_jpg.shape}")
    img_png = read_image(str(SCRIPT_DIR / "assets" / "interlaced_png" / "wizard_low.png"))
    if img_png.ndim != 3 or img_png.numel() < 100:
        raise RuntimeError(f"Unexpected shape of img_png: {img_png.shape}")


def smoke_test_compile() -> None:
    try:
        model = resnet50().cuda()
        model = torch.compile(model)
        x = torch.randn(1, 3, 224, 224, device="cuda")
        out = model(x)
        print(f"torch.compile model output: {out.shape}")
    except RuntimeError:
        if platform == "win32":
            print("Successfully caught torch.compile RuntimeError on win")
        else:
            raise


def smoke_test_torchvision_resnet50_classify(device: str = "cpu") -> None:
    img = read_image(str(SCRIPT_DIR / ".." / "gallery" / "assets" / "dog2.jpg")).to(device)

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    expected_category = "German shepherd"
    print(f"{category_name} ({device}): {100 * score:.1f}%")
    if category_name != expected_category:
        raise RuntimeError(f"Failed ResNet50 classify {category_name} Expected: {expected_category}")


def main() -> None:
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    smoke_test_torchvision()
    smoke_test_torchvision_read_decode()
    smoke_test_torchvision_resnet50_classify()
    if torch.cuda.is_available():
        smoke_test_torchvision_resnet50_classify("cuda")
        smoke_test_compile()

    if torch.backends.mps.is_available():
        smoke_test_torchvision_resnet50_classify("mps")


if __name__ == "__main__":
    main()
