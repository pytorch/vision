<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# torchvision

[![total torchvision downloads](https://pepy.tech/badge/torchvision)](https://pepy.tech/project/torchvision)
[![documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/vision/stable/index.html)

`torchvision` 包含了计算机视觉领域广泛使用的热门数据集、主流模型架构以及常用的图像数据变换算子。

## 安装指南

请参阅[官方安装指南](https://pytorch.org/get-started/locally/)在您的系统上安装稳定版本的 `torch` 和 `torchvision`。

若需从源码构建，请参阅我们的[贡献指南页面](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation)。

以下是 `torchvision` 版本与支持的 Python 版本对照表：

| `torch`            | `torchvision`      | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.10`, `<=3.14`  |
| `2.13`             | `0.28`             | `>=3.10`, `<=3.14`  |
| `2.12`             | `0.27`             | `>=3.10`, `<=3.14`  |
| `2.11`             | `0.26`             | `>=3.10`, `<=3.14`  |
| `2.10`             | `0.25`             | `>=3.10`, `<=3.14`  |


<details>
    <summary>历史版本</summary>

| `torch` | `torchvision`     | Python                    |
|---------|-------------------|---------------------------|
| `2.9`              | `0.24`             | `>=3.10`, `<=3.14`  |
| `2.8`              | `0.23`             | `>=3.9`, `<=3.13`   |
| `2.7`              | `0.22`             | `>=3.9`, `<=3.13`   |
| `2.6`              | `0.21`             | `>=3.9`, `<=3.12`   |
| `2.5`              | `0.20`             | `>=3.9`, `<=3.12`   |
| `2.4`              | `0.19`             | `>=3.8`, `<=3.12`   |
| `2.3`              | `0.18`             | `>=3.8`, `<=3.12`   |
| `2.2`              | `0.17`             | `>=3.8`, `<=3.11`   |
| `2.1`              | `0.16`             | `>=3.8`, `<=3.11`   |
| `2.0`              | `0.15`             | `>=3.8`, `<=3.11`   |
| `1.13`  | `0.14`            | `>=3.7.2`, `<=3.10`       |
| `1.12`  | `0.13`            | `>=3.7`, `<=3.10`         |
| `1.11`  | `0.12`            | `>=3.7`, `<=3.10`         |
| `1.10`  | `0.11`            | `>=3.6`, `<=3.9`          |
| `1.9`   | `0.10`            | `>=3.6`, `<=3.9`          |
| `1.8`   | `0.9`             | `>=3.6`, `<=3.9`          |
| `1.7`   | `0.8`             | `>=3.6`, `<=3.9`          |
| `1.6`   | `0.7`             | `>=3.6`, `<=3.8`          |
| `1.5`   | `0.6`             | `>=3.5`, `<=3.8`          |
| `1.4`   | `0.5`             | `==2.7`, `>=3.5`, `<=3.8` |
| `1.3`   | `0.4.2` / `0.4.3` | `==2.7`, `>=3.5`, `<=3.7` |
| `1.2`   | `0.4.1`           | `==2.7`, `>=3.5`, `<=3.7` |
| `1.1`   | `0.3`             | `==2.7`, `>=3.5`, `<=3.7` |
| `<=1.0` | `0.2`             | `==2.7`, `>=3.5`, `<=3.7` |

</details>

## 图像处理后端

Torchvision 目前支持以下图像处理后端：

- torch 张量 (torch tensors)
- PIL 图像：
    - [Pillow](https://python-pillow.org/)
    - [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) - 基于 SIMD 硬件指令集加速的 Pillow 即插即用高性能替代方案（速度大幅提升）。

更多信息请阅读我们的[官方变换文档](https://pytorch.org/vision/stable/transforms.html)。

## 文档

您可以在 PyTorch 官方网站查阅完整的 API 文档：<https://pytorch.org/vision/stable/index.html>

## 参与贡献

关于如何参与贡献，请参阅 [CONTRIBUTING](CONTRIBUTING.md) 文件。

## 数据集免责声明

本项目是一个下载并预处理公开数据集的工具库。我们并不托管或分发这些数据集，亦不对其质量、公平性做任何担保，更不声明您拥有使用该数据集的许可。确定您是否有权根据该数据集的许可证使用该数据集，完全是您自身的责任。

如果您是数据集所有者，并希望更新其中的任何部分（描述、引用文献等），或者不希望您的高价值数据集包含在本库中，请通过 GitHub Issue 与我们联系。感谢您对机器学习社区所做出的贡献！

## 预训练模型许可证

本库中提供的预训练模型可能具有根据训练所用数据集衍生的专属许可证或条款与条件。确定您是否有权在具体的应用场景中使用这些模型，完全是您自身的责任。

具体而言，SWAG 模型基于 CC-BY-NC 4.0 许可证发布。有关更多详情，请参阅 [SWAG LICENSE](https://github.com/facebookresearch/SWAG/blob/main/LICENSE)。

## 引用 TorchVision

如果您在工作或研究中发现 TorchVision 对您有所帮助，请考虑引用以下 BibTeX 条目：

```bibtex
@software{torchvision2016,
    title        = {TorchVision: PyTorch's Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
```

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）翻译维护，最后同步更新于 2026年09月28日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
