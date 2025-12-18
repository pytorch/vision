# torchvision

[![total torchvision downloads](https://pepy.tech/badge/torchvision)](https://pepy.tech/project/torchvision)
[![documentation](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Ftorchvision%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/vision/stable/index.html)

torchvision paketi; popüler veri kümeleri, model mimarileri ve bilgisayarlı görü için yaygın görüntü dönüşümlerinden oluşur.

## Kurulum

Sisteminize `torch` ve `torchvision` paketlerinin kararlı sürümlerini kurmak için lütfen [resmi talimatlara](https://pytorch.org/get-started/locally/) başvurun.

Kaynak koddan derlemek için [katkıda bulunma sayfamıza](https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation) göz atın.

Aşağıda, ilgili `torchvision` sürümleri ve desteklenen Python sürümleri yer almaktadır.

---

Bu çevirinin altına eklemek istediğiniz bir versiyon tablosu veya iletişim notu var mı? İsterseniz bu teknik metni projenizin genel duyurusuna uygun şekilde daha da detaylandırabilirim.

| `torch`            | `torchvision`      | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.10`, `<=3.14`  |
| `2.9`              | `0.24`             | `>=3.10`, `<=3.14`  |
| `2.8`              | `0.23`             | `>=3.9`, `<=3.13`   |
| `2.7`              | `0.22`             | `>=3.9`, `<=3.13`   |
| `2.6`              | `0.21`             | `>=3.9`, `<=3.12`   |

<details>
    <summary>eskki versiyon</summary>

| `torch` | `torchvision`     | Python                    |
|---------|-------------------|---------------------------|
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

## Görüntü Arka Uçları (Image Backends)

Torchvision şu anda aşağıdaki görüntü arka uçlarını desteklemektedir:

* torch tensor'ları
* PIL görüntüleri:
* [Pillow](https://python-pillow.org/)
* [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) - SIMD ile Pillow için **çok daha hızlı**, doğrudan değiştirilebilir bir alternatif.



Daha fazlasını [dokümantasyonumuzdan](https://pytorch.org/vision/stable/transforms.html) okuyabilirsiniz.

## Dokümantasyon

API dokümantasyonunu PyTorch web sitesinde bulabilirsiniz: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)

## Katkıda Bulunma

Nasıl yardımcı olabileceğinizi öğrenmek için [CONTRIBUTING](https://www.google.com/search?q=CONTRIBUTING.md) dosyasına bakın.

## Veri Kümeleri Hakkında Feragatname

Bu, halka açık veri kümelerini indiren ve hazırlayan bir yardımcı kütüphanedir. Bu veri kümelerini barındırmıyoruz veya dağıtmıyoruz, kalitelerine veya tarafsızlıklarına kefil olmuyoruz veya veri kümesini kullanma lisansınız olduğunu iddia etmiyoruz. Veri kümesini, veri kümesinin lisansı kapsamında kullanma izniniz olup olmadığını belirlemek sizin sorumluluğunuzdadır.

Bir veri kümesi sahibiyseniz ve herhangi bir bölümünü güncellemek istiyorsanız (açıklama, alıntı vb.) veya veri kümenizin bu kütüphaneye dahil edilmesini istemiyorsanız, lütfen bir GitHub issue'su aracılığıyla iletişime geçin. ML topluluğuna katkılarınız için teşekkürler!

## Önceden Eğitilmiş Model Lisansı

Bu kütüphanede sağlanan önceden eğitilmiş modellerin, eğitim için kullanılan veri kümesinden türetilen kendi lisansları veya şart ve koşulları olabilir. Modelleri kendi kullanım durumunuz için kullanma izniniz olup olmadığını belirlemek sizin sorumluluğunuzdadır.

Daha spesifik olarak, SWAG modelleri CC-BY-NC 4.0 lisansı altında yayınlanmıştır. Ek detaylar için [SWAG LİSANSI](https://github.com/facebookresearch/SWAG/blob/main/LICENSE) dosyasına bakın.

## TorchVision'a Atıfta Bulunma

Çalışmalarınızda TorchVision'ı faydalı buluyorsanız, lütfen aşağıdaki BibTeX girişini atıfta bulunmayı düşünün:

---

Bu teknik dokümantasyonun devamında yer alan BibTeX kod bloğunu da çevirmemi veya formatlamamı ister misiniz?

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
