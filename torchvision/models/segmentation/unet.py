from typing import Callable, List, Tuple

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class UnetFrameworkEncoderLayer(nn.Module):
    def __init__(self, feature_layer: nn.Module, projection_layer: nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.feature_layer = feature_layer
        self.projection_layer = projection_layer

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # a residual layer which computes the features that are going to be passed to the decoder
        residual = self.feature_layer(x)
        # a projection layer which takes in the residual layer and projects it for the next encoder layer
        x = self.projection_layer(residual)
        return x, residual


class UnetFrameworkDecoderLayer(nn.Module):
    def __init__(
        self,
        feature_layer: nn.Module,
        combination_layer: nn.Module,
        projection_layer: nn.Module,
        upsample_layer: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.feature_layer = feature_layer
        self.projection_layer = projection_layer
        self.upsample_layer = upsample_layer
        self.combination_layer = combination_layer

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # an upsample layer which processes the previous layer's output
        x = self.upsample_layer(x)
        # a combination layer which combines the upsampled features with the residual features
        x = self.combination_layer(x, residual)
        # a projection layer which can be used to manipulate the combined features
        x = self.projection_layer(x)
        # the decoder feature layer which computes the final output
        x = self.feature_layer(x)
        return x


class UnetFrameworkEncoder(nn.Module):
    def __init__(self, layers: List[UnetFrameworkEncoderLayer], *args, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # a chain of encoder layers
        features, residuals = [], []
        for layer in self.layers:
            x, residual = layer(x)
            features.append(x)
            residuals.append(residual)
        return features, residuals


class UnetFrameworkDecoder(nn.Module):
    def __init__(self, layers: List[UnetFrameworkDecoderLayer], *args, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor, residual: List[torch.Tensor]) -> List[torch.Tensor]:
        # a chain of decoder layers
        outputs = []
        for layer, res in zip(self.layers, residual[::-1]):
            input = layer(input, res)
            outputs.append(input)
        return outputs


class UnetFramework(nn.Module):
    def __init__(
        self,
        encoder: UnetFrameworkEncoderLayer,
        decoder: UnetFrameworkDecoderLayer,
        bottleneck: nn.Module,
        stem_head: nn.Module,
        task_head: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.stem_head = stem_head
        self.task_head = task_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass through the stem
        x = self.stem_head(x)
        # pass the inputs through the encoder
        encoder_outs, residuals_outs = self.encoder(x)
        # pass the final enocder outputs through the bottleneck
        x = self.bottleneck(encoder_outs[-1])
        # pass the inputs through the decoder
        decoder_outs = self.decoder(x, residuals_outs)
        # pass the final decoder outputs through the task head
        x = self.task_head(decoder_outs[-1])
        return x


class UnetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(3):
            layers.append(
                Conv2dNormActivation(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UnetEncoderLayer(UnetFrameworkEncoderLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:
        feature_layer = UnetConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        projection_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        super().__init__(feature_layer, projection_layer)


class UnetUpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UnetCombinationLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, residual], dim=1)
        return x


class UnetDecoderLayer(UnetFrameworkDecoderLayer):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
    ) -> None:

        feature_layers = [
            UnetConvLayer(
                in_channels=channels * 2 if i == 0 else channels,
                out_channels=channels,
                kernel_size=kernel_size,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
            for i in range(3)
        ]

        # vanilla unet does not project the features after re-combining
        feature_layer = nn.Sequential(*feature_layers)
        projection_layer = nn.Identity()
        upsample_layer = UnetUpsampleLayer(in_channels=channels * 2)
        super().__init__(
            feature_layer=feature_layer,
            projection_layer=projection_layer,
            upsample_layer=upsample_layer,
            combination_layer=UnetCombinationLayer(),
        )


class Unet(UnetFramework):
    def __init__(
        self,
        stem_head: nn.Module,
        task_head: nn.Module,
        in_channels: int,
        block_channels: List[int],
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:

        # encoder channels will be [in_channels, 64, 128, 256, 512]
        encoder_channels = [in_channels] + block_channels

        # decoder channels should be [512]
        decoder_channels = block_channels[::-1]

        encoder_layers = [
            UnetEncoderLayer(
                in_channels=encoder_channels[i],
                out_channels=encoder_channels[i + 1],
                activation_layer=activation_layer,
                norm_layer=norm_layer,
            )
            for i in range(len(block_channels))
        ]

        encoder = UnetFrameworkEncoder(encoder_layers)

        decoder_layers = [
            UnetDecoderLayer(
                channels=decoder_channels[i],
                kernel_size=3,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
            )
            for i in range(len(block_channels))
        ]

        decoder = UnetFrameworkDecoder(decoder_layers)

        bottleneck_layers = []
        for i in range(3):
            bottleneck_layers.append(
                Conv2dNormActivation(
                    in_channels=block_channels[-1] if i == 0 else block_channels[-1] * 2,
                    out_channels=block_channels[-1] * 2,
                    kernel_size=3,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        bottleneck = nn.Sequential(*bottleneck_layers)

        super().__init__(
            encoder=encoder, decoder=decoder, bottleneck=bottleneck, task_head=task_head, stem_head=stem_head
        )


class UnetSegmentation(Unet):
    def __init__(
        in_channels: int,
        num_classes: int,
        block_channels: List[int],
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        task_head = nn.Conv2d(block_channels[0], num_classes, kernel_size=1, bias=False)
        stem_head = nn.Identity()
        super().__init__(
            stem_head=stem_head,
            task_head=task_head,
            in_channels=in_channels,
            block_channels=block_channels,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
        )


class UnetDiffusion(Unet):
    def __init__(
        self,
        in_channels: int,
        block_channels: List[int],
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        stem_head = nn.Conv2d(in_channels, block_channels[0], kernel_size=1, bias=False)
        task_head = nn.Sequential(nn.Conv2d(block_channels[0], 3, kernel_size=1, bias=False), nn.Tanh())
        super().__init__(
            stem_head=stem_head,
            task_head=task_head,
            in_channels=in_channels,
            block_channels=block_channels,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
        )
