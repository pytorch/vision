import copy
import math
from typing import Optional, Callable, Tuple

import torch
from torch import nn, Tensor


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    """Transormer Encoder Layer"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation_layer()

        self.norm_first = norm_first

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = self.norm1(x)
        if pos_embedding is not None:
            q = k = x + pos_embed
        else:
            q = k = x
        x = src + self.dropout1(
            self.self_attn(q, k, value=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        )
        if not self.norm_first:
            x = self.norm1(x)
        if self.norm_first:
            x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x))))))
        else:
            x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos_embed: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool = False,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos_embed: Optional[Tensor] = None,
        query_pos_embed: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embed=pos_embed,
                query_pos=query_pos_embed,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.norm3 = norm_layer(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation_layer()
        self.norm_first = norm_first

    def _with_pos_embed(self, x: Tensor, pos_embed: Optional[Tensor] = None) -> Tensor:
        return x + pos_embed if pos_embed is not None else x

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos_embed: Optional[Tensor] = None,
        query_pos_embed: Optional[Tensor] = None,
    ) -> None:
        x = tgt
        if self.norm_first:
            x = self.norm1(x)
        q = k = self._with_pos_embed(x, query_pos_embed)
        x = tgt + self.dropout1(
            self.self_attn(q, k, value=x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        )
        if not self.norm_first:
            x = self.norm1(x)
        if self.norm_first:
            x = x + self.dropout2(
                self.multihead_attn(
                    query=self._with_pos_embed(self.norm2(x), query_pos_embed),
                    key=self._with_pos_embed(memory, pos_embed),
                    value=memory,
                    attn_mask=memory_mask,
                    key_padding_mask=memory_key_padding_mask,
                )[0]
            )
            x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x))))))
        else:
            x = self.norm2(
                x
                + self.dropout2(
                    self.multihead_attn(
                        query=self._with_pos_embed(x, query_pos_embed),
                        key=self._with_pos_embed(memory, pos_embed),
                        value=memory,
                        attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask,
                    )[0]
                )
            )
            x = self.norm3(x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x))))))


class Transformer(nn.Module):
    """Transformer"""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
        return_intermediate_dec: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first, activation_layer, norm_layer
        )
        encoder_norm = norm_layer(d_model) if norm_first else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first, activation_layer, norm_layer
        )
        decoder_norm = norm_layer(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, decoder_norm)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, mask: Tensor, query_pos_embed: Tensor, pos_embed: Tensor) -> Tuple[Tensor, Tensor]:
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_pos_embed = query_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_pos_embed)  # TODO: torch.fx
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_pos_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class DETR:
    pass


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensors: Tensor, mask: Tensor) -> Tensor:
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensors.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensors: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h, w = tensors.shape[-2:]
        i = torch.arange(w, device=tensors.device)
        j = torch.arange(h, device=tensors.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(tensors.shape[0], 1, 1, 1)
        )
        return pos
