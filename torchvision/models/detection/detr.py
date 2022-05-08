import copy
from typing import Optional, Callable

import torch
from torch import nn, Tensor


def _get_clones(module, N):
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
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
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
        pos_embed: Optional[Tensor] = None,
    ):
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
    ):
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
    ):
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
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
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

    def _with_pos_embed(self, x: Tensor, pos_embed: Tensor = None):
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
    ):
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
        norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
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

    def forward(self, src: Tensor, mask: Tensor, query_pos_embed: Tensor, pos_embed: Tensor):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_pos_embed = query_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed) # TODO: torch.fx
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_pos_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class DETR:
    pass
