"""Mega-based separation models.

Authors
 * Luca Della Libera
"""

from typing import Optional

import numpy as np
import torch
from mega_pytorch import MegaLayer
from torch import nn

import speechbrain as sb
from speechbrain.lobes.models.dual_path import select_norm
from speechbrain.lobes.models.resepformer import (
    SBTransformerBlock_wnormandskip as SBTBW,
)
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding


__all__ = ["SBTransformerBlock_wnormandskip"]


EPS = torch.finfo(torch.get_default_dtype()).eps


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        laplacian_attn_fn=False,
    ):
        super().__init__()

        self.self_att = MegaLayer(
            dim=d_model,
            ema_heads=nhead,
            attn_dim_qk=kdim or d_model,
            attn_dim_value=vdim or d_model,
            laplacian_attn_fn=laplacian_attn_fn,
            causal=causal,
        )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output = self.self_att(src1)

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        laplacian_attn_fn=False,
        layerdrop_prob=0.0,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    laplacian_attn_fn=laplacian_attn_fn,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst


class SBTransformerBlock_wnormandskip(SBTBW):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        causal=False,
        laplacian_attn_fn=False,
        use_norm=True,
        use_skip=True,
        norm_type="gln",
    ):
        super(SBTBW, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.causal = causal

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            causal=causal,
            laplacian_attn_fn=laplacian_attn_fn,
        )

        self.use_norm = use_norm
        self.use_skip = use_skip

        if use_norm:
            self.norm = select_norm(
                norm=norm_type, dim=d_model, shape=3, eps=EPS
            )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(
                input_size=d_model, max_len=100000
            )


def test_SBTransformerBlock_wnormandskip():
    batch_size = 7
    sequence_length = 150
    num_layers = 4
    d_model = 128
    nhead = 8
    lr = 0.1

    model = SBTransformerBlock_wnormandskip(num_layers, d_model, nhead).cuda()
    torch.manual_seed(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    input = torch.rand(batch_size, sequence_length, d_model).cuda()
    output = model(input)
    loss = output.sum()
    loss.backward()
    print(input.shape)
    print(output.shape)
    optimizer.step()


if __name__ == "__main__":
    test_SBTransformerBlock_wnormandskip()
