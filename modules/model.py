from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def trunc_normal_(x, mean=0., std=1.):
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TextTransformer(nn.Module):
    def __init__(self, embed_dim: int = 512, width: int = 512, layers: int = 12, heads: int = 16, context_length: int = 77, vocab_size: int = 49408, emb_dropout: float = 0., dropout=None):
        super().__init__()
        if dropout is None:
            dropout = [0.0 for i in range(layers)]
        print('dropout used:{}'.format(dropout))
        self.width = width
        self.layers = layers
        self.context_length = context_length
        self.vocab_size = vocab_size

        attn_mask = self.build_attention_mask()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))

        self.ln_final = LayerNorm(width)

        self.dropout = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout

        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))

        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout[i]) for i in range(layers)])

        self.initialize_parameters()

    @property
    def dtype(self):
        return self.text_projection.dtype

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: torch.Tensor):

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class FusionTransformer(nn.Module):
    def __init__(self, clip_length=10, embed_dim=2048, n_layers=6, batch_size=1, vsfusion=True):
        super(FusionTransformer, self).__init__()
        self.clip_length = clip_length
        self.vsfusion = vsfusion
        drop_rate = 0.
        if vsfusion:
            embed_dim *= 2
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if vsfusion:
            self.text_projection = nn.Linear(embed_dim, embed_dim//2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cls_projection = nn.Linear(batch_size*embed_dim, batch_size)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, s, v=None):
        n_clip = s.shape[0]
        if self.vsfusion:
            x = torch.concat([v.unsqueeze(0), s.unsqueeze(0)], dim=-1)
        else:
            x = s.unsqueeze(0)
        x = F.pad(x, (0, 0, 0, self.clip_length - x.shape[1], 0, 0))
        x = torch.concat([self.cls_token, x], dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        assert o.shape[0] == self.clip_length + 1
        cls_emb = o[0]
        vs_emb = o[1:1+n_clip].mean(dim=0)
        if self.vsfusion:
            vs_emb = self.text_projection(vs_emb)
        return cls_emb, vs_emb

    def fuse_script(self, s):
        n_clip = s.shape[0]
        x = s.unsqueeze(0)
        x = F.pad(x, (0, 0, 0, self.clip_length - x.shape[1], 0, 0))
        fake_cls_token = torch.zeros(self.cls_token.shape, requires_grad = False).cuda()
        x = torch.concat([fake_cls_token, x], dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        assert o.shape[0] == self.clip_length + 1
        s_emb = o[1:1+n_clip].mean(dim=0)
        return s_emb

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)