"""T2T-ViT variant for extracting fixed candlestick-chart embeddings."""
import torch
import torch.nn as nn
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .transformer_block import Block

def get_1d_sinusoid_encoding(n_position, d_hid):
    """Return a one-dimensional sinusoidal positional encoding."""
    position = np.arange(n_position)[:, None]
    div_term = np.exp(np.arange(0, d_hid, 2) * -(np.log(10000.0) / d_hid))
    pe = np.zeros((n_position, d_hid))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float).unsqueeze(0)

class VerticalT2TModule(nn.Module):
    """Perform one learnable tokens-to-token aggregation along the time axis."""

    def __init__(self, img_size=224, patch_width=4, in_chans=3, embed_dim=256,
                 num_heads=4, mlp_ratio=2.0, drop_rate=0.0):
        super().__init__()
        self.patch_width = patch_width
        initial_tokens = img_size // patch_width
        self.num_patches = (initial_tokens + 1) // 2
        self.soft_split = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(img_size, patch_width),
            stride=(1, patch_width),
        )
        self.local_attention = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            drop=drop_rate,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
        )
        # Overlapping 1D aggregation is the restructurization step. It merges
        # neighboring time tokens without mixing the chart's price axis.
        self.restructure = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self.soft_split(x)
        x = x.squeeze(2).transpose(1, 2)
        x = self.local_attention(x)
        x = self.restructure(x.transpose(1, 2)).transpose(1, 2)
        return x


# Backward-compatible name for external imports.
KlineTokenizer = VerticalT2TModule

class T2T_ViT_Kline(nn.Module):
    """Transformer encoder specialized for candlestick-chart images."""
    def __init__(self, img_size=224, patch_width=4, in_chans=3, num_classes=2,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=2.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.3, attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.num_features = self.embed_dim = embed_dim

        self.tokens_to_token = VerticalT2TModule(
            img_size=img_size,
            patch_width=patch_width,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(get_1d_sinusoid_encoding(num_patches + 1, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

    def extract_features(self, x):
        """Extract the 256-dimensional chart representation."""
        return self.forward_features(x)

@register_model
def t2t_vit_kline(pretrained=False, **kwargs):
    model = T2T_ViT_Kline(**kwargs)
    if pretrained:
        print("No generic pretrained weights are available; using random initialization.")
    return model
