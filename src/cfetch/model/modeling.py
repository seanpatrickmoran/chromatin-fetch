from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VAE"]


class SymmetricConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pt, ph, pw = self.padding
        self._pad = (pw, pw, ph, ph, pt, pt)
        self.padding = (0, 0, 0)

    def forward(self, x):
        return super().forward(F.pad(x, self._pad))


class RMS_norm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
        if self.bias is not None:
            y = y + self.bias
        return y


class Upsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix bfloat16 support for nearest-exact interpolation by upcasting then casting back
        return super().forward(x.float()).type_as(x)

def patchify_2x(x: torch.Tensor) -> torch.Tensor:
    """
    patch_size=2 only.
    - 4D: [B,C,H,W] -> [B, C*4, H/2, W/2]
    - 5D: [B,C,T,H,W] -> [B, C*4, T, H/2, W/2]
    """
    if x.dim() == 4:
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(B, C * 4, H // 2, W // 2)

    if x.dim() == 5:
        B, C, T, H, W = x.shape
        x = x.view(B, C, T, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        return x.view(B, C * 4, T, H // 2, W // 2)

    raise ValueError(f"patchify_2x: expected 4D/5D, got {tuple(x.shape)}")


def unpatchify_2x(x: torch.Tensor) -> torch.Tensor:
    """
    patch_size=2 inverse.
    - 4D: [B,C*4,H,W] -> [B,C,H*2,W*2]
    - 5D: [B,C*4,T,H,W] -> [B,C,T,H*2,W*2]
    """
    if x.dim() == 4:
        B, C4, H, W = x.shape
        C = C4 // 4
        x = x.view(B, C, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(B, C, H * 2, W * 2)

    if x.dim() == 5:
        B, C4, T, H, W = x.shape
        C = C4 // 4
        x = x.view(B, C, 2, 2, T, H, W)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return x.view(B, C, T, H * 2, W * 2)

    raise ValueError(f"unpatchify_2x: expected 4D/5D, got {tuple(x.shape)}")


# Blocks
class Resample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode

        if mode == "upsample2d":
            self.spatial = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time = None

        elif mode == "upsample3d":
            self.spatial = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            # produce 2x time by causal conv -> 2*dim channels then reshape
            self.time = SymmetricConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.spatial = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time = None

        elif mode == "downsample3d":
            self.spatial = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time = SymmetricConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        else:
            self.spatial = nn.Identity()
            self.time = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        B, C, T, H, W = x.shape

        if self.mode == "upsample3d":
            x = self.time(x)                                 # [B,2C,T,H,W]
            x = x.view(B, 2, C, T, H, W)                     # [B,2,C,T,H,W]
            x = x.permute(0, 2, 3, 1, 4, 5).contiguous()     # [B,C,T,2,H,W]
            x = x.view(B, C, T * 2, H, W)                    # [B,C,2T,H,W]
            T = x.shape[2]

        # spatial resample by folding time into batch
        x2 = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x2 = self.spatial(x2)
        _, C2, H2, W2 = x2.shape
        x = x2.view(B, T, C2, H2, W2).permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "downsample3d":
            x = self.time(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            SymmetricConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            SymmetricConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = SymmetricConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    Single-head attention on spatial tokens per-frame.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        identity = x
        B, C, T, H, W = x.shape

        xt = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # [BT,C,H,W]
        xt = self.norm(xt)
        qkv = self.to_qkv(xt)  # [BT,3C,H,W]

        # reshape to tokens
        qkv = qkv.view(B * T, 3, C, H * W).permute(0, 1, 3, 2).contiguous()  # [BT,3,HW,C]
        q = qkv[:, 0].unsqueeze(1)  # [BT,1,HW,C]
        k = qkv[:, 1].unsqueeze(1)
        v = qkv[:, 2].unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v)  # [BT,1,HW,C]
        out = out.squeeze(1).permute(0, 2, 1).contiguous().view(B * T, C, H, W)

        out = self.proj(out)
        out = out.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return out + identity


# Encoder / Decoder
class Down_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_downsample: bool,
        down_flag: bool,
    ):
        super().__init__()
        blocks = []
        for _ in range(mult):
            blocks.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            blocks.append(Resample(out_dim, mode=mode))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Up_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temporal_upsample: bool,
        up_flag: bool,
    ):
        super().__init__()
        blocks = []
        for _ in range(mult):
            blocks.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            blocks.append(Resample(out_dim, mode=mode))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temporal_downsample=(True, True, False),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [dim * u for u in (1, *tuple(dim_mult))]

        self.conv_in = SymmetricConv3d(4, dims[0], 3, padding=1)

        downs = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = bool(temporal_downsample[i]) if i < len(temporal_downsample) else False
            downs.append(
                Down_ResidualBlock(
                    in_dim=in_d,
                    out_dim=out_d,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temporal_downsample=t_down,
                    down_flag=(i != (len(dim_mult) - 1)),
                )
            )
        self.downs = nn.Sequential(*downs)

        self.mid = nn.Sequential(
            ResidualBlock(dims[-1], dims[-1], dropout),
            AttentionBlock(dims[-1]),
            ResidualBlock(dims[-1], dims[-1], dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            SymmetricConv3d(dims[-1], z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.downs(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temporal_upsample=(False, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        dm = tuple(dim_mult)
        dims = [dim * u for u in (dm[-1], *reversed(dm))]

        self.conv_in = SymmetricConv3d(z_dim, dims[0], 3, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        ups = []
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = bool(temporal_upsample[i]) if i < len(temporal_upsample) else False
            ups.append(
                Up_ResidualBlock(
                    in_dim=in_d,
                    out_dim=out_d,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temporal_upsample=t_up,
                    up_flag=(i != (len(dm) - 1)),
                )
            )
        self.ups = nn.Sequential(*ups)

        self.head = nn.Sequential(
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            SymmetricConv3d(dims[-1], 4, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid(x)
        x = self.ups(x)
        x = self.head(x)
        return x


# VAE
class VAE(nn.Module):
    """
    forward returns: (x_recon, mu, log_var)
    - No cache/chunking: best for fixed-length training and torch.compile.
    """

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        temporal_downsample=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if temporal_downsample is not None:
            temperal_downsample = temporal_downsample
        self.z_dim = int(z_dim)

        self.encoder = Encoder3d(
            dim=dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_downsample=temperal_downsample,
            dropout=dropout,
        )
        self.post_enc = SymmetricConv3d(z_dim * 2, z_dim * 2, 1)

        self.pre_dec = SymmetricConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim=dec_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temporal_upsample=tuple(reversed(tuple(temperal_downsample))),
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor, scale=(0.0, 1.0)) -> tuple[torch.Tensor, torch.Tensor]:
        x = patchify_2x(x)
        h = self.encoder(x)
        mu, log_var = self.post_enc(h).chunk(2, dim=1)

        s0, s1 = scale
        if isinstance(s0, torch.Tensor):
            mu = (mu - s0.view(1, self.z_dim, 1, 1, 1)) * s1.view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - float(s0)) * float(s1)
        return mu, log_var

    def decode(self, z: torch.Tensor, scale=(0.0, 1.0)) -> torch.Tensor:
        s0, s1 = scale
        if isinstance(s0, torch.Tensor):
            z = z / s1.view(1, self.z_dim, 1, 1, 1) + s0.view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / float(s1) + float(s0)

        z = self.pre_dec(z)
        x = self.decoder(z)
        x = unpatchify_2x(x)
        return x

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        scale=(0.0, 1.0),
        sample_latent: bool = False,
    ):
        mu, log_var = self.encode(x, scale)
        z = self.reparameterize(mu, log_var) if sample_latent else mu
        x_recon = self.decode(z, scale)
        return x_recon, mu, log_var

    def sample(self, imgs: torch.Tensor, scale=(0.0, 1.0), deterministic: bool = False) -> torch.Tensor:
        mu, log_var = self.encode(imgs, scale)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)
