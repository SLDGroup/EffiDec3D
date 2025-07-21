import torch.nn as nn


class PatchPartition(nn.Module):
    def __init__(self, channels):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        return x


class LineConv(nn.Module):
    def __init__(self, channels):
        super(LineConv, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(
            channels, channels * expansion, kernel_size=1, bias=False
        )
        self.act = nn.GELU()
        self.line_conv_1 = nn.Conv3d(
            channels * expansion, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x


class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm3d(channels)
        self.pointwise_conv_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(
            channels, channels, padding=1, kernel_size=3, groups=channels, bias=False
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.pointwise_conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose3d(
            channels, channels, kernel_size=r, stride=r, groups=channels
        )
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class Block(nn.Module):
    def __init__(self, channels, r, heads):
        super(Block, self).__init__()

        self.patch1 = PatchPartition(channels)
        self.LocalRC = LocalRepresentationsCongregation(channels)
        self.LineConv1 = LineConv(channels)
        self.patch2 = PatchPartition(channels)
        self.GlobalST = GlobalSparseTransformer(channels, r, heads)
        self.LocalRD = LocalReverseDiffusion(channels, r)
        self.LineConv2 = LineConv(channels)

    def forward(self, x):
        x = self.patch1(x) + x
        x = self.LocalRC(x) + x
        x = self.LineConv1(x) + x
        x = self.patch2(x) + x
        x = self.LocalRD(self.GlobalST(x)) + x
        x = self.LineConv2(x) + x
        return x
