# ported from: Originally ported from: https://github.com/neonbjb/tortoise-tts

import math

import torch
from torch import nn
from torch.nn import functional as F


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 conversion for improved numerical stability.
    
    This class extends PyTorch's GroupNorm to perform normalization in float32
    precision, regardless of the input tensor's dtype. This helps prevent 
    numerical instability issues when using lower precision dtypes.
    """

    def forward(self, x):
        """Forward pass with automatic float32 conversion.

        Args:
            x (torch.Tensor): Input tensor of any dtype.

        Returns:
            torch.Tensor: Normalized tensor converted back to input dtype.
        """
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """N-dimensional convolution factory function.

    Args:
        dims (int): Number of dimensions (1, 2, or 3).
        *args: Arguments passed to the convolution constructor.
        **kwargs: Keyword arguments passed to the convolution constructor.

    Returns:
        nn.Module: Appropriate convolution module for the given dimensions.

    Raises:
        ValueError: If dimensions are not 1, 2, or 3.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels):
    """Create a GroupNorm32 normalization layer with adaptive group size.

    Automatically determines the optimal number of groups based on the number
    of channels, ensuring the number of channels is divisible by the group size.

    Args:
        channels (int): Number of input channels.

    Returns:
        GroupNorm32: Normalization layer with appropriate group size.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


def zero_module(module):
    """Initialize all parameters of a module to zero.

    Args:
        module (nn.Module): PyTorch module to initialize.

    Returns:
        nn.Module: Module with zeroed parameters.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class QKVAttention(nn.Module):
    """Multi-head QKV attention mechanism.
    
    Implements scaled dot-product attention with query, key, and value tensors
    combined in a single input tensor. Supports optional masking and bias.
    """

    def __init__(self, n_heads):
        """Initialize QKV attention.

        Args:
            n_heads (int): Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, qk_bias=0):
        """Apply QKV attention.

        Args:
            qkv (torch.Tensor): Input tensor of shape [N x (H * 3 * C) x T] containing
                concatenated queries, keys, and values.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            qk_bias (float, optional): Bias added to attention scores. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor of shape [N x (H * C) x T] after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = weight + qk_bias
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
            weight[mask.logical_not()] = -torch.inf
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Self-attention block with spatial attention capabilities.
    
    This block allows different spatial positions to attend to each other through
    multi-head self-attention. It includes normalization, optional activation,
    and residual connections.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        out_channels=None,
        do_activation=False,
    ):
        """Initialize attention block.

        Args:
            channels (int): Number of input channels.
            num_heads (int, optional): Number of attention heads. Defaults to 1.
            num_head_channels (int, optional): Channels per head. If -1, divide channels
                by num_heads. Defaults to -1.
            out_channels (int, optional): Number of output channels. If None, same as
                input channels. Defaults to None.
            do_activation (bool, optional): Whether to apply SiLU activation after
                normalization. Defaults to False.
        """
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.do_activation = do_activation
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, out_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.x_proj = nn.Identity() if out_channels == channels else conv_nd(1, channels, out_channels, 1)
        self.proj_out = zero_module(conv_nd(1, out_channels, out_channels, 1))

    def forward(self, x, mask=None, qk_bias=0):
        """Forward pass of attention block.

        Args:
            x (torch.Tensor): Input tensor of shape [B x C x *spatial_dims].
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            qk_bias (float, optional): Bias added to attention scores. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        b, c, *spatial = x.shape
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)
            if mask.shape[1] != x.shape[-1]:
                mask = mask[:, : x.shape[-1], : x.shape[-1]]

        x = x.reshape(b, c, -1)
        x = self.norm(x)
        if self.do_activation:
            x = F.silu(x, inplace=True)
        qkv = self.qkv(x)
        h = self.attention(qkv, mask=mask, qk_bias=qk_bias)
        h = self.proj_out(h)
        xp = self.x_proj(x)
        return (xp + h).reshape(b, xp.shape[1], *spatial)


class ConditioningEncoder(nn.Module):
    """Encoder for conditioning signals using self-attention.
    
    This module encodes mel-spectrograms or similar conditioning signals into
    a latent space using a series of attention blocks. It first projects the
    input to the embedding dimension using a 1x1 convolution, then applies
    multiple attention blocks.
    """

    def __init__(
        self,
        spec_dim,
        embedding_dim,
        attn_blocks=6,
        num_attn_heads=4,
    ):
        """Initialize conditioning encoder.

        Args:
            spec_dim (int): Dimension of input spectrogram features.
            embedding_dim (int): Dimension of the embedding space.
            attn_blocks (int, optional): Number of attention blocks. Defaults to 6.
            num_attn_heads (int, optional): Number of attention heads per block.
                Defaults to 4.
        """
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        """Encode input spectrogram into latent representation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, spec_dim, sequence_length].

        Returns:
            torch.Tensor: Encoded representation of shape [batch_size, embedding_dim, sequence_length].
        """
        h = self.init(x)
        h = self.attn(h)
        return h
