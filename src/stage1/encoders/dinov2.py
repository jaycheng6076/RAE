import torch
from torch import nn
from transformers import Dinov2WithRegistersModel
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=False
            )
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)


@register_encoder()
class Dinov2RegisterTokens(nn.Module):
    """
    DINOv2 encoder that extracts only the 4 register tokens.

    DINOv2 with registers outputs:
    - Position 0: CLS token
    - Positions 1-4: Register tokens (4 tokens)
    - Positions 5+: Patch tokens

    This encoder returns only the 4 register tokens for training SOCAE.
    """

    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
        include_cls: bool = False,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=False
            )
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.include_cls = include_cls
        # Number of register tokens (typically 4 for DINOv2 with registers)
        self.num_register_tokens = getattr(
            self.encoder.config, "num_register_tokens", 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that extracts only register tokens.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Register tokens [B, num_tokens, hidden_size]
            where num_tokens = 4 (or 5 if include_cls=True)
        """
        out = self.encoder(x, output_hidden_states=True)
        hidden_states = out.last_hidden_state

        if self.include_cls:
            # Return CLS + 4 register tokens (positions 0-4)
            register_tokens = hidden_states[:, :5]
        else:
            # Return only register tokens (positions 1-4)
            register_tokens = hidden_states[:, 1:5]

        return register_tokens
