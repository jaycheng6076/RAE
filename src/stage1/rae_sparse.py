from math import sqrt
from typing import Any, Dict, Optional, Protocol, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor

from .bottleneck.socae import (
    SparseContrastiveAutoEncoder,
    SparseOrthogonalContrastiveAutoEncoder,
)
from .decoders import GeneralDecoder
from .encoders import ARCHS


def _get_obj_from_str(string: str) -> type:
    """Import a class from a module path string."""
    import importlib

    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def _instantiate_quantizer(
    config: Union[Dict[str, Any], nn.Module, None]
) -> Optional[nn.Module]:
    """Instantiate a quantizer from a config dict or return the module directly."""
    if config is None:
        return None
    if isinstance(config, nn.Module):
        return config

    if hasattr(config, "get") and "target" in config:
        quantizer_cls = _get_obj_from_str(config["target"])
        params = config.get("params", {})
        if hasattr(params, "items"):
            params = dict(params)
        quantizer = quantizer_cls(**params)

        ckpt_path = config.get("ckpt", None)
        if ckpt_path is not None:
            print(f"Loading pre-trained quantizer from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "ema" in state_dict:
                state_dict = state_dict["ema"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            quantizer.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded pre-trained quantizer from {ckpt_path}")

        return quantizer
    raise ValueError(
        f"Invalid quantizer config: expected dict-like object with 'target' key or nn.Module, got {type(config)}"
    )


class SparseInputProjection(nn.Module):
    """
    Projects sparse high-dimensional input to decoder hidden size.

    Designed to handle sparse input efficiently by:
    1. Using a linear projection (works well with sparse input)
    2. Optionally using sparse-aware operations
    """

    def __init__(
        self,
        sparse_dim: int,
        output_dim: int,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.output_dim = output_dim

        # Linear projection from sparse dim to decoder dim
        # Note: For very sparse input, the effective computation is already efficient
        # because most input values are zero
        self.projection = nn.Linear(sparse_dim, output_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse input tensor [B, N, sparse_dim] with mostly zeros
        Returns:
            Projected tensor [B, N, output_dim]
        """
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class SparseToImageDecoder(nn.Module):
    """
    Decoder that takes sparse SOCAE features and reconstructs images.

    Architecture:
    1. Sparse input projection: sparse_dim -> decoder_hidden_size
    2. Transformer decoder layers (same as GeneralDecoder)
    3. Pixel prediction head
    """

    def __init__(
        self,
        sparse_dim: int,
        decoder_config,
        num_patches: int,
        projection_dropout: float = 0.0,
    ):
        super().__init__()

        # Sparse input projection
        self.sparse_projection = SparseInputProjection(
            sparse_dim=sparse_dim,
            output_dim=decoder_config.decoder_hidden_size,
            dropout=projection_dropout,
        )

        # Standard ViT-MAE decoder (reuse GeneralDecoder but replace embed)
        self.decoder = GeneralDecoder(decoder_config, num_patches)

        # Remove the original decoder_embed since we use sparse_projection
        # We'll bypass it in forward
        self.num_patches = num_patches
        self.config = decoder_config

    def forward(
        self,
        sparse_features: torch.Tensor,
        drop_cls_token: bool = False,
    ):
        """
        Args:
            sparse_features: [B, N, sparse_dim] sparse SOCAE features
            drop_cls_token: Whether input has cls token to drop
        Returns:
            Reconstructed patches
        """
        # Project sparse features to decoder hidden size
        x = self.sparse_projection(sparse_features)  # [B, N, decoder_hidden_size]

        # Interpolate if needed
        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.decoder.interpolate_latent(x_)
        else:
            x_ = self.decoder.interpolate_latent(x)

        # Add trainable cls token
        cls_token = self.decoder.trainable_cls_token.expand(x_.shape[0], -1, -1)
        x = torch.cat([cls_token, x_], dim=1)

        # Add positional embedding
        hidden_states = x + self.decoder.decoder_pos_embed

        # Apply transformer layers
        for layer_module in self.decoder.decoder_layers:
            layer_outputs = layer_module(
                hidden_states, head_mask=None, output_attentions=False
            )
            hidden_states = layer_outputs[0]

        # Final norm and prediction
        hidden_states = self.decoder.decoder_norm(hidden_states)
        logits = self.decoder.decoder_pred(hidden_states)

        # Remove cls token
        logits = logits[:, 1:, :]

        return logits

    def unpatchify(self, patchified_pixel_values):
        """Convert patches back to image."""
        return self.decoder.unpatchify(patchified_pixel_values)


class Stage1Protocal(Protocol):
    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor: ...


Quantizer = Union[
    SparseContrastiveAutoEncoder, SparseOrthogonalContrastiveAutoEncoder, None
]


class SparseRAE(nn.Module):
    """
    Sparse RAE that uses SOCAE sparse features directly as decoder input.

    Unlike rae_s.py which reconstructs from sparse back to original dim,
    this version feeds the high-dimensional sparse features directly to
    a sparse-aware decoder.

    Flow:
      x -> encode -> z (768-dim) -> SOCAE.encode -> z_sparse (12288-dim) -> decode -> x_rec
    """

    def __init__(
        self,
        # ---- encoder configs ----
        encoder_cls: str = "Dinov2withNorm",
        encoder_config_path: str = "facebook/dinov2-base",
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = "vit_mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- quantizer configs ----
        quantizer: Quantizer = None,
        sparse_dim: int = 12288,  # SOCAE hidden_dim
        projection_dropout: float = 0.0,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        # Encoder setup (same as rae_s.py)
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        self.sparse_dim = sparse_dim

        assert (
            self.encoder_input_size % self.encoder_patch_size == 0
        ), f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"

        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # Decoder config - NOTE: hidden_size is now sparse_dim
        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
        decoder_config.hidden_size = sparse_dim  # Set to sparse dim, not latent_dim
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))

        # Use sparse-aware decoder
        self.decoder = SparseToImageDecoder(
            sparse_dim=sparse_dim,
            decoder_config=decoder_config,
            num_patches=self.base_patches,
            projection_dropout=projection_dropout,
        )

        # Load pretrained decoder weights if available
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            # Only load matching keys (sparse_projection is new)
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(
                    f"Missing keys when loading pretrained decoder: {keys.missing_keys}"
                )

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        # Normalization stats
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

        # Quantizer (SOCAE)
        self.quantizer = _instantiate_quantizer(quantizer)

        if self.quantizer is not None:
            assert self.quantizer.hidden_dim == sparse_dim, (
                f"Quantizer hidden_dim ({self.quantizer.hidden_dim}) must match "
                f"sparse_dim ({sparse_dim})"
            )

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent (before SOCAE)."""
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = (
                self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            )
            latent_var = (
                self.latent_var.to(z.device) if self.latent_var is not None else 1
            )
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def encode_sparse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply SOCAE to get sparse features (without decoding back).

        Args:
            z: Encoder output [B, C, H, W] or [B, N, C]
        Returns:
            Sparse features [B, N, sparse_dim] with mostly zeros
        """
        if self.quantizer is None:
            raise ValueError("Quantizer (SOCAE) must be provided for sparse encoding")

        if self.reshape_to_2d:
            b, c, h, w = z.shape
            z_flat = z.view(b, c, h * w).transpose(1, 2)  # [B, N, C]
        else:
            z_flat = z
            b = z_flat.shape[0]

        # Get sparse features (NOT reconstructed)
        z_sparse = self.quantizer.encode(
            z_flat.reshape(-1, z_flat.shape[-1])
        )  # [B*N, sparse_dim]
        z_sparse = z_sparse.view(b, -1, self.sparse_dim)  # [B, N, sparse_dim]

        return z_sparse

    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to image.

        Args:
            z_sparse: Sparse features [B, N, sparse_dim]
        Returns:
            Reconstructed image [B, 3, H, W]
        """
        # Decode using sparse-aware decoder
        output = self.decoder(z_sparse, drop_cls_token=False)
        x_rec = self.decoder.unpatchify(output)

        # Denormalize
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(
            x_rec.device
        )

        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: x -> encode -> z -> SOCAE.encode -> z_sparse -> decode -> x_rec

        Note: This uses sparse features DIRECTLY, not reconstructed features.
        """
        z = self.encode(x)
        z_sparse = self.encode_sparse(z)
        x_rec = self.decode(z_sparse)
        return x_rec

    def forward_with_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that also returns intermediate features for analysis.
        """
        z = self.encode(x)
        z_sparse = self.encode_sparse(z)
        x_rec = self.decode(z_sparse)

        return {
            "x_rec": x_rec,
            "z": z,
            "z_sparse": z_sparse,
            "sparsity": (z_sparse == 0).float().mean(),
            "active_per_patch": (z_sparse > 0).sum(dim=-1).float().mean(),
        }
