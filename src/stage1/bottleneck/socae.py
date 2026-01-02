import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SparseContrastiveAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int,
        larger_topk: int | None = 0,
        dead_topk: int | None = None,
        dead_threshold: int = 64,
        ncl_with_larger_k: bool = False,
        track_dead_neuron_pre_relu: bool = True,
        activation_history_size: int = 1000,
        encoder_bias: bool = False,
        z_dead_k_in_positives_for_ncl: bool = False,
        input_whitening: bool = False,
    ) -> None:
        """
        Initializes the SparseContrastiveAutoEncoder model.

        Args:
            input_dim (int): The dimension of the input data.
            hidden_dim (int): The dimension of the hidden representation (latent space).
            topk (int): The number of top activations to keep in normal operation (enforces sparsity).
            larger_topk (int): The number of top activations for the larger sparse representation (less sparse).
            dead_topk (int, optional): The number of top activations for dead neuron recovery. Defaults to log2(input_dim/2).
            dead_threshold (int): The threshold to consider a neuron as dead (number of batches without activation).
            ncl_with_larger_k (bool, optional): Whether to use z_larger_k for NCL loss. Defaults to False.
            track_dead_neuron_pre_relu (bool, optional): Whether to track neuron activity before applying ReLU.
                                                        If True, neuron activity is tracked on the raw values before ReLU.
                                                        If False, neuron activity is tracked after applying ReLU.
                                                        Defaults to True.
            activation_history_size (int, optional): Size of the sliding window for tracking neuron activations.
                                                    Defaults to 1000.
            encoder_bias (bool, optional): Whether to use bias in the encoder. Defaults to False.
            z_dead_k_in_positives_for_ncl (bool, optional): Whether to include dead neuron activations as part of positives
                                                          in the NCL loss calculation. Defaults to False.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(input_dim))

        # Decoder network: maps hidden representation back to input space
        # Note: bias=False because we use self.bias
        self.decoder: nn.Linear = nn.Linear(hidden_dim, input_dim, bias=False)

        # Encoder network: maps input data to hidden representation space
        # Encoder can be more complex than just a linear layer, but we use it for now.
        self.encoder_bias = encoder_bias
        self.encoder: nn.Linear = nn.Linear(input_dim, hidden_dim, bias=encoder_bias)
        # Tied init when encoder is a linear layer, but with separated gradients
        self.encoder.weight = nn.Parameter(self.decoder.weight.T.clone().contiguous())

        self._unit_norm_decoder()

        # Sparsity control parameters
        self.topk = topk  # Regular sparsity level
        self.larger_topk: int = (
            topk * 4 if larger_topk is None else larger_topk
        )  # Relaxed sparsity for better reconstruction
        self.dead_topk: int = dead_topk or 2 ** (
            math.floor(math.log2(input_dim // 2))
        )  # Number of activations for dead neuron recovery
        self.dead_threshold = dead_threshold  # Age threshold to consider a neuron dead

        # Neuron activity tracking: counts batches since each neuron last activated
        self.stats_last_nonzero: Tensor
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long)
        )

        # Histogram tracking: counts non-zero activations for each hidden dimension
        # Uses a sliding window of the last 1000 batches
        self.activation_history_size = activation_history_size
        self.activation_history_idx = 0
        self.stats_activation_history: Tensor
        self.register_buffer(
            "stats_activation_history",
            torch.zeros(self.activation_history_size, hidden_dim, dtype=torch.long),
        )

        # Summary statistics of activation history
        self.stats_activation_counts: Tensor
        self.register_buffer(
            "stats_activation_counts", torch.zeros(hidden_dim, dtype=torch.long)
        )

        # Creates a mask that only allows dead neurons to be activated
        # This helps target and revive neurons that haven't been used recently
        def dead_mask_fn(x: Tensor) -> Tensor:
            # True for neurons that haven't fired in dead_threshold batches
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # Zero out all non-dead neurons
            return x

        self.dead_mask_fn: Callable[[Tensor], Tensor] = dead_mask_fn

        self.ncl_with_larger_k = ncl_with_larger_k
        self.z_dead_k_in_positives_for_ncl = z_dead_k_in_positives_for_ncl

        self.track_dead_neuron_pre_relu = track_dead_neuron_pre_relu

        self.mse_scale: float = 1.0

        # Simulated annealing related data
        self.starting_k_multipler: int = 1
        self.annealing_final_k = topk

        # data buffer for simulated annealing
        self.dead_buffer_size = 5
        self.dead_buffer: Tensor
        self.register_buffer(
            "dead_buffer", torch.zeros(self.dead_buffer_size, dtype=torch.float)
        )
        self.dead_buffer_index = 0
        self.input_whitening: bool = input_whitening
        self.data_mean: Tensor
        self.register_buffer(
            "data_mean", torch.zeros([1, self.input_dim], dtype=torch.float)
        )
        self.whitening_transform: Tensor
        self.register_buffer(
            "whitening_transform",
            torch.zeros((self.input_dim, self.input_dim), dtype=torch.float),
        )

    def init_from_data(
        self, samples: Tensor, scale_mse: bool = False, init_pre_bias: bool = True
    ) -> None:
        """
        Initialize the model parameters from the given data samples.

        Args:
            samples (Tensor): A tensor containing the data samples.
        """
        if scale_mse:
            self.mse_scale = (
                1.0
                / torch.pow(samples.float().mean(dim=0) - samples.float(), 2)
                .mean()
                .item()
            )
            logger.info(f"mse_scale: {self.mse_scale}")
        # Initialize pre-base using geometric median
        if init_pre_bias:
            assert samples.shape[0] >= 32768
            self.bias.data = (
                compute_geometric_median(samples[:32768].float().cpu())
                .median.float()
                .to(self.bias.device)
            )
        if self.input_whitening:
            assert samples.shape[0] >= 100_000
            self.data_mean = torch.mean(samples, dim=0, keepdim=True)
            centered_embeddings = samples - self.data_mean
            _, s, v = torch.svd(centered_embeddings, some=True)
            transform = v / torch.sqrt(s + 1e-6)
            self.whitening_transform = transform.contiguous()

    def apply_whitening(self, x: Tensor) -> Tensor:
        centered_data = x - self.data_mean
        whitened_data = torch.matmul(centered_data, self.whitening_transform)
        return whitened_data

    def simulated_annealing_init(
        self, batch_size: int, confidence: float = 0.95
    ) -> None:
        lbd = -np.log(confidence)
        p_empty = (self.hidden_dim - self.topk) / self.hidden_dim
        starting_k_multipler = int(
            np.log(lbd / batch_size) / np.log(p_empty) / batch_size + 1
        )
        self.starting_k_multipler: int = max(starting_k_multipler, int(16))
        self.topk *= self.starting_k_multipler
        if self.larger_topk > 0:
            self.larger_topk = self.topk * 4
        # We will simulated the effect of disbale encoder bias until most dead neurons are suppresesed
        if self.encoder_bias:
            self.encoder.bias.data.fill_(0)
            self.encoder.bias.requires_grad = False

    def simulated_annealing(self, dead_neuron_cnt: int) -> None:
        self.dead_buffer[self.dead_buffer_index] = dead_neuron_cnt
        self.dead_buffer_index = (self.dead_buffer_index + 1) % self.dead_buffer_size
        if (
            self.dead_buffer.mean() < self.hidden_dim / 100
            and min(self.topk - 1, max(self.annealing_final_k, int(self.topk * 0.99)))
            >= self.annealing_final_k
        ):
            self.topk = min(
                self.topk - 1, max(self.annealing_final_k, int(self.topk * 0.99))
            )
            if self.larger_topk > 0:
                self.larger_topk = self.topk * 4

            if self.encoder_bias:
                self.encoder.bias.requires_grad = True
        elif (
            self.dead_buffer.mean() >= self.hidden_dim / 100
            and max(int(self.topk * 1.1), self.topk + 1)
            < self.annealing_final_k * self.starting_k_multipler
        ):
            self.topk = max(int(self.topk * 1.1), self.topk + 1)
            if self.larger_topk > 0:
                self.larger_topk = self.topk * 4
            if self.encoder_bias:
                self.encoder.bias.requires_grad = False

    def _compute_top_k_with_stats(
        self, x: Tensor, topk: int, update_stats: bool = False
    ) -> Tensor:
        """
        Compute sparse representation by keeping only top-k activations.
        Optionally update neuron activity statistics to track dead neurons.

        Args:
            x: Input tensor of activations [batch_size × hidden_dim] or higher order tensor
               where the last dimension is hidden_dim
            topk: Number of top activations to keep per sample
            update_stats: Whether to update neuron firing statistics

        Returns:
            Sparse tensor with only top-k activations preserved
        """
        # Find top k activations for each sample in the batch
        values, indices = torch.topk(x, k=topk, dim=-1)
        if not self.track_dead_neuron_pre_relu:
            values = torch.relu(values)  # Ensure non-negative activations

        # Update neuron activity statistics if requested
        if update_stats:
            # Create temporary tensor to track which neurons fired in this batch
            tmp = torch.zeros_like(self.stats_last_nonzero)
            # Count neurons that fired above threshold (1e-5)
            tmp.scatter_add_(
                0,
                indices.reshape(-1),
                values.gt(1e-5).to(tmp.dtype).reshape(-1),
            )
            # Reset counter for neurons that fired in this batch
            self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
            # Increment age counter for all neurons
            self.stats_last_nonzero += 1

            # Store the current batch's neuron activation pattern in the circular buffer at the current index
            self.stats_activation_history[self.activation_history_idx] = tmp

            # Update the activation counts by summing the entire history
            self.stats_activation_counts = self.stats_activation_history.sum(dim=0)

            # Move to the next index in the circular buffer
            self.activation_history_idx = (
                self.activation_history_idx + 1
            ) % self.activation_history_size

        # Create sparse representation by keeping only top-k values
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, indices, values)
        if self.track_dead_neuron_pre_relu:
            z_topk = torch.relu(z_topk)  # Ensure non-negative activations
        return z_topk

    def top_k(
        self, x: Tensor, is_training: bool = True
    ) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Compute three different sparse representations with different sparsity levels:
        - Regular top-k for normal operation (most sparse)
        - Larger top-k for better reconstruction (less sparse)
        - Dead neuron top-k to revive dead neurons (focused on inactive neurons)

        Args:
            x: Raw activations from encoder [batch_size × hidden_dim]
            is_training: Whether the model is in training mode. If False, only the regular
                         representation is computed and the other two are set to None.

        Returns:
            Tuple of three sparse representations (regular, larger, dead-focused).
            During inference (is_training=False), returns (z_k, None, None).
        """
        # Standard sparse representation, update neuron statistics
        z_k = self._compute_top_k_with_stats(x, self.topk, update_stats=True)

        # During inference, only compute the standard representation
        if not is_training:
            return z_k, None, None

        # Less sparse representation for better reconstruction quality
        # Only computed during training
        z_large_k = (
            None
            if self.larger_topk == 0
            else self._compute_top_k_with_stats(x, self.larger_topk, update_stats=False)
        )

        # Representation focusing on dead neurons to revive them
        # First masks input to highlight only dead neurons, then computes top-k
        # Only computed during training
        z_dead_k = self._compute_top_k_with_stats(
            self.dead_mask_fn(x.clone()), self.dead_topk, update_stats=False
        )

        return z_k, z_large_k, z_dead_k

    def forward(
        self, x: Tensor, is_training: bool = True
    ) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input data tensor [batch_size × input_dim]
            is_training: Boolean flag indicating whether the model is in training mode.
                         If True, computes all three sparse representations. If False,
                         only computes the standard sparse representation.

        Returns:
            Tuple of (z_k, z_large_k, z_dead_k):
                - z_k: Standard sparse representation with topk activations [batch_size × hidden_dim]
                - z_large_k: Less sparse representation with larger_topk activations [batch_size × hidden_dim]
                              (None during inference)
                - z_dead_k: Representation focused on reviving dead neurons [batch_size × hidden_dim]
                              (None during inference)
        """
        # Encode input to get raw activations by subtracting bias from input
        # This centers the data before encoding
        z = self.encoder(x - self.bias)  # [batch_size × hidden_dim]

        # Generate three sparse representations with varying sparsity levels
        # Each representation serves a different purpose in the training process
        # During inference, only z_k is computed
        return self.top_k(z, is_training=is_training)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to sparse representation without updating statistics.
        Used for inference when we don't want to modify the model's state.

        Args:
            x: Input data tensor [batch_size × input_dim]

        Returns:
            Sparse representation with top-k activations [batch_size × hidden_dim]
            where only the highest k activations per sample are preserved
        """
        # Subtract bias from input before encoding (same as in forward method)
        z = self.encoder(x - self.bias)
        # Apply top-k sparsity without updating neuron activity statistics
        return self._compute_top_k_with_stats(z, self.topk, update_stats=False)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode the hidden representation back to the original input space.
        Adds the bias to ensure proper reconstruction.

        Args:
            z: Hidden representation tensor [batch_size × hidden_dim]

        Returns:
            Reconstructed input tensor [batch_size × input_dim]
        """
        # Add bias after decoding to compensate for bias in encoding
        return self.decoder(z) + self.bias

    def _normalized_mse(self, recon: Tensor, xs: Tensor) -> Tensor:
        """
        Compute normalized mean squared error.
        Normalizes by the variance of the input to make the loss scale-invariant.

        Args:
            recon: Reconstructed tensor [batch_size × input_dim]
            xs: Target tensor [batch_size × input_dim]

        Returns:
            Normalized MSE loss value (scalar)
        """
        # Calculate mean across batch dimension (baseline predictor)
        xs_mu = xs.mean(dim=0)  # [input_dim]

        # Normalize MSE by the MSE of predicting the mean (variance)
        # This makes the loss scale-invariant and more stable across different data scales
        loss = F.mse_loss(recon, xs) / F.mse_loss(xs_mu[None, :].expand_as(xs), xs)
        return loss

    def compute_loss(
        self,
        x: Tensor,
        z_k: Tensor,
        z_large_k: Tensor | None,
        z_dead_k: Tensor,
        ncl_temperature: float = 0.2,
        ncl_sim_threshold: float = 0.8,
    ) -> Dict[str, Tensor]:
        """
        Compute reconstruction losses and other regularization losses for the three different sparse representations.

        Args:
            x: Input data tensor [batch_size × input_dim]
            z_k: Regular sparse representation [batch_size × hidden_dim]
            z_large_k: Less sparse representation [batch_size × hidden_dim]
            z_dead_k: Dead neuron focused representation [batch_size × hidden_dim]
            ncl_temperature: Temperature parameter for contrastive loss (controls sharpness)
            ncl_sim_threshold: Similarity threshold for contrastive loss (pairs above this aren't negative samples)

        Returns:
            Dictionary with the different loss components (all scalar tensors)
        """
        # 1. Standard reconstruction loss using regular sparse representation
        # Decode the sparse representation and compute MSE loss normalized by input dimension
        recons_k = self.decode(z_k)
        loss_k = self.mse_scale * F.mse_loss(recons_k, x)

        # 2. Loss for the less sparse representation (should be lower than loss_k)
        # Using more active neurons should give better reconstruction quality
        loss_large_k = (
            torch.tensor(0.0, device=x.device)
            if z_large_k is None
            else self.mse_scale * F.mse_loss(self.decode(z_large_k), x)
        )

        # 3. Loss for dead neuron recovery
        # Uses normalized MSE on the residual error (what regular representation missed)
        # This encourages dead neurons to focus on details missed by active neurons
        recons_dead_k = self.decode(z_dead_k)
        loss_dead_k = self._normalized_mse(
            recons_dead_k, x - recons_k.detach() + self.bias.detach()
        ).nan_to_num(
            0
        )  # Handle potential NaN values from division

        # 4. Contrastive loss to prevent feature collapse
        # Choose between using regular or larger sparse representation based on config
        features = z_k if not self.ncl_with_larger_k or z_large_k is None else z_large_k
        if self.z_dead_k_in_positives_for_ncl:
            loss_ncl = self.info_nce_with_threshold(
                features,
                features + self._mask_z_dead_inplace(z_dead_k.clone()),
                temperature=ncl_temperature,
                threshold=ncl_sim_threshold,
            )
        else:
            loss_ncl = self.self_info_nce_with_threshold(
                features,
                temperature=ncl_temperature,
                threshold=ncl_sim_threshold,
            )
        # Calculate best_ncl_loss using F.cross_entropy with a single ideal example
        # Create ideal logits: high value (1/temperature) for positive, zeros for negatives
        ideal_logits = torch.zeros(1, x.shape[0], device=x.device)
        # Set the first position (positive pair) to 1/temperature
        ideal_logits[0, 0] = 1.0 / ncl_temperature
        # Label is 0 (the positive example)
        ideal_label = torch.zeros(1, device=x.device, dtype=torch.long)
        best_ncl_loss = F.cross_entropy(ideal_logits, ideal_label)
        loss_ncl = loss_ncl

        # Return all loss components in a dictionary for flexible weighting in the main loss function
        return {
            "reconstruct_loss_k": loss_k,
            "reconstruct_loss_large_k": loss_large_k,
            "reconstruct_loss_dead_k": loss_dead_k,
            "ncl_loss": loss_ncl,
            "best_ncl_loss": best_ncl_loss,
        }

    def self_info_nce_with_threshold(
        self,
        features: Tensor,
        temperature: float = 0.2,
        threshold: float = 0.8,
    ) -> Tensor:
        """
        InfoNCE loss that masks out negatives with cosine similarity > threshold.
        This is a modified contrastive loss that prevents treating similar samples as negatives.

        Args:
            features: Tensor of shape [batch_size × hidden_dim], raw embeddings from the batch
            temperature: Temperature scaling scalar. Lower values make the model more sensitive to differences
            threshold: Cosine-similarity cutoff for masking negatives. Pairs with similarity > threshold
                      are excluded from negative samples

        Returns:
            Scalar contrastive loss value
        """
        # 1. Normalize feature vectors to unit length for cosine similarity
        features = F.normalize(features, p=2, dim=1)  # [batch_size × hidden_dim]

        # 2. Compute cosine similarity matrix between all pairs, scaled by temperature
        sim_matrix = (
            torch.matmul(features, features.T) / temperature
        )  # [batch_size × batch_size]

        # 3. Build mask to identify pairs that are too similar to be treated as negatives
        with torch.no_grad():
            cosine = (
                sim_matrix * temperature
            )  # Undo temperature scaling to get raw cosine similarity
            # Identify pairs above similarity threshold (too similar to be negative samples)
            false_neg_mask = cosine > threshold  # [batch_size × batch_size]
            # Ensure self-similarities (diagonal) are handled by labels, not mask
            batch_size = features.size(0)
            diag = torch.eye(batch_size, device=features.device, dtype=torch.bool)
            false_neg_mask[diag] = (
                False  # Don't mask diagonal elements (self-similarities)
            )

        # 4. Mask out logits for samples that are too similar (set to -inf so softmax→0)
        masked_logits = sim_matrix.masked_fill(false_neg_mask, float("-inf"))

        # 5. Create labels: each sample's positive is itself (diagonal elements)
        labels = torch.arange(batch_size, device=features.device)

        # 6. Compute cross-entropy loss with masked logits
        # This treats each sample as its own positive and all others as negatives
        # except those masked out due to high similarity
        loss = F.cross_entropy(masked_logits, labels)
        return loss

    def info_nce_with_threshold(
        self,
        features: Tensor,
        positives: Tensor,
        temperature: float = 0.2,
        threshold: float = 0.8,
    ) -> Tensor:
        """
        InfoNCE loss that uses provided positive examples and masks out negatives with cosine similarity > threshold.
        Similar to self_info_nce_with_threshold, but uses external positive examples instead of self-similarity.

        The positive pairs are defined by cosine<feature_i, positive_i>, while the negative pairs
        are from in-batch features (cosine<feature_i, feature_j> where i≠j).

        Args:
            features: Tensor of shape [batch_size × hidden_dim], query embeddings from the batch
            positives: Tensor of shape [batch_size × hidden_dim], positive example embeddings
            temperature: Temperature scaling scalar. Lower values make the model more sensitive to differences
            threshold: Cosine-similarity cutoff for masking negatives. Pairs with similarity > threshold
                      are excluded from negative samples

        Returns:
            Scalar contrastive loss value
        """
        # 1. Normalize feature vectors to unit length for cosine similarity
        features = F.normalize(features, p=2, dim=1)  # [batch_size × hidden_dim]
        positives = F.normalize(positives, p=2, dim=1)  # [batch_size × hidden_dim]

        batch_size = features.size(0)

        # 2. Compute similarity between features and their positives (for positive pairs)
        pos_sim = torch.sum(features * positives, dim=1) / temperature  # [batch_size]

        # 3. Compute similarity matrix between features and all features (for negative pairs)
        neg_sim_matrix = (
            torch.matmul(features, features.T) / temperature
        )  # [batch_size × batch_size]

        # 4. Build mask to identify pairs that are too similar to be treated as negatives
        with torch.no_grad():
            cosine = (
                neg_sim_matrix * temperature
            )  # Undo temperature scaling to get raw cosine similarity
            # Identify pairs above similarity threshold (too similar to be negative samples)
            false_neg_mask = cosine > threshold  # [batch_size × batch_size]

        # 5. Create logits matrix by combining positive and negative similarities
        # First, mask out logits for samples that are too similar (set to -inf so softmax→0)
        masked_neg_sim = neg_sim_matrix.masked_fill(false_neg_mask, float("-inf"))

        # Create a matrix where each row contains the positive similarity followed by all negative similarities
        logits = torch.zeros(batch_size, batch_size + 1, device=features.device)
        logits[:, 0] = pos_sim  # First column contains positive similarities
        logits[:, 1:] = (
            masked_neg_sim  # Remaining columns contain negative similarities
        )

        # 6. Create labels: the positive example is at index 0 for each row
        labels = torch.zeros(batch_size, device=features.device, dtype=torch.long)

        # 7. Compute cross-entropy loss with the logits
        loss = F.cross_entropy(logits, labels)
        return loss

    def get_activation_histogram(self) -> Tuple[Tensor, Tensor]:
        """
        Get the histogram of non-zero activations for each hidden dimension.

        Returns:
            Tuple of Tensors:
            1/ The first Tensor (Shape: [hidden_dim]) containing the count of non-zero activations
            for each hidden dimension over the sliding window of the last activation_history_size batches.
            2/ The second Tensor is a vector contains a list activated column indeices within hidden_dim
            collected from the loop buffer.
        """
        return (
            self.stats_activation_counts.clone(),
            self.stats_activation_history.nonzero()[:, 1],
        )

    def _unit_norm_decoder(self) -> None:
        """
        Normalize the decoder weights to unit norm along each column.
        This ensures that each feature in the latent space has consistent influence
        on the reconstruction.
        """
        with torch.no_grad():
            self.decoder.weight /= self.decoder.weight.norm(dim=0, keepdim=True)

    def _mask_z_dead_inplace(self, z_dead: torch.Tensor) -> torch.Tensor:
        """
        In-place mask of z_dead: retains only the least positive element per row, zeroing out all else.
        Rows without positives remain all zeros.

        Args:
            z_dead (torch.Tensor): [B, h] tensor with non-negative values.

        Returns:
            torch.Tensor: The same tensor z_dead modified to keep only the minimal positives.
        """
        # Identify positive entries
        pos_mask = z_dead > 0  # [B, h]

        # Mark non-positives as +inf in-place to exclude them from min
        z_dead[~pos_mask] = float("inf")

        # Compute row-wise minimal values and indices
        min_vals, min_idx = z_dead.min(dim=1)  # [B], [B]

        # Zero out entire tensor
        z_dead.zero_()

        # Scatter back only where a positive existed (min_vals != inf)
        mask_rows = min_vals != float("inf")
        if mask_rows.any():
            batch_idx = torch.nonzero(mask_rows, as_tuple=True)[0]
            cols = min_idx[mask_rows]
            z_dead[batch_idx, cols] = min_vals[mask_rows]

        return z_dead


class SparseOrthogonalContrastiveAutoEncoder(SparseContrastiveAutoEncoder):
    """
    Sparse Orthogonal Contrastive AutoEncoder (SOCAE)
    ================================================

    A model that learns efficient sparse representations of input data.

    Mathematical Formulation:
    ------------------------
    x' = W_dec @ ReLU(TopK(W_enc @ (x - b) + b_enc)) + b

    * Perfect reconstruction (x' = x) occurs when:
      - No sparsity constraint (ReLU + TopK) is applied
      - W_dec @ W_enc = I (identity matrix)
      - No b_enc

    Learning Objectives:
    ------------------
    1. High-quality reconstruction
       - Minimizes reconstruction error between input and output

    2. Manifold preservation
       - Maintains local data structure
       - Uses self non-negative contrastive loss and orthogonal regularization

    3. Efficiency for downstream tasks
       - Achieves (implicitly) balanced, sparse activations across latent space
       - Using dead neuron recovery and orthogonal regularization
       - TODO: Directly optimize balanced activations across latent space

    For more details: [Design Doc](https://fburl.com/gdoc/u4rfv97z)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        topk: int,
        larger_topk: int,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initializes the SparseOrthogonalContrastiveAutoEncoder model.

        Args:
            input_dim (int): The dimension of the input data.
            hidden_dim (int): The dimension of the hidden representation (latent space).
            topk (int): The number of top activations to keep in normal operation (enforces sparsity).
            larger_topk (int): The number of top activations for the larger sparse representation (less sparse).
            **kwargs: Additional arguments to pass to the [parent class constructor](https://fburl.com/code/6u0vvlxf).
                      This allows for flexibility when new parameters are added to the base class.
                      Common kwargs include:

                      - dead_topk (int, optional): The number of top activations for dead neuron recovery.
                        If not provided, defaults to 2^(floor(log2(input_dim // 2))).

                      - dead_threshold (int, optional): The threshold to consider a neuron as dead
                        (number of batches without activation). Defaults to 64.

                      - ncl_with_larger_k (bool, optional): Whether to use z_larger_k for NCL loss.
                        Defaults to False.

                      - track_dead_neuron_pre_relu (bool, optional): Whether to track neuron activity before applying ReLU.
                        If True, neuron activity is tracked on the raw values before ReLU.
                        If False, neuron activity is tracked after applying ReLU.
                        Defaults to True.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            topk=topk,
            larger_topk=larger_topk,
            # pyre-ignore[6]: Intentionally passing **kwargs to parent method which handles extraction of typed parameters
            **kwargs,
        )
        # Initialize decoder weights to be as orthogonal as possible to improve representation efficiency
        nn.init.orthogonal_(self.decoder.weight)
        self._unit_norm_decoder()

    # pyre-ignore[14]: `compute_loss` overrides method defined in
    #  `SparseContrastiveAutoEncoder` inconsistently.
    def compute_loss(
        self,
        x: Tensor,
        z_k: Tensor,
        z_large_k: Tensor,
        z_dead_k: Tensor,
        *,  # Force keyword arguments after this
        enable_orth_reg: bool = True,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        """
        Compute reconstruction losses and other regularization losses for the three different sparse representations.

        Args:
            x: Input data tensor [batch_size × input_dim]
            z_k: Regular sparse representation [batch_size × hidden_dim]
            z_large_k: Less sparse representation [batch_size × hidden_dim]
            z_dead_k: Dead neuron focused representation [batch_size × hidden_dim]
            enable_orth_reg: Whether to apply orthogonality regularization to decoder weights
            **kwargs: Additional arguments to pass to the parent class compute_loss method: https://fburl.com/code/6u0vvlxf
                      This allows for flexibility when new parameters are added to the base class.
                      Common kwargs include:

                      - ncl_temperature (float, optional): Temperature parameter for contrastive loss.
                        Controls the sharpness of the similarity distribution. Defaults to 0.2.

                      - ncl_sim_threshold (float, optional): Similarity threshold for contrastive loss.
                        Pairs with similarity above this threshold aren't treated as negative samples.
                        Defaults to 0.8.

        Returns:
            Dictionary with the different loss components (all scalar tensors)
        """
        # Get base losses from parent class implementation (reconstruction and contrastive losses)
        base_loss = super().compute_loss(
            x,
            z_k,
            z_large_k,
            z_dead_k,
            # pyre-ignore[6]: Intentionally passing **kwargs to parent method which handles extraction of typed parameters
            **kwargs,
        )

        # Add orthogonality regularization to encourage more efficient representations
        if enable_orth_reg:
            loss_orth = self._compute_loss_orth_unit(self.decoder.weight.T)

        else:
            # No orthogonality regularization if disabled
            loss_orth = torch.tensor(0.0)
        # Add orthogonality loss to the loss dictionary
        base_loss["orth_loss"] = loss_orth

        # Return complete loss dictionary for flexible weighting in the training loop
        return base_loss

    @classmethod
    def _compute_loss_orth_unit(cls, W: Tensor) -> Tensor:
        """
        Compute orthogonality loss for weight matrix W in a memory-efficient way.

        This method calculates two components:
        1. L_norm: Penalizes deviation from unit norm for each row vector
        2. L_orth: Penalizes non-orthogonality between different row vectors
           without explicitly computing the full Gram matrix

        Args:
            W: Weight matrix of shape [hidden_dim, input_dim]

        Returns:
            Combined orthogonality loss (L_norm + L_orth)
        """
        hidden_dim, _ = W.shape

        # Calculate row-wise L2 norms
        norms = W.norm(dim=1)
        # Penalize deviation from unit norm
        L_norm = ((norms - 1) ** 2).sum() / hidden_dim

        # Compute W^T @ W efficiently
        C = W.T @ W
        # Sum of all squared elements in the Gram matrix
        total_sq = C.pow(2).sum()
        # Sum of squared diagonal elements (equals sum of norms^4)
        diag_sq = norms.pow(4).sum()
        # Sum of squared off-diagonal elements (represents non-orthogonality)
        off_sq = total_sq - diag_sq
        # Normalize by number of off-diagonal elements
        L_orth = off_sq / (hidden_dim * (hidden_dim - 1))

        return L_norm + L_orth
