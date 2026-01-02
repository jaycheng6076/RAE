# Copyright (c) Meta Platforms.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from stage1 import RAE
from stage1.bottleneck.socae import (
    SparseContrastiveAutoEncoder,
    SparseOrthogonalContrastiveAutoEncoder,
)
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import wandb_utils
from utils.model_utils import get_obj_from_str, instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SOCAE bottleneck on frozen encoder outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config containing socae and encoder sections.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Directory with ImageFolder structure.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ckpts",
        help="Directory to store training outputs.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image resolution (assumes square images).",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=None,
        help="Override training.global_seed from the config.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging if set.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch compile for encoder.",
    )
    return parser.parse_args()


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def build_encoder(encoder_config: dict, device: torch.device) -> RAE:
    """Build and freeze the encoder from RAE config."""
    rae: RAE = instantiate_from_config(encoder_config).to(device)
    rae.encoder.eval()
    rae.encoder.requires_grad_(False)
    return rae


def build_socae(
    socae_config: dict, device: torch.device
) -> SparseOrthogonalContrastiveAutoEncoder:
    """Build the SOCAE model from config."""
    if "target" not in socae_config:
        raise KeyError("Expected key 'target' in socae config.")
    model_cls = get_obj_from_str(socae_config["target"])
    params = socae_config.get("params", {})
    model = model_cls(**params).to(device)
    return model


def main():
    args = parse_args()

    # Distributed init
    rank, world_size, device = setup_distributed()

    # Config init
    full_cfg = OmegaConf.load(args.config)

    # Get encoder config (for extracting latents)
    encoder_section = full_cfg.get("encoder", None)
    if encoder_section is None:
        raise ValueError(
            "Config must define an 'encoder' section for extracting latents."
        )
    encoder_config = OmegaConf.to_container(encoder_section, resolve=True)

    # Get SOCAE config
    socae_section = full_cfg.get("socae", None)
    if socae_section is None:
        raise ValueError(
            "Config must define a 'socae' section for the bottleneck model."
        )
    socae_config = OmegaConf.to_container(socae_section, resolve=True)

    # Training config
    training_section = full_cfg.get("training", None)
    training_cfg = (
        OmegaConf.to_container(training_section, resolve=True)
        if training_section is not None
        else {}
    )
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    # Loss weights from config
    loss_section = full_cfg.get("loss", {})
    loss_cfg = (
        OmegaConf.to_container(loss_section, resolve=True) if loss_section else {}
    )
    recon_k_weight = float(loss_cfg.get("recon_k_weight", 1.0))
    recon_large_k_weight = float(loss_cfg.get("recon_large_k_weight", 0.1))
    recon_dead_k_weight = float(loss_cfg.get("recon_dead_k_weight", 0.1))
    ncl_weight = float(loss_cfg.get("ncl_weight", 0.01))
    orth_weight = float(loss_cfg.get("orth_weight", 0.01))
    ncl_temperature = float(loss_cfg.get("ncl_temperature", 0.2))
    ncl_sim_threshold = float(loss_cfg.get("ncl_sim_threshold", 0.8))
    enable_orth_reg = bool(loss_cfg.get("enable_orth_reg", True))

    # Training hyperparameters
    batch_size = int(training_cfg.get("batch_size", 256))
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert (
            global_batch_size % world_size == 0
        ), "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size

    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 100))
    default_seed = int(training_cfg.get("global_seed", 0))
    simulated_annealing = bool(training_cfg.get("simulated_annealing", False))
    init_from_data = bool(training_cfg.get("init_from_data", True))
    init_samples = int(training_cfg.get("init_samples", 100000))

    # Seed setup
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Experiment directories
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir

    # Build encoder (frozen)
    encoder_rae = build_encoder(encoder_config, device)
    if args.compile:
        encoder_rae.encode = torch.compile(encoder_rae.encode)
    encoder_rae.eval()

    # Build SOCAE
    socae = build_socae(socae_config, device)
    ema_model = deepcopy(socae).to(device).eval()
    ema_model.requires_grad_(False)

    ddp_model = DDP(
        socae,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    socae = ddp_model.module

    # Optimizer
    optimizer, optim_msg = build_optimizer(socae.parameters(), training_cfg)

    # AMP setup
    scaler, autocast_kwargs = get_autocast_scaler(args)

    # Data setup
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(
                first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )
    loader, sampler = prepare_dataloader(
        args.data_path,
        batch_size,
        num_workers,
        rank,
        world_size,
        transform=stage1_transform,
    )

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError(
            "Dataloader returned zero batches. Check dataset and batch size settings."
        )

    # Scheduler
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    # Resume checkpoint
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(
            f"Resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming..."
        )
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(
                f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step})."
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    # Initialize SOCAE from data samples (for geometric median bias, etc.)
    if init_from_data and start_epoch == 0:
        logger.info(f"Initializing SOCAE from {init_samples} data samples...")
        init_loader = DataLoader(
            loader.dataset,
            batch_size=min(batch_size, 512),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        collected_samples = []
        with torch.no_grad():
            for images, _ in init_loader:
                images = images.to(device, non_blocking=True)
                z = encoder_rae.encode(images)
                if encoder_rae.reshape_to_2d:
                    b, c, h, w = z.shape
                    z = z.view(b, c, h * w).transpose(1, 2).reshape(-1, c)
                else:
                    z = z.reshape(-1, z.shape[-1])
                collected_samples.append(z.cpu())
                if sum(s.shape[0] for s in collected_samples) >= init_samples:
                    break
        all_samples = torch.cat(collected_samples, dim=0)[:init_samples].to(device)
        socae.init_from_data(all_samples, scale_mse=False, init_pre_bias=True)
        del collected_samples, all_samples
        torch.cuda.empty_cache()
        logger.info("SOCAE initialization from data completed.")

    # Simulated annealing setup
    if simulated_annealing and start_epoch == 0:
        socae.simulated_annealing_init(batch_size)
        logger.info(
            f"Simulated annealing initialized with starting k multiplier: {socae.starting_k_multipler}"
        )

    # Logging
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"SOCAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(
            f"SOCAE architecture: input_dim={socae.input_dim}, hidden_dim={socae.hidden_dim}, topk={socae.topk}"
        )
        logger.info(
            f"Loss weights: recon_k={recon_k_weight}, recon_large_k={recon_large_k_weight}, "
            f"recon_dead_k={recon_dead_k_weight}, ncl={ncl_weight}, orth={orth_weight}"
        )
        logger.info(
            f"NCL temperature: {ncl_temperature}, NCL sim threshold: {ncl_sim_threshold}"
        )
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        logger.info(optim_msg)
        if sched_msg:
            logger.info(sched_msg)
        logger.info(
            f"Training for {num_epochs} epochs, batch size {batch_size} per GPU."
        )
        logger.info(
            f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch."
        )

    dist.barrier()

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(1, device=device)
        )
        num_batches = 0

        # Save checkpoint at epoch start
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )

        for step, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(**autocast_kwargs):
                # Get encoder outputs (frozen)
                with torch.no_grad():
                    z = encoder_rae.encode(images)
                    if encoder_rae.reshape_to_2d:
                        b, c, h, w = z.shape
                        z = z.view(b, c, h * w).transpose(1, 2)  # [B, N, C]
                    # Flatten for SOCAE: [B*N, C]
                    z_flat = z.reshape(-1, z.shape[-1])

                # Forward through SOCAE (training mode returns all three representations)
                z_k, z_large_k, z_dead_k = ddp_model(z_flat, is_training=True)

                # Compute losses
                losses = socae.compute_loss(
                    z_flat,
                    z_k,
                    z_large_k,
                    z_dead_k,
                    enable_orth_reg=enable_orth_reg,
                    ncl_temperature=ncl_temperature,
                    ncl_sim_threshold=ncl_sim_threshold,
                )

                # Weighted total loss
                total_loss = (
                    recon_k_weight * losses["reconstruct_loss_k"]
                    + recon_large_k_weight * losses["reconstruct_loss_large_k"]
                    + recon_dead_k_weight * losses["reconstruct_loss_dead_k"]
                    + ncl_weight * losses["ncl_loss"]
                    + orth_weight
                    * losses.get("orth_loss", torch.tensor(0.0, device=device))
                )

            # Backward
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Update EMA
            update_ema(ema_model, ddp_model.module, ema_decay)

            # Unit norm decoder weights
            socae._unit_norm_decoder()

            # Simulated annealing update
            if simulated_annealing:
                dead_count = (
                    (socae.stats_last_nonzero > socae.dead_threshold).sum().item()
                )
                socae.simulated_annealing(dead_count)

            # Track metrics
            epoch_metrics["total"] += total_loss.detach()
            epoch_metrics["recon_k"] += losses["reconstruct_loss_k"].detach()
            epoch_metrics["recon_large_k"] += losses[
                "reconstruct_loss_large_k"
            ].detach()
            epoch_metrics["recon_dead_k"] += losses["reconstruct_loss_dead_k"].detach()
            epoch_metrics["ncl"] += losses["ncl_loss"].detach()
            if "orth_loss" in losses:
                epoch_metrics["orth"] += losses["orth_loss"].detach()
            num_batches += 1

            # Logging
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                dead_neurons = (
                    (socae.stats_last_nonzero > socae.dead_threshold).sum().item()
                )
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon_k": losses["reconstruct_loss_k"].detach().item(),
                    "loss/recon_large_k": losses["reconstruct_loss_large_k"]
                    .detach()
                    .item(),
                    "loss/recon_dead_k": losses["reconstruct_loss_dead_k"]
                    .detach()
                    .item(),
                    "loss/ncl": losses["ncl_loss"].detach().item(),
                    "loss/best_ncl": losses["best_ncl_loss"].detach().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "topk": socae.topk,
                    "dead_neurons": dead_neurons,
                    "dead_ratio": dead_neurons / socae.hidden_dim,
                }
                if "orth_loss" in losses:
                    stats["loss/orth"] = losses["orth_loss"].detach().item()

                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in stats.items()
                    )
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)

            global_step += 1

        # Epoch summary
        if rank == 0 and num_batches > 0:
            epoch_stats = {
                "epoch/loss_total": (epoch_metrics["total"] / num_batches).item(),
                "epoch/loss_recon_k": (epoch_metrics["recon_k"] / num_batches).item(),
                "epoch/loss_recon_large_k": (
                    epoch_metrics["recon_large_k"] / num_batches
                ).item(),
                "epoch/loss_recon_dead_k": (
                    epoch_metrics["recon_dead_k"] / num_batches
                ).item(),
                "epoch/loss_ncl": (epoch_metrics["ncl"] / num_batches).item(),
            }
            if epoch_metrics["orth"].item() > 0:
                epoch_stats["epoch/loss_orth"] = (
                    epoch_metrics["orth"] / num_batches
                ).item()

            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    # Save final checkpoint
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
