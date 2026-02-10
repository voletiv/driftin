"""Train drifting model with multi-resolution feature encoders on multi-GPU DDP.

Optimized for throughput:
- Pre-computes real image features at startup (encoder is frozen + deterministic)
- Reduced encoder input (112x112 default, not 224)
- torch.compile on encoder
- bf16 autocast on encoder forward

Launch:
    torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_multires_ddp \
        --encoder dinov2-multires --batch-size 128 --steps 50000 \
        --output-dir outputs/drift_dinov2_multires
"""

import os
import time
import csv
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .config import UNetConfig, UNetLargeConfig, MultiResDriftConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.encoders import build_encoder
from .training.compute_v import compute_drift_multitemp_batched
from .eval.sample import drift_sample, save_sample_grid
from .utils.plot_loss import plot_loss_curve
from .utils.samples_to_gif import samples_to_gif


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def all_gather_flat(tensor):
    """All-gather tensors across all ranks. Returns concatenated along dim 0."""
    tensor = tensor.contiguous()
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def precompute_features(encoder, dataset, device, batch_size=256, rank=0, is_main=False):
    """Pre-compute features for all images in the dataset.

    Returns a list of tensors [N_total, L, C] for each feature group,
    plus the C_j values. Stored on CPU to save GPU memory.
    """
    if is_main:
        print("Pre-computing real image features...")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    all_feats_by_group = None
    n_processed = 0

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for images, _ in loader:
            images = images.to(device)
            groups = encoder(images)

            if all_feats_by_group is None:
                all_feats_by_group = [[] for _ in groups]

            for i, (feat, C_j) in enumerate(groups):
                all_feats_by_group[i].append(feat.float().cpu())

            n_processed += images.shape[0]
            if is_main and n_processed % 10000 == 0:
                print(f"  {n_processed}/{len(dataset)} images")

    # Concatenate and store
    result_feats = []
    result_cjs = []
    for i, feat_list in enumerate(all_feats_by_group):
        cat = torch.cat(feat_list, dim=0)  # [N_total, L, C]
        result_feats.append(cat)
        result_cjs.append(groups[i][1])

    if is_main:
        total_bytes = sum(f.numel() * f.element_size() for f in result_feats)
        print(f"  Pre-computed {n_processed} images, {total_bytes / 1e9:.2f} GB on CPU")
        for i, (f, c) in enumerate(zip(result_feats, result_cjs)):
            print(f"    Group {i}: shape={list(f.shape)}, C_j={c}")

    return result_feats, result_cjs


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """Dataset that returns pre-computed features by index."""

    def __init__(self, feature_tensors, cjs):
        self.feature_tensors = feature_tensors  # list of [N, L, C] on CPU
        self.cjs = cjs
        self.n = feature_tensors[0].shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return tuple of feature tensors for this sample
        return tuple(ft[idx] for ft in self.feature_tensors)


def get_cifar10(batch_size, world_size, rank, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if rank == 0:
        datasets.CIFAR10(root="./data", train=True, download=True)
    dist.barrier()
    dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return loader, sampler, dataset


def multires_drift_loss_distributed(gen_groups, pos_groups, temps, rank, world_size):
    """Compute multi-resolution drifting loss with globally-gathered features."""
    total_loss = 0.0
    n_groups = len(gen_groups)

    for (gen_feat, C_j), (pos_feat, _) in zip(gen_groups, pos_groups):
        B_local = gen_feat.shape[0]

        with torch.no_grad():
            gen_det = gen_feat.detach()
            pos_det = pos_feat.detach()

            global_gen = all_gather_flat(gen_det)
            global_pos = all_gather_flat(pos_det)

            gen_t = global_gen.transpose(0, 1).contiguous()
            pos_t = global_pos.transpose(0, 1).contiguous()

            V = compute_drift_multitemp_batched(gen_t, pos_t, temps=temps)

            V_local = V[:, rank * B_local : (rank + 1) * B_local, :]
            target = gen_det + V_local.transpose(0, 1)

        total_loss = total_loss + F.mse_loss(gen_feat, target)

    return total_loss / n_groups


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        print(f"DDP: {world_size} GPUs")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = MultiResDriftConfig()
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.encoder:
        cfg.encoder = args.encoder
    if args.temps:
        cfg.temperatures = [float(t) for t in args.temps.split(",")]
    if args.pool_size:
        cfg.pool_size = args.pool_size

    encoder_input_size = args.encoder_size or 112

    if is_main:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        stamp = None
    stamp_list = [stamp]
    dist.broadcast_object_list(stamp_list, src=0)
    stamp = stamp_list[0]

    out_dir = args.output_dir
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # UNet
    unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
    ).to(memory_format=torch.channels_last).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"UNet parameters: {n_params:,}")

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    # Multi-res encoder (built on rank 0 first to download weights)
    if rank == 0:
        feat_encoder = build_encoder(cfg.encoder, pool_size=cfg.pool_size,
                                     input_size=encoder_input_size).to(device)
    dist.barrier()
    if rank != 0:
        feat_encoder = build_encoder(cfg.encoder, pool_size=cfg.pool_size,
                                     input_size=encoder_input_size).to(device)
    dist.barrier()
    feat_encoder.eval()

    # Compile the encoder for faster forward/backward
    feat_encoder = torch.compile(feat_encoder)

    if is_main:
        n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
        print(f"Feature encoder: {cfg.encoder}, {n_feat_params:,} params (frozen, compiled)")
        print(f"  Input size: {encoder_input_size}x{encoder_input_size}")

    # ---- Pre-compute real features (the big optimization) ----
    transform_noflip = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if rank == 0:
        full_dataset = datasets.CIFAR10(root="./data", train=True, download=False,
                                        transform=transform_noflip)
    dist.barrier()
    if rank != 0:
        full_dataset = datasets.CIFAR10(root="./data", train=True, download=False,
                                        transform=transform_noflip)

    precomp_feats, precomp_cjs = precompute_features(
        feat_encoder, full_dataset, device, batch_size=512, rank=rank, is_main=is_main
    )

    # Also compute flipped versions for augmentation
    transform_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if rank == 0:
        flip_dataset = datasets.CIFAR10(root="./data", train=True, download=False,
                                        transform=transform_flip)
    dist.barrier()
    if rank != 0:
        flip_dataset = datasets.CIFAR10(root="./data", train=True, download=False,
                                        transform=transform_flip)

    flip_feats, _ = precompute_features(
        feat_encoder, flip_dataset, device, batch_size=512, rank=rank, is_main=is_main
    )

    # Stack: [2, N_total, L, C] for each group -- original + flipped
    stacked_feats = []
    for orig, flipped in zip(precomp_feats, flip_feats):
        stacked_feats.append(torch.stack([orig, flipped], dim=0))  # [2, 50000, L, C]

    del precomp_feats, flip_feats
    if is_main:
        print("Pre-computation complete. Real features cached on CPU.")

    # Data loader for images (we still need images for gen, and indices for feature lookup)
    img_loader, img_sampler, _ = get_cifar10(cfg.batch_size, world_size, rank)

    # EMA
    ema = EMA(model.module, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )

    scaler = torch.amp.GradScaler("cuda")

    # Logging
    log_path = os.path.join(out_dir, "loss_log.csv")
    if is_main:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "time_s", "images_per_sec", "timestamp"])
        global_bs = cfg.batch_size * world_size
        print(f"Training Drifting (multi-res {cfg.encoder}) for {cfg.total_steps} steps")
        print(f"  per-GPU batch={cfg.batch_size}, global batch={global_bs}")
        print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
        print(f"  encoder input={encoder_input_size}x{encoder_input_size}")
        print(f"  output dir: {out_dir}")

    # Index tracking for pre-computed features
    # We need to know which CIFAR indices the DataLoader returns.
    # Simplest: use a custom dataset that returns (image, index)
    class IndexedCIFAR10(torch.utils.data.Dataset):
        def __init__(self, root, train, transform):
            self.ds = datasets.CIFAR10(root=root, train=train, download=False, transform=transform)
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            return img, idx

    indexed_dataset = IndexedCIFAR10("./data", train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
    indexed_sampler = DistributedSampler(indexed_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    indexed_loader = DataLoader(
        indexed_dataset, batch_size=cfg.batch_size, sampler=indexed_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    start_time = time.time()
    epoch = 0
    data_iter = iter(indexed_loader)

    z_sample = torch.randn(64, unet_cfg.in_ch, unet_cfg.image_size, unet_cfg.image_size, device=device)

    for step in range(1, cfg.total_steps + 1):
        try:
            real_images, indices = next(data_iter)
        except StopIteration:
            epoch += 1
            indexed_sampler.set_epoch(epoch)
            data_iter = iter(indexed_loader)
            real_images, indices = next(data_iter)

        B = real_images.shape[0]

        # Look up pre-computed real features (with random flip augmentation)
        flip_choice = torch.randint(0, 2, (B,))  # 0=original, 1=flipped
        pos_groups = []
        for i, C_j in enumerate(precomp_cjs):
            # stacked_feats[i]: [2, 50000, L, C]
            feats = stacked_feats[i][flip_choice, indices]  # [B, L, C]
            pos_groups.append((feats.to(device), C_j))

        # Generate images from noise
        z = torch.randn_like(real_images, device=device).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = model(z)

        # Extract features from generated images (with grad, through encoder)
        gen_f32 = gen_images.float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_groups_raw = feat_encoder(gen_f32)
        # Cast back to float32 for drift computation
        gen_groups = [(f.float(), c) for f, c in gen_groups_raw]

        # Compute drift loss with global all-gather
        loss = multires_drift_loss_distributed(
            gen_groups, pos_groups,
            temps=tuple(cfg.temperatures),
            rank=rank, world_size=world_size,
        )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model.module)

        if (step == 1 or step % cfg.log_every == 0) and is_main:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            imgs_per_sec = steps_per_sec * cfg.batch_size * world_size
            eta_h = (cfg.total_steps - step) / steps_per_sec / 3600
            print(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"step {step:>7d}/{cfg.total_steps} | "
                f"loss {loss.item():.4f} | "
                f"{steps_per_sec:.1f} it/s ({imgs_per_sec:.0f} img/s) | "
                f"ETA {eta_h:.1f}h"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.6f}", f"{elapsed:.1f}", f"{imgs_per_sec:.0f}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            plot_loss_curve(log_path)

        if (step == 1 or step % cfg.sample_every == 0) and is_main:
            model.eval()
            samples = drift_sample(ema.shadow, 64, device, z=z_sample)
            save_sample_grid(
                samples,
                os.path.join(out_dir, "samples", f"drift_step{step:07d}.png"),
            )
            samples_to_gif(os.path.join(out_dir, "samples"), duration=300)
            model.train()
            print(f"  Saved sample grid at step {step}")

        if step % cfg.save_every == 0 and is_main:
            ckpt = {
                "step": step,
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "drift": cfg},
            }
            path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    if is_main:
        elapsed = time.time() - start_time
        print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
        torch.save({
            "step": cfg.total_steps,
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "config": {"unet": unet_cfg, "drift": cfg},
        }, os.path.join(out_dir, "checkpoints", "drift_final.pt"))

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/drift_multires")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--encoder", type=str, default="dinov2-multires",
                        choices=["dinov2-multires", "convnextv2", "mocov2",
                                 "dinov3", "eva02", "siglip2", "clip"])
    parser.add_argument("--encoder-size", type=int, default=112,
                        help="Encoder input resolution (default 112, was 224)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size")
    parser.add_argument("--temps", type=str, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    train(args)
