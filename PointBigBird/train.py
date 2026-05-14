"""End-to-end JEPA training with the PointBigBird backbone on CIFAR-10."""
from __future__ import annotations

import os, time, copy, argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pbb import (
    PBBConfig, PBBEncoder, PBBPredictor,
    build_loaders, orderings_from_batch,
    TargetCenter, ema_update, make_momentum_schedule,
    gather_target_features, jepa_loss, diag_dict, fmt_diag,
    save_atomic, ensure_dir, short_params,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    args = p.parse_args()

    cfg = PBBConfig()
    if args.epochs:     cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr = args.lr
    if args.ckpt_dir:   cfg.ckpt_dir = args.ckpt_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg.ckpt_dir)

    train_loader, train_eval_loader, test_loader = build_loaders(cfg)
    print(f"loaders: train={len(train_loader.dataset)}  "
          f"train_eval={len(train_eval_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    # Models
    context_encoder = PBBEncoder(
        d_model=cfg.d_model, n_layers=cfg.n_layers_enc,
        n_heads=cfg.n_heads, dim_head=cfg.dim_head,
        block_size=cfg.block_size, window=cfg.window,
        n_random=cfg.n_random, n_global=cfg.n_global,
        ffn_mult=cfg.ffn_mult, fourier_dim=cfg.fourier_dim,
        fourier_scale=cfg.fourier_scale,
        serial_orders=cfg.serial_orders,
    ).to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for q in target_encoder.parameters(): q.requires_grad_(False)
    predictor = PBBPredictor(
        d_model=cfg.d_model, d_pred=cfg.d_pred,
        n_layers=cfg.n_layers_pred,
        n_heads=cfg.n_heads_pred, dim_head=cfg.dim_head_pred,
        fourier_dim=cfg.fourier_dim, fourier_scale=cfg.fourier_scale,
        ffn_mult=cfg.ffn_mult,
    ).to(device)
    center = TargetCenter(cfg.d_model, momentum=cfg.center_momentum).to(device)

    print(f"context_encoder: {short_params(context_encoder)}  "
          f"predictor: {short_params(predictor)}")

    params = list(context_encoder.parameters()) + list(predictor.parameters())
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    sched = CosineAnnealingLR(opt, T_max=total_steps)
    mgen  = make_momentum_schedule(cfg.ema_start, cfg.ema_end, total_steps)

    history = {"loss": [], "probe_acc": []}
    global_step = 0
    best_loss = float("inf")
    m = cfg.ema_start

    for epoch in range(1, cfg.epochs + 1):
        context_encoder.train(); predictor.train()
        epoch_loss, steps = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            ctx_p = batch["ctx_pixels"].to(device)
            ctx_c = batch["ctx_coords"].to(device)
            pool_p = batch["pool_pixels"].to(device)
            pool_c = batch["pool_coords"].to(device)
            tgt_c  = batch["tgt_coords"].to(device)
            tgt_pp = batch["tgt_pool_pos"].to(device)
            ctx_ords  = {k: {kk: vv.to(device) for kk, vv in v.items()}
                          for k, v in orderings_from_batch(batch, "ctx").items()}
            pool_ords = {k: {kk: vv.to(device) for kk, vv in v.items()}
                          for k, v in orderings_from_batch(batch, "pool").items()}

            with torch.no_grad():
                g_tgt = target_encoder(pool_p, pool_c, pool_ords)
                h_tgt_raw = gather_target_features(g_tgt, tgt_pp)
                center.update(h_tgt_raw)
                h_tgt = F.layer_norm(center(h_tgt_raw), (h_tgt_raw.size(-1),))

            g_ctx = context_encoder(ctx_p, ctx_c, ctx_ords)
            h_pred = predictor(g_ctx, tgt_c)

            loss = jepa_loss(h_pred, h_tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            try:    m = next(mgen)
            except StopIteration: m = cfg.ema_end
            ema_update(target_encoder, context_encoder, m)

            global_step += 1
            epoch_loss += loss.item(); steps += 1

            if global_step % cfg.log_every == 0:
                d = diag_dict(loss, h_pred, h_tgt, g_ctx, center)
                print(fmt_diag(d, global_step, epoch, sched.get_last_lr()[0], m))

        avg = epoch_loss / max(steps, 1)
        history["loss"].append(avg)
        print(f"=== ep {epoch:03d}/{cfg.epochs}  avg_loss={avg:.4f}  "
              f"m={m:.4f}  {time.time()-t0:.1f}s ===")

        if avg < best_loss:
            best_loss = avg
            save_atomic({
                "epoch": epoch, "cfg": cfg.__dict__,
                "context_encoder": context_encoder.state_dict(),
                "target_encoder":  target_encoder.state_dict(),
                "predictor":       predictor.state_dict(),
                "center":          center.state_dict(),
                "opt":             opt.state_dict(),
                "sched":           sched.state_dict(),
                "global_step":     global_step,
                "history":         history,
            }, os.path.join(cfg.ckpt_dir, "pbb_best.pt"))
        save_atomic({"epoch": epoch, "context_encoder": context_encoder.state_dict()},
                    os.path.join(cfg.ckpt_dir, "pbb_last.pt"))


if __name__ == "__main__":
    main()
