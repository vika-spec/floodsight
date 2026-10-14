"""Train the Siamese U-Net on Kuro Siwo with an event-level split.

Laptop (RTX 5050, 8 GB):
    python -m floodsight.model.train --root D:/data/KuroSiwo --epochs 15 --batch 8 --pretrained \\
        --val-events 1111013 --test-events 427 1111006 --max-per-event 5000

Tips
  * Install PyTorch built for CUDA 12.8+ (RTX 50-series): https://pytorch.org/get-started/locally/
  * Out of memory? --batch 4. Windows DataLoader trouble? --workers 0.
  * Best checkpoint (by validation flood IoU) -> models/siamese_unet.pt, picked up by the pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import IGNORE, KuroSiwo, find_samples, split_by_event
from .metrics import confusion, scores


def _disable_quickedit() -> None:
    """Windows: clicking in a console with QuickEdit on FREEZES the program until a key is pressed.
    Turn QuickEdit off for this console while training runs."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)                     # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value | 0x0080) & ~0x0040)   # extended flags on, QuickEdit off
    except Exception:  # noqa: BLE001
        pass


def _bar(it, **kw):
    from tqdm import tqdm
    return tqdm(it, dynamic_ncols=True, leave=False, **kw)


def dice_loss(logits, target, cls: int, ignore: int = IGNORE):
    import torch

    valid = target != ignore
    p = torch.softmax(logits.float(), 1)[:, cls]
    t = (target == cls).float()
    p, t = p[valid], t[valid]
    inter = (p * t).sum()
    return 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)


def terrain_loss(logits, post, target, max_slope_deg: float, ignore: int = IGNORE):
    """Physics prior: still flood water cannot lie on ground steeper than the water rule allows. Mean flood
    probability on such pixels (slope is input channel 2, stored as degrees / 30). Zero if there are none."""
    import torch

    steep = (post[:, 2] * 30.0 > max_slope_deg) & (target != ignore)
    if not steep.any():
        return logits.sum() * 0.0
    p = torch.softmax(logits.float(), 1)[:, 2]
    return p[steep].mean()


def evaluate(model, loader, device, n_classes=3, desc="validate"):
    import torch

    model.eval()
    cm = np.zeros((n_classes, n_classes), np.int64)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for pre, post, lab in _bar(loader, desc=desc, unit="batch"):
            pred = model(pre.to(device), post.to(device)).argmax(1).cpu().numpy()
            cm += confusion(pred, lab.numpy(), n_classes, IGNORE)
    return cm


def main(argv=None):
    import warnings

    import torch

    # harmless: with mixed precision the first optimizer step can be skipped while the loss scale calibrates
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")
    from torch.utils.data import DataLoader

    from .net import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="models/siamese_unet.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--arch", default="siamese",
                    choices=["siamese", "early_fusion", "post_only", "siamese_unetpp", "siamese_attn_unet",
                             "siamese_deeplabv3p", "siamese_segformer", "segformer_early"],
                    help="network (model comparison; the production model is siamese)")
    ap.add_argument("--terrain-loss", type=float, default=0.0,
                    help="weight of a physics penalty on flood predicted on slopes steeper than water can lie "
                         "(detection.water_max_slope_deg); 0 = off")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--val-events", nargs="*")
    ap.add_argument("--test-events", nargs="*", help="events never used for training or model selection")
    ap.add_argument("--exclude-events", nargs="*", default=[],
                    help="events left out completely (kept as unseen events for later honest tests)")
    ap.add_argument("--px", type=float, default=10.0, help="Kuro Siwo pixel size in metres")
    ap.add_argument("--max-train", type=int, default=0, help="subsample training tiles (quick experiments)")
    ap.add_argument("--max-val", type=int, default=0, help="subsample validation tiles (quick experiments)")
    ap.add_argument("--max-per-event", type=int, default=0,
                    help="cap training tiles per event so one large flood does not dominate")
    ap.add_argument("--event-level", type=int, help="folder depth that identifies the flood event (see --inspect)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.last.pt after an interruption")
    ap.add_argument("--sanity", action="store_true",
                    help="pipeline check only: allow a random tile split when there are <3 events (scores NOT valid)")
    a = ap.parse_args(argv)

    _disable_quickedit()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    samples = find_samples(a.root, a.event_level)
    if a.exclude_events:
        samples = [s for s in samples if str(s["event"]) not in {str(e) for e in a.exclude_events}]
        print(f"excluded events (never seen): {', '.join(map(str, a.exclude_events))}")
    n_events = len({s["event"] for s in samples})
    if n_events < 3 and not a.sanity:
        raise SystemExit(f"Only {n_events} flood event(s) detected: an event-level split is impossible. "
                         "Download more shards, or add --sanity to test the training pipeline only.")
    if n_events < 3:
        print("WARNING --sanity: random tile split inside one event. Use this only to check that the pipeline "
              "trains; the validation score is NOT a valid measure of generalisation.")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(samples))
        cut = int(len(samples) * 0.85)
        sp = {"train": [samples[i] for i in idx[:cut]], "val": [samples[i] for i in idx[cut:]], "test": []}
    else:
        sp = split_by_event(samples, a.val_events, a.test_events)
    if a.max_val and len(sp["val"]) > a.max_val:
        sp["val"] = list(np.random.default_rng(1).choice(sp["val"], a.max_val, replace=False))
    train = sp["train"]
    if a.max_per_event:
        rng = np.random.default_rng(2)
        by_ev = defaultdict(list)
        for s_ in train:
            by_ev[s_["event"]].append(s_)
        train = []
        for ev, lst in sorted(by_ev.items()):
            if len(lst) > a.max_per_event:
                lst = list(rng.choice(lst, a.max_per_event, replace=False))
            train += lst
    if a.max_train and len(train) > a.max_train:
        train = list(np.random.default_rng(0).choice(train, a.max_train, replace=False))
    print(f"tiles  train {len(train)}  val {len(sp['val'])}  test {len(sp['test'])}")
    for k, lst in (("train", train), ("val", sp["val"]), ("test", sp["test"])):
        cnt = defaultdict(int)
        for s_ in lst:
            cnt[s_["event"]] += 1
        print(f"  {k:5} events: {dict(sorted(cnt.items()))}")

    dl = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    tr = DataLoader(KuroSiwo(train, augment=True, px_m=a.px), batch_size=a.batch, shuffle=True, drop_last=True, **dl)
    va = DataLoader(KuroSiwo(sp["val"], px_m=a.px), batch_size=a.batch, **dl)

    # class weights from label frequencies (flood pixels are rare)
    freq = np.ones(3)
    for s in train[: min(300, len(train))]:
        from .dataset import load_sample
        lab = load_sample(s, a.px)[2]
        freq += np.bincount(lab[lab != IGNORE].ravel(), minlength=3)[:3]
    w = torch.tensor(np.clip((freq.sum() / freq) ** 0.5, 1, 20) / 3, dtype=torch.float32, device=device)
    print("class weights:", w.cpu().numpy().round(2))

    model = build_model(a.arch, a.encoder, a.pretrained).to(device).to(memory_format=torch.channels_last)
    from ..core import load_config
    max_slope = float(load_config()["detection"]["water_max_slope_deg"])
    print(f"architecture: {a.arch}  ({sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters)")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = torch.nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    last = out.with_suffix(".last.pt")
    best, history, start = -1.0, [], 0
    if a.resume:
        if not last.exists():
            raise SystemExit(f"--resume: {last} not found (it is written after each epoch of a run).")
        st = torch.load(last, map_location=device, weights_only=False)
        if st.get("epochs", a.epochs) != a.epochs or st.get("steps_per_epoch", len(tr)) != len(tr):
            raise SystemExit(f"--resume: use the same --epochs ({st.get('epochs')}), --batch and data options "
                             "as the interrupted run (the learning-rate schedule depends on them).")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        best, history, start = st["best"], st["history"], st["epoch"]
        print(f"Resumed after epoch {start} (best val flood IoU so far {best:.3f})")

    run_t0 = time.time()
    for ep in range(start, a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        pbar = _bar(tr, desc=f"epoch {ep + 1}/{a.epochs} train", unit="batch")
        for pre, post, lab in pbar:
            pre, post, lab = pre.to(device, non_blocking=True), post.to(device, non_blocking=True), lab.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pre, post)
                loss = ce(logits, lab) + dice_loss(logits, lab, 2)
                if a.terrain_loss:
                    loss = loss + a.terrain_loss * terrain_loss(logits, post, lab, max_slope)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach().item()
            n += 1
            if n % 20 == 0:
                pbar.set_postfix(loss=f"{tot / n:.3f}")
        sc = scores(evaluate(model, va, device, desc=f"epoch {ep + 1}/{a.epochs} validate"))
        flood_iou = sc["per_class"][2]["iou"]
        history.append({"epoch": ep + 1, "loss": tot / max(len(tr), 1), "val_flood_iou": flood_iou,
                        "val_miou": sc["miou"], "sec": round(time.time() - t0)})
        done = ep + 1 - start
        eta = (time.time() - run_t0) / done * (a.epochs - ep - 1)
        h = history[-1]
        print(f"[{datetime.now():%H:%M}] epoch {ep + 1}/{a.epochs}  loss {h['loss']:.3f}  "
              f"val flood IoU {flood_iou:.3f}  val mIoU {h['val_miou']:.3f}  "
              f"({h['sec'] / 60:.1f} min, ~{eta / 60:.0f} min left)", flush=True)
        Path(out.with_suffix(".history.json")).write_text(json.dumps(history, indent=1))   # survives interruption
        if flood_iou > best:
            best = flood_iou
            torch.save({"state_dict": model.state_dict(), "encoder": a.encoder, "arch": a.arch, "px_m": a.px,
                        "terrain_loss": a.terrain_loss,
                        "budget": {"epochs": a.epochs, "train_tiles": len(train), "max_per_event": a.max_per_event,
                                   "batch": a.batch, "lr": a.lr, "pretrained": a.pretrained,
                                   "train_minutes_so_far": round(sum(x["sec"] for x in history) / 60, 1)},
                        "epoch": ep + 1, "val": sc, "split": {k: sorted({s['event'] for s in v}) for k, v in sp.items()}},
                       out)
            print(f"  * new best -> saved {out}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best, "history": history,
                    "epochs": a.epochs, "steps_per_epoch": len(tr)}, last)

    if sp["test"]:
        from .evaluate import evaluate_events, print_table
        ck = torch.load(out, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        res = evaluate_events(model, sp["test"], device, a.batch, a.px, dl)
        print("\nHELD-OUT TEST EVENTS (never used for training or model selection):")
        print_table(res)
        Path(out.with_suffix(".test.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
