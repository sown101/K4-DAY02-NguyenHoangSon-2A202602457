"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Mỗi lần chạy ghi vào <out_dir>/<exp_id>/seed<k>/:
    config.json, history.csv, lr_steps.npy, best.pt, last.pt (để resume khi Colab ngắt),
    val_logits.npy (+ test_logits.npy nếu save_test_predictions), summary.json
và predictions/<exp_id>_seed<k>_val.csv (+ _test.csv), curves/<exp_id>_<desc>[_seed<k>].png.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import random
import sys
import time
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as ds  # noqa: E402
import losses  # noqa: E402
import model as mdl  # noqa: E402


def _import_eval():
    """Tìm eval.py của repo gốc (đi ngược lên từ thư mục code/) và import nó."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "eval.py").exists():
            sys.path.insert(0, str(parent))
            break
    import eval as ev
    return ev


ev = _import_eval()


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    pred_tag: str | None = None       # tiền tố file predictions (mặc định = exp_id)
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    eval_crop_pct: float = 0.875
    aug: str = "basic"                # basic | color | trivial | randaug | flipv
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    cache: bool = False               # nạp trước bytes JPEG vào RAM
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # ce_weighted: None -> 1/n_c; focal: None -> không alpha
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    eval_batch_size: int = 128
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    deterministic: bool = False
    device: str = "auto"
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    resume: bool = True
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- chỉ để gỡ lỗi (KHÔNG dùng cho số liệu báo cáo) ---
    debug_subset: int | None = None   # lấy N ảnh đầu mỗi tập sau khi kiểm tra split
    max_steps_per_epoch: int | None = None


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.pred_tag or cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    desc = cfg.desc or cfg.backbone
    suffix = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU+CUDA). Worker DataLoader được seed trong dataset.seed_worker.

    Mặc định cudnn.benchmark=True (nhanh) nên không tái lập từng bit; deterministic=True thì ngược lại.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD+momentum, trục E) với các nhóm tham số của model.param_groups."""
    groups = mdl.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer không hợp lệ: {cfg.optimizer}")


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số LR theo BƯỚC: warmup tuyến tính 0 -> 1, rồi cosine 1 -> 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo bước (iteration)."""
    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warmup))


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước tối ưu (slide trang 56).

    EMA trên toàn bộ state_dict: tham số và cả buffer BN (running_mean/var) cùng được làm trơn, nên
    thống kê BN khớp với trọng số EMA. Buffer kiểu int (num_batches_tracked) thì chép thẳng.
    Decay được "warmup" d_t = min(d, (1 + t) / (10 + t)) để EMA không bị kéo về trọng số khởi tạo.
    """

    def __init__(self, model, decay: float):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]


def _dev_type(device) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, epoch: int = 0, lr_log: list | None = None) -> dict:
    """Một epoch. Trả về {"train_loss", "train_acc" (NaN nếu Mixup/CutMix), "lr"}."""
    mdl.set_train_mode(model)
    rng = np.random.default_rng(cfg.seed * 1000 + epoch)
    use_amp = cfg.amp and _dev_type(device) == "cuda"
    tot_loss, tot_correct, n = 0.0, 0, 0
    for step, (x, y, _) in enumerate(loader):
        if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
            break
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y).to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = None
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng)
        with torch.autocast(_dev_type(device), dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} ở epoch {epoch}, bước {step}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if lr_log is not None:
            lr_log.append(optimizer.param_groups[0]["lr"])
        bs = y.shape[0]
        tot_loss += loss.item() * bs
        if not cfg.mix:
            tot_correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return {"train_loss": tot_loss / max(n, 1),
            "train_acc": float("nan") if cfg.mix else tot_correct / max(n, 1),
            "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device, amp: bool = True):
    """Chạy model ở eval, không gradient. Trả về (filenames, y_true, logits[N, 9], loss CE).

    Loss val luôn là cross-entropy thường (không smoothing/focal) để so sánh được giữa các thí nghiệm;
    `criterion` giữ trong chữ ký cho tương thích, không dùng.
    """
    from inference import predict_logits
    names, y, logits = predict_logits(model, loader, device, amp=amp)
    loss = F.cross_entropy(torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(y)).item()
    return names, y, logits, loss


def metrics_from_logits(y, logits) -> dict:
    from inference import softmax_np
    probs = softmax_np(logits)
    return ev.compute_metrics(np.asarray(y), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Loss train/val, macro-F1/top-1 val (và acc train) theo epoch, LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ncol = 3 if lr_steps is not None and len(lr_steps) else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4))
    ax[0].plot(h["epoch"], h["train_loss"], "o-", label="train loss (loss huấn luyện)")
    ax[0].plot(h["epoch"], h["val_loss"], "s-", label="val loss (CE)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1")
    ax[1].plot(h["epoch"], h["val_top1"], "s-", label="val top-1")
    if h["train_acc"].notna().any():
        ax[1].plot(h["epoch"], h["train_acc"], "^--", alpha=0.7, label="train acc (có augmentation)")
    best = h.loc[h["val_macro_f1"].idxmax()]
    ax[1].axvline(best["epoch"], color="gray", ls=":", label=f"best ep {int(best['epoch'])}: F1 {best['val_macro_f1']:.4f}")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metric")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    if ncol == 3:
        ax[2].plot(np.arange(len(lr_steps)), lr_steps)
        ax[2].set(xlabel="bước (iteration)", ylabel="LR backbone", title="LR (warmup + cosine)")
        ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _env() -> dict:
    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _build_criterion(cfg: Config, train_labels):
    counts = np.bincount(np.asarray(train_labels), minlength=ds.NUM_CLASSES)
    if cfg.loss == "ce_weighted":
        return losses.build_criterion("ce_weighted", weight=losses.class_weights(counts, cfg.class_weight_beta or 0.0))
    if cfg.loss == "focal":
        alpha = None if cfg.class_weight_beta is None else losses.class_weights(counts, cfg.class_weight_beta)
        return losses.build_criterion("focal", gamma=cfg.focal_gamma, alpha=alpha)
    if cfg.loss == "ls":
        return losses.build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    return losses.build_criterion(cfg.loss)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    Test chỉ được chạy MỘT lần ở cuối, khi cfg.save_test_predictions=True (Bước 4).
    """
    set_seed(cfg.seed, cfg.deterministic)
    device = _resolve_device(cfg.device)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps({**dataclasses.asdict(cfg), "env": _env()}, indent=2, ensure_ascii=False))

    # 1-2. dữ liệu + kiểm tra split (dừng nếu vi phạm)
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    split_report = ds.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    if cfg.debug_subset:
        train_df, val_df, test_df = (d.head(cfg.debug_subset) for d in (train_df, val_df, test_df))

    # 4. model (trước loader để lấy mean/std đúng với trọng số)
    model = mdl.build_model(cfg.backbone, True, ds.NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate)
    mean = tuple(model.pretrained_cfg.get("mean", ds.IMAGENET_MEAN))
    std = tuple(model.pretrained_cfg.get("std", ds.IMAGENET_STD))
    n_params = mdl.count_params(model)
    gmacs = mdl.count_gmacs(model, cfg.img_size)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)

    # 3. loader
    train_tf = ds.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    eval_tf = ds.build_transforms(False, cfg.img_size, "basic", mean, std, cfg.eval_crop_pct)
    train_loader = ds.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, True, cfg.sampler,
                                  cfg.num_workers, cfg.seed, cfg.cache)
    val_loader = ds.make_loader(val_df, cfg.images_dir, eval_tf, cfg.eval_batch_size, False,
                                num_workers=cfg.num_workers, cache=cfg.cache)
    steps = len(train_loader) if not cfg.max_steps_per_epoch else min(len(train_loader), cfg.max_steps_per_epoch)

    criterion = _build_criterion(cfg, train_df["Label"]).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps)
    use_amp = cfg.amp and _dev_type(device) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, lr_steps, best_f1, best_epoch, start_epoch = [], [], -1.0, -1, 0
    last_ckpt, best_ckpt = rd / "last.pt", rd / "best.pt"
    if cfg.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        history, lr_steps = ck["history"], ck["lr_steps"]
        best_f1, best_epoch, start_epoch = ck["best_f1"], ck["best_epoch"], ck["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] resume từ epoch {start_epoch}")

    # 5. vòng epoch, chọn checkpoint theo MACRO-F1 VAL (hòa -> giữ epoch sớm hơn vì dùng '>')
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device,
                             ema, epoch, lr_steps)
        train_time = time.time() - t0
        eval_model = ema.module if ema is not None else model
        _, yv, lv, vloss = evaluate(eval_model, val_loader, None, device, cfg.amp)
        m = metrics_from_logits(yv, lv)
        row = {"epoch": epoch + 1, **tr, "val_loss": vloss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"], "train_time_s": train_time,
               "epoch_time_s": time.time() - t0}
        history.append(row)
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch + 1
            torch.save(eval_model.state_dict(), best_ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "history": history,
                    "lr_steps": lr_steps, "best_f1": best_f1, "best_epoch": best_epoch, "epoch": epoch}, last_ckpt)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch + 1:2d}/{cfg.epochs} train_loss {tr['train_loss']:.4f} "
              f"val_loss {vloss:.4f} val_F1 {m['macro_f1']:.4f} val_top1 {m['top1']:.4f} "
              f"({row['epoch_time_s']:.0f}s)", flush=True)

    # 6. nạp checkpoint tốt nhất, lưu logit + predictions val
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    names_v, yv, lv, vloss = evaluate(model, val_loader, None, device, cfg.amp)
    mv = metrics_from_logits(yv, lv)
    np.save(rd / "val_logits.npy", lv)
    pd.DataFrame({"Filename": names_v, "y_true": yv}).to_csv(rd / "val_index.csv", index=False)
    from inference import softmax_np
    ev.save_predictions(pred_path(cfg, "val"), names_v, yv, softmax_np(lv))

    # 7. test: đúng MỘT lần, chỉ ở Bước 4
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(test_df, cfg.images_dir, eval_tf, cfg.eval_batch_size, False,
                                     num_workers=cfg.num_workers)
        names_t, yt, lt, _ = evaluate(model, test_loader, None, device, cfg.amp)
        np.save(rd / "test_logits.npy", lt)
        pd.DataFrame({"Filename": names_t, "y_true": yt}).to_csv(rd / "test_index.csv", index=False)
        ev.save_predictions(pred_path(cfg, "test"), names_t, yt, softmax_np(lt))

    # 8. history, đường cong, tóm tắt
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_steps))
    plot_curves(history, curve_path(cfg),
                f"{cfg.exp_id} | {model.arch_name} | seed {cfg.seed} | {cfg.desc or 'baseline recipe'}", lr_steps)

    lat_b1 = float("nan")
    if _dev_type(device) == "cuda":
        from benchmark import latency_report
        lat_b1 = latency_report(model, 1, cfg.img_size, "fp32", device, warmup=10, iters=50)["p50"]

    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": model.arch_name, "weight_tag": model.weight_tag,
               "init": cfg.init, "desc": cfg.desc, "params_M": round(n_params, 3), "gmacs": round(gmacs, 3),
               "img_size": cfg.img_size, "epochs": cfg.epochs, "best_epoch": best_epoch,
               "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_balanced_acc": mv["balanced_acc"],
               "val_ece": mv["ece"], "val_f1_chinee": float(mv["f1"][0]), "val_f1_snake": float(mv["f1"][7]),
               "train_s_per_epoch": float(np.mean([h["train_time_s"] for h in history])),
               "latency_b1_p50_ms": lat_b1, "split_n": split_report["n"],
               "debug": bool(cfg.debug_subset or cfg.max_steps_per_epoch)}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    last_ckpt.unlink(missing_ok=True)  # chỉ cần để resume; giữ best.pt cho Bước 3 (TTA, độ trễ)
    all_csv = Path(cfg.out_dir) / "summary_all.csv"
    row = pd.DataFrame([{**{k: v for k, v in summary.items() if k != "split_n"}, "config": json.dumps(dataclasses.asdict(cfg))}])
    row.to_csv(all_csv, mode="a", header=not all_csv.exists(), index=False)
    print(f"[{cfg.exp_id} seed{cfg.seed}] XONG: best ep {best_epoch}, val macro-F1 {mv['macro_f1']:.4f}, "
          f"top-1 {mv['top1']:.4f}")
    return summary


def _cast(tp, value: str):
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value.lower() in ("none", "null") and type(None) in args:
            return None
        for t in args:
            if t is type(None):
                continue
            try:
                return _cast(t, value)
            except ValueError:
                continue
        raise ValueError(f"không ép được '{value}' sang {tp}")
    if tp is bool:
        if value.lower() in ("1", "true", "yes", "y"):
            return True
        if value.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"'{value}' không phải bool")
    if tp in (int, float, str):
        return tp(value)
    raise ValueError(f"kiểu chưa hỗ trợ: {tp}")


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    hints = typing.get_type_hints(Config)
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' phải có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        if key not in hints:
            raise KeyError(f"Config không có trường '{key}'. Các trường: {sorted(hints)}")
        out[key] = _cast(hints[key], value)
    return out


def main(argv=None) -> None:
    """`python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một thí nghiệm DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
