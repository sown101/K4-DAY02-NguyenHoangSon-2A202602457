"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99
  - KHÔNG tính tiền xử lý (đọc ảnh, resize, normalize): chỉ đo forward của model trên tensor đã ở GPU.
"""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây) với warmup và đồng bộ trước/sau mỗi lần đo."""
    if iters < 50:
        raise ValueError("cần >= 50 lần đo")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def _device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    import platform
    return f"CPU ({platform.processor() or platform.machine()})"


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, k_views: int = 1,
                   channels_last: bool = False) -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16) | "fp16" (model.half(); model bị đổi tại chỗ, hãy truyền bản sao).
    k_views > 1: mỗi lần đo chạy K lượt forward (TTA thật, không nhân ước lượng).
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    use_amp = dtype == "amp"
    dev_type = "cuda" if device.startswith("cuda") else "cpu"

    def fn():
        with torch.inference_mode(), torch.autocast(dev_type, dtype=torch.float16, enabled=use_amp):
            for _ in range(k_views):
                model(x)

    sync = torch.cuda.synchronize if dev_type == "cuda" else None
    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {"gpu": _device_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": fused_bn, "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            "mean": r["mean"], "n": r["n"], "images_per_s": batch_size / (r["p50"] / 1000.0),
            "torch": torch.__version__, "preprocessing": "không tính"}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view đo thật, kèm so sánh với K x p50 của 1 view (slide trang 63)."""
    one = latency_report(model, k_views=1, **kw)
    k = latency_report(model, k_views=k_views, **kw)
    k["p50_1view"] = one["p50"]
    k["ratio_vs_k_times_1view"] = k["p50"] / (k_views * one["p50"])
    return k
