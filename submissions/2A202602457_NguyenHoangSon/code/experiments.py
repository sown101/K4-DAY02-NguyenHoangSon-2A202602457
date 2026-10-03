"""experiments.py - Bước 3 (suy luận), Bước 4 (chung kết), Bước 5 (biểu đồ, results.xlsx).

Mọi hàm dùng lại dataset/model/train/inference/benchmark; notebook chỉ gọi các hàm ở đây.

Quy tắc (README S2, S4):
  - Bước 3 chỉ dùng VAL. Nhiệt độ T khớp trên VAL.
  - Bước 4: `finalize` tính dự đoán TEST đúng một lần cho mỗi (exp_id, seed); nếu file test đã tồn tại
    thì từ chối chạy lại (trừ khi force=True, và khi đó phải ghi lý do vào báo cáo).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

import benchmark as bm
import dataset as ds
import inference as inf
import model as mdl
import train as tr

ev = tr.ev


# --------------------------------------------------------------------------- #
# Nạp mô hình đã huấn luyện
# --------------------------------------------------------------------------- #
def is_vit(name: str) -> bool:
    return any(k in name for k in ("vit", "deit"))


def load_run(run_dir, device: str = "cuda", dynamic_img_size: bool | None = None):
    """Đọc config.json + best.pt của một lần chạy. Trả về (cfg, model ở eval mode).

    ViT/DeiT được tạo với dynamic_img_size=True (nội suy position embedding) để chạy được
    ở độ phân giải khác 224 (I04); ở 224 kết quả giống hệt model gốc.
    """
    run_dir = Path(run_dir)
    d = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    d.pop("env", None)
    cfg = tr.Config(**d)
    arch = mdl.resolve_name(cfg.backbone)
    kw = dict(pretrained=False, num_classes=ds.NUM_CLASSES)
    if dynamic_img_size if dynamic_img_size is not None else is_vit(arch):
        kw["dynamic_img_size"] = True
    model = timm.create_model(arch, **kw)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.arch_name = arch
    return cfg, model.to(device).eval()


def split_frame(cfg, split: str) -> pd.DataFrame:
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    df = {"train": train_df, "val": val_df, "test": test_df}[split]
    return df.head(cfg.debug_subset) if cfg.debug_subset else df  # debug_subset chỉ dùng khi gỡ lỗi


def eval_loader(cfg, model, split: str, img_size: int = 224, crop_pct: float = 0.875,
                batch_size: int = 128, num_workers: int | None = None):
    mean = tuple(model.pretrained_cfg.get("mean", ds.IMAGENET_MEAN))
    std = tuple(model.pretrained_cfg.get("std", ds.IMAGENET_STD))
    tf = ds.build_transforms(False, img_size, "basic", mean, std, crop_pct)
    nw = cfg.num_workers if num_workers is None else num_workers
    return ds.make_loader(split_frame(cfg, split), cfg.images_dir, tf, batch_size, False, num_workers=nw)


def metrics(y, probs) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    m = ev.compute_metrics(np.asarray(y), probs.argmax(1), probs)
    return {"macro_f1": m["macro_f1"], "top1": m["top1"], "balanced_acc": m["balanced_acc"], "ece": m["ece"],
            "nll": m["nll"], "f1_chinee": float(m["f1"][0]), "f1_snake": float(m["f1"][7])}


# --------------------------------------------------------------------------- #
# Logit theo phương pháp suy luận (dùng chung cho Bước 3 và Bước 4)
# --------------------------------------------------------------------------- #
METHODS = {
    "1view": "1 view: resize + center crop (I00)",
    "hflip": "TTA lật ngang, K=2",
    "crop5": "TTA 5 crop 224 từ ảnh 256, K=5",
    "crop10": "TTA 5 crop + lật, K=10",
}


def method_logits(model, cfg, split: str, method: str, space: str = "logit", device: str = "cuda",
                  img_size: int = 224, amp: bool = False):
    """Trả về (filenames, y, logits_or_logprobs[N, 9], K).

    space="logit": trung bình logit (đầu ra là logit, temperature scaling áp dụng trực tiếp).
    space="prob":  trung bình xác suất, trả về log(prob) để vẫn khớp T được (softmax(log p / T)).
    method "res<r>" = 1 view ở độ phân giải r (I04). amp=False: tính FP32 (chuẩn tham chiếu).
    """
    if method.startswith("res"):
        r = int(method[3:])
        names, y, lg = inf.predict_logits(model, eval_loader(cfg, model, split, r), device, amp=amp)
        return names, y, lg, 1
    if method == "1view":
        names, y, lg = inf.predict_logits(model, eval_loader(cfg, model, split, img_size), device, amp=amp)
        return names, y, lg, 1
    if method == "hflip":
        names, y, outs = inf.predict_views(model, eval_loader(cfg, model, split, img_size), device,
                                           [None, inf.view_hflip], amp=amp)
    elif method in ("crop5", "crop10"):
        full = round(img_size / 0.875)  # ảnh gốc 256, giữ nguyên rồi cắt 5 góc
        loader = eval_loader(cfg, model, split, full, crop_pct=1.0)
        names, y, outs = inf.predict_views(model, loader, device,
                                           inf.multicrop_view_fns(img_size, flip=method == "crop10"), amp=amp)
    else:
        raise ValueError(f"method không hợp lệ: {method}")
    agg = np.mean(np.stack(outs), 0) if space == "logit" else inf.probs_to_logits(inf.aggregate_views(outs, "prob"))
    return names, y, agg, len(outs)


# --------------------------------------------------------------------------- #
# Bước 3: bộ thí nghiệm suy luận trên VAL
# --------------------------------------------------------------------------- #
LAT_ITERS, THR_BATCH = 100, 32   # giảm khi chạy thử trên CPU


def _lat(model, img_size, k=1, dtype="fp32", batch=1, device="cuda", fused=False):
    r = bm.latency_report(copy.deepcopy(model) if dtype == "fp16" else model, batch, img_size, dtype, device,
                          k_views=k, fused_bn=fused, iters=LAT_ITERS)
    return r


def _ens_latency(models, img_size, device="cuda", batch=1):
    x = torch.randn(batch, 3, img_size, img_size, device=device)

    def fn():
        with torch.inference_mode():
            for m in models:
                m(x)
    r = bm.bench(fn, 10, LAT_ITERS, torch.cuda.synchronize if device.startswith("cuda") else None)
    r["images_per_s"] = batch / (r["p50"] / 1000)
    return r


def run_inference_suite(run_dir, out_dir, ensemble_runs=(), ema_run=None, bn_run=None,
                        resolutions=(192, 256, 288), device="cuda") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy I00-I08 trên VAL cho model ở `run_dir`. Lưu Inference.csv, Latency.csv, val_probs.npz.

    ensemble_runs: các run_dir khác để ensemble với model chính (I05, dùng val_logits.npy đã lưu).
    ema_run: run_dir của thí nghiệm có EMA (I06, so với model chính không EMA).
    bn_run:  run_dir của một CNN có BatchNorm để thử gộp BN (I08b); DeiT/ConvNeXt không có BN.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, model = load_run(run_dir, device)
    exp = f"{cfg.exp_id}/seed{cfg.seed}"
    rows, lat_rows, store = [], [], {}

    def add(code, name, probs, k, lat, ckpt=exp, note=""):
        m = metrics(y, probs)
        store[code] = probs
        rows.append({"exp_id": code, "phương pháp": name, "mô hình/checkpoint": ckpt, "K": k,
                     "macro_f1_val": m["macro_f1"], "top1_val": m["top1"], "ece_val": m["ece"],
                     "f1_chinee_val": m["f1_chinee"], "f1_snake_val": m["f1_snake"],
                     "p50_ms_b1": lat["p50"] if lat else np.nan, "p95_ms_b1": lat["p95"] if lat else np.nan,
                     "p99_ms_b1": lat["p99"] if lat else np.nan,
                     "throughput_img_s_b32": lat.get("thr32", np.nan) if lat else np.nan, "ghi chú": note})

    def lat_pair(img, k=1, dtype="fp32", m=None, fused=False, label=""):
        m = m if m is not None else model
        a = _lat(m, img, k, dtype, 1, device, fused)
        b = _lat(m, img, k, dtype, THR_BATCH, device, fused)
        for r in (a, b):
            lat_rows.append({"cấu hình": label, "gpu": r["gpu"], "dtype": r["dtype"], "batch": r["batch"],
                             "img_size": img, "K": k, "gộp BN": "có" if fused else "không", "p50_ms": r["p50"],
                             "p95_ms": r["p95"], "p99_ms": r["p99"], "images_per_s": r["images_per_s"],
                             "torch": r["torch"], "tiền xử lý": "không tính"})
        return {"p50": a["p50"], "p95": a["p95"], "p99": a["p99"], "thr32": b["images_per_s"]}

    # I00 mốc
    names, y, lg0, _ = method_logits(model, cfg, "val", "1view", device=device)
    add("I00", METHODS["1view"], inf.softmax_np(lg0), 1, lat_pair(cfg.img_size, label="I00 1 view fp32"))

    # I01 / I02 / I03 TTA, gộp xác suất vs logit
    for code, meth in (("I01", "hflip"), ("I02a", "crop5"), ("I02b", "crop10")):
        _, _, lp, k = method_logits(model, cfg, "val", meth, "prob", device)
        lat = lat_pair(cfg.img_size, k, label=f"{code} {meth} fp32")
        add(code, METHODS[meth] + " (gộp xác suất)", inf.softmax_np(lp), k, lat)
        _, _, ll, _ = method_logits(model, cfg, "val", meth, "logit", device)
        add(code.replace("I01", "I03a").replace("I02a", "I03b").replace("I02b", "I03c"),
            METHODS[meth] + " (gộp logit)", inf.softmax_np(ll), k, lat)
    flips = inf.softmax_np(lg0).argmax(1), store["I01"].argmax(1)
    w2r = int(((flips[0] != y) & (flips[1] == y)).sum())
    r2w = int(((flips[0] == y) & (flips[1] != y)).sum())
    rows[1]["ghi chú"] = f"so với I00: {w2r} ảnh sai->đúng, {r2w} ảnh đúng->sai"

    # I04 độ phân giải kiểm tra
    for r in resolutions:
        _, _, lr, _ = method_logits(model, cfg, "val", f"res{r}", device=device)
        add(f"I04_{r}", f"1 view ở {r} px (train 224)", inf.softmax_np(lr), 1, lat_pair(r, label=f"I04 {r}px fp32"))

    # I05 ensemble (val_logits.npy đã lưu, cùng thứ tự val)
    if ensemble_runs:
        ref_idx = pd.read_csv(Path(run_dir) / "val_index.csv")["Filename"].tolist()
        probs = [inf.softmax_np(lg0)]
        models = [model]
        tags = [exp]
        for rd in ensemble_runs:
            rd = Path(rd)
            assert pd.read_csv(rd / "val_index.csv")["Filename"].tolist() == ref_idx, f"{rd}: thứ tự val khác"
            probs.append(inf.softmax_np(np.load(rd / "val_logits.npy")))
            models.append(load_run(rd, device)[1])
            tags.append(f"{rd.parent.name}/{rd.name}")
        e1 = _ens_latency(models, cfg.img_size, device, 1)
        e32 = _ens_latency(models, cfg.img_size, device, THR_BATCH)
        lat_rows.append({"cấu hình": "I05 ensemble fp32", "gpu": bm._device_name(device), "dtype": "fp32", "batch": 1,
                         "img_size": cfg.img_size, "K": len(models), "gộp BN": "không", "p50_ms": e1["p50"],
                         "p95_ms": e1["p95"], "p99_ms": e1["p99"], "images_per_s": e1["images_per_s"],
                         "torch": torch.__version__, "tiền xử lý": "không tính"})
        add("I05", f"Ensemble {len(models)} mô hình (trung bình xác suất)", inf.ensemble_probs(probs), len(models),
            {"p50": e1["p50"], "p95": e1["p95"], "p99": e1["p99"], "thr32": e32["images_per_s"]}, " + ".join(tags))
        del models[1:]

    # I06 EMA (trọng số EMA từ lúc train, không tốn thêm khi suy luận)
    if ema_run:
        ema_lg = np.load(Path(ema_run) / "val_logits.npy")
        add("I06", "Trọng số EMA (train có EMA, cùng công thức còn lại)", inf.softmax_np(ema_lg), 1,
            rows[0] | {"p50": rows[0]["p50_ms_b1"], "p95": rows[0]["p95_ms_b1"], "p99": rows[0]["p99_ms_b1"],
                       "thr32": rows[0]["throughput_img_s_b32"]}, f"{Path(ema_run).parent.name}", "cùng kiến trúc nên độ trễ = I00")

    # I07 temperature scaling: T khớp trên toàn bộ VAL; ECE "trung thực" bằng cross-fit 2 nửa val
    T = inf.fit_temperature(lg0, y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    half = [idx[: len(y) // 2], idx[len(y) // 2:]]
    cross = np.empty_like(lg0)
    for a, b in ((0, 1), (1, 0)):
        t_ab = inf.fit_temperature(lg0[half[a]], y[half[a]])
        cross[half[b]] = lg0[half[b]] / t_ab
    ece_cross = metrics(y, inf.softmax_np(cross))["ece"]
    lat0 = {"p50": rows[0]["p50_ms_b1"], "p95": rows[0]["p95_ms_b1"], "p99": rows[0]["p99_ms_b1"],
            "thr32": rows[0]["throughput_img_s_b32"]}
    add("I07", f"Temperature scaling T={T:.3f} (khớp trên val)", inf.apply_temperature(lg0, T), 1, lat0,
        note=f"ECE trước {rows[0]['ece_val']:.4f}; ECE cross-fit 2 nửa val {ece_cross:.4f}; accuracy không đổi")

    # I08 FP16 / AMP
    half_model = copy.deepcopy(model).half()
    _, _, lg16 = inf.predict_logits(half_model, eval_loader(cfg, model, "val"), device, view=lambda x: x.half(), amp=False)
    del half_model
    add("I08a", "FP16 (model.half())", inf.softmax_np(lg16), 1, lat_pair(cfg.img_size, dtype="fp16", label="I08a fp16"),
        note=f"số ảnh đổi nhãn so với FP32: {int((lg16.argmax(1) != lg0.argmax(1)).sum())}")
    _, _, lgamp, _ = method_logits(model, cfg, "val", "1view", device=device, amp=True)
    add("I08b", "AMP autocast fp16", inf.softmax_np(lgamp), 1, lat_pair(cfg.img_size, dtype="amp", label="I08b amp"),
        note=f"số ảnh đổi nhãn so với FP32: {int((lgamp.argmax(1) != lg0.argmax(1)).sum())}")

    # I08c gộp BN trên một CNN có BN (DeiT không có BN)
    if bn_run:
        bcfg, bmodel = load_run(bn_run, device)
        _, by, blg, _ = method_logits(bmodel, bcfg, "val", "1view", device=device)
        x = torch.randn(2, 3, bcfg.img_size, bcfg.img_size, device=device)
        fused = inf.fuse_conv_bn(bmodel, check_input=x)
        _, _, flg, _ = method_logits(fused, bcfg, "val", "1view", device=device)
        lb = _lat(bmodel, bcfg.img_size, device=device)
        lf = _lat(fused, bcfg.img_size, device=device, fused=True)
        for lab, r, f in (("I08c trước gộp BN", lb, False), ("I08c sau gộp BN", lf, True)):
            lat_rows.append({"cấu hình": f"{lab} ({bcfg.exp_id})", "gpu": r["gpu"], "dtype": "fp32", "batch": 1,
                             "img_size": bcfg.img_size, "K": 1, "gộp BN": "có" if f else "không", "p50_ms": r["p50"],
                             "p95_ms": r["p95"], "p99_ms": r["p99"], "images_per_s": r["images_per_s"],
                             "torch": r["torch"], "tiền xử lý": "không tính"})
        mb, mf = metrics(by, inf.softmax_np(blg)), metrics(by, inf.softmax_np(flg))
        rows.append({"exp_id": "I08c", "phương pháp": f"Gộp BN vào conv ({fused.fused_pairs} cặp)",
                     "mô hình/checkpoint": f"{bcfg.exp_id}/seed{bcfg.seed}", "K": 1, "macro_f1_val": mf["macro_f1"],
                     "top1_val": mf["top1"], "ece_val": mf["ece"], "f1_chinee_val": mf["f1_chinee"],
                     "f1_snake_val": mf["f1_snake"], "p50_ms_b1": lf["p50"], "p95_ms_b1": lf["p95"],
                     "p99_ms_b1": lf["p99"], "throughput_img_s_b32": np.nan,
                     "ghi chú": f"trước gộp: F1 {mb['macro_f1']:.4f}, p50 {lb['p50']:.2f} ms; sai số lớn nhất "
                                f"{fused.fuse_max_abs_err:.2e}; khác model chính nên không so F1 với I00"})

    inf_df = pd.DataFrame(rows)
    base = inf_df.loc[inf_df.exp_id == "I00", "p50_ms_b1"].iloc[0]
    inf_df["chi_phí_tương_đối_vs_I00"] = inf_df["p50_ms_b1"] / base
    inf_df.loc[inf_df.exp_id == "I08c", "chi_phí_tương_đối_vs_I00"] = np.nan
    lat_df = pd.DataFrame(lat_rows)
    inf_df.to_csv(out_dir / "Inference.csv", index=False)
    lat_df.to_csv(out_dir / "Latency.csv", index=False)
    np.savez(out_dir / "val_probs.npz", y=y, **store)
    (out_dir / "temperature.json").write_text(json.dumps({"T": T, "ece_cross_fit": ece_cross}))
    return inf_df, lat_df


def plot_tradeoff(inf_df: pd.DataFrame, path) -> None:
    """Scatter macro-F1 val theo độ trễ p50 batch 1 (GUIDE mục 4.2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = inf_df.dropna(subset=["p50_ms_b1"])
    d = d[d.exp_id != "I08c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["p50_ms_b1"], d["macro_f1_val"])
    for _, r in d.iterrows():
        ax.annotate(r["exp_id"], (r["p50_ms_b1"], r["macro_f1_val"]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    ax.set(xlabel="độ trễ p50, batch 1 (ms)", ylabel="macro-F1 val", title="Đánh đổi độ chính xác - độ trễ (val)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Bước 4: chung kết
# --------------------------------------------------------------------------- #
def finalize(exp_id: str, run_root, seeds, method: str = "1view", space: str = "logit",
             temperature: bool = False, pred_dir="predictions", device: str = "cuda", force: bool = False) -> pd.DataFrame:
    """Ghi predictions/<exp_id>_seed<k>_{val,test}.csv bằng phương pháp suy luận đã chốt trên VAL.

    temperature=True: khớp T trên val (cho từng seed), ghi bản đã hiệu chuẩn vào <exp_id>_*, bản chưa
    hiệu chuẩn vào <exp_id>uncal_* (để eval.py grade chấm I4a). TEST chạy một lần mỗi seed.
    """
    pred_dir = Path(pred_dir)
    rows = []
    for s in seeds:
        test_path = pred_dir / f"{exp_id}_seed{s}_test.csv"
        if test_path.exists() and not force:
            raise FileExistsError(f"{test_path} đã có: test chỉ được chạy một lần mỗi seed (README S2)")
        cfg, model = load_run(Path(run_root) / exp_id / f"seed{s}", device)
        out = {}
        for split in ("val", "test"):
            names, y, lg, k = method_logits(model, cfg, split, method, space, device)
            out[split] = (names, y, lg)
        T = inf.fit_temperature(out["val"][2], out["val"][1]) if temperature else 1.0
        for split, (names, y, lg) in out.items():
            ev.save_predictions(pred_dir / f"{exp_id}_seed{s}_{split}.csv", names, y, inf.apply_temperature(lg, T))
            if temperature:
                ev.save_predictions(pred_dir / f"{exp_id}uncal_seed{s}_{split}.csv", names, y, inf.softmax_np(lg))
        rows.append({"exp_id": exp_id, "seed": s, "method": method, "space": space, "K": k, "T": T})
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(pred_dir / f"{exp_id}_finalize_log.csv", index=False)
    return df


def plot_confusion(confusion_csv, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = pd.read_csv(confusion_csv, index_col=0).to_numpy()
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(9), ds.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), ds.CLASS_NAMES)
    ax.set(xlabel="nhãn dự đoán", ylabel="nhãn thật", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def show_errors(pred_csv, images_dir, true_cls: int, pred_cls: int, path, n: int = 8) -> int:
    """Lưu lưới ảnh nhãn thật `true_cls` bị đoán thành `pred_cls` (phân tích lỗi). Trả về số ca."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    p = pd.read_csv(pred_csv)
    bad = p[(p.y_true == true_cls) & (p.y_pred == pred_cls)]
    if bad.empty:
        return 0
    bad = bad.assign(conf=bad[f"p{pred_cls}"]).sort_values("conf", ascending=False).head(n)
    fig, axes = plt.subplots(1, len(bad), figsize=(2.6 * len(bad), 3.2), squeeze=False)
    for ax, (_, r) in zip(axes[0], bad.iterrows()):
        ax.imshow(Image.open(Path(images_dir) / r.Filename))
        ax.set_title(f"p={r.conf:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Thật: {ds.CLASS_NAMES[true_cls]} -> đoán: {ds.CLASS_NAMES[pred_cls]}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return int(((p.y_true == true_cls) & (p.y_pred == pred_cls)).sum())


# --------------------------------------------------------------------------- #
# Bước 5: results.xlsx
# --------------------------------------------------------------------------- #
def load_summaries(run_root, prefix: str) -> pd.DataFrame:
    rows = [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in sorted(Path(run_root).glob(f"{prefix}*/seed*/summary.json"))]
    return pd.DataFrame(rows)


def _fmt_ms(mean, std):
    return f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f} (1 seed)"


def build_xlsx(path, run_root, step3_dir, eval_out, pred_dir, final_tag: str, base_tag: str, final_desc: str,
               ablation_notes: dict, backbone_notes: dict | None = None) -> None:
    """Gom mọi kết quả thành results.xlsx với 7 sheet theo GUIDE mục 6.1.

    ablation_notes: {exp_id: (trục, khác T00 ở điểm nào)}; backbone_notes: {exp_id: ghi chú}.
    """
    run_root, step3_dir, eval_out = Path(run_root), Path(step3_dir), Path(eval_out)

    b = load_summaries(run_root, "B")
    backbones = pd.DataFrame({
        "exp_id": b.exp_id, "backbone": b.backbone.str.split(".").str[0], "tag trọng số": b.weight_tag,
        "#tham số (M)": b.params_M, "GMAC": b.gmacs, "độ phân giải": b.img_size, "epoch": b.epochs, "seed": b.seed,
        "macro-F1 val": b.val_macro_f1, "top-1 val": b.val_top1, "F1 Chinee val": b.val_f1_chinee,
        "F1 Snake val": b.val_f1_snake, "thời gian train/epoch (s)": b.train_s_per_epoch,
        "độ trễ batch-1 sơ bộ (ms)": b.latency_b1_p50_ms, "best epoch": b.best_epoch,
        "ghi chú": [(backbone_notes or {}).get(e, "") for e in b.exp_id]})

    t = load_summaries(run_root, "T")
    t = t[t.seed == 0]
    base_f1 = float(t.loc[t.exp_id == "T00", "val_macro_f1"].iloc[0])
    training = pd.DataFrame({
        "exp_id": t.exp_id, "backbone": t.backbone, "trục": [ablation_notes.get(e, ("", ""))[0] for e in t.exp_id],
        "khác T00 ở điểm nào": [ablation_notes.get(e, ("", ""))[1] for e in t.exp_id], "seed": t.seed,
        "macro-F1 val": t.val_macro_f1, "top-1 val": t.val_top1, "Δ macro-F1 vs T00": t.val_macro_f1 - base_f1,
        "F1 Chinee val": t.val_f1_chinee, "F1 Snake val": t.val_f1_snake, "ECE val": t.val_ece,
        "best epoch": t.best_epoch, "ghi chú": "1 seed: chỉ kết luận khi Δ rõ ràng lớn hơn nhiễu seed"})

    inference = pd.read_csv(step3_dir / "Inference.csv")
    latency = pd.read_csv(step3_dir / "Latency.csv")

    final_rows = []
    for tag, desc in ((final_tag, final_desc), (base_tag, "Mốc: công thức nền T00 + suy luận 1 view I00")):
        ps = pd.read_csv(eval_out / f"{tag}_per_seed.csv")
        val_f1 = []
        for _, r in ps.iterrows():
            vp = Path(pred_dir) / r.file.replace("_test.csv", "_val.csv")
            val_f1.append(metrics(*_read_val(vp))["macro_f1"] if vp.exists() else np.nan)
        ps["val_macro_f1"] = val_f1
        for _, r in ps.iterrows():
            final_rows.append({"exp_id": tag, "cấu hình": desc, "seed": r.seed, "macro-F1 val": r.val_macro_f1,
                               "macro-F1 test": r.macro_f1, "top-1 test": r.top1, "balanced acc test": r.balanced_acc,
                               "ECE test": r.ece})
        s = json.loads((eval_out / f"{tag}_summary.json").read_text(encoding="utf-8"))
        vf = np.array(val_f1, dtype=float)
        final_rows.append({"exp_id": f"{tag} (mean ± std, {len(ps)} seed)", "cấu hình": desc, "seed": "tổng hợp",
                           "macro-F1 val": _fmt_ms(np.nanmean(vf), np.nanstd(vf, ddof=1) if len(vf) > 1 else np.nan),
                           "macro-F1 test": _fmt_ms(s["macro_f1"]["mean"], s["macro_f1"]["std"]),
                           "top-1 test": _fmt_ms(s["top1"]["mean"], s["top1"]["std"]),
                           "balanced acc test": _fmt_ms(s["balanced_acc"]["mean"], s["balanced_acc"]["std"]),
                           "ECE test": _fmt_ms(s["ece"]["mean"], s["ece"]["std"])})
    final = pd.DataFrame(final_rows)

    per_class = []
    for tag, label in ((final_tag, "chung kết"), (base_tag, "mốc")):
        pc = pd.read_csv(eval_out / f"{tag}_per_class.csv")
        per_class.append(pd.DataFrame({"cấu hình": f"{tag} ({label})", "lớp": pc["class"], "số ảnh test": pc.support,
                                       "precision": pc.precision_mean, "precision std": pc.precision_std,
                                       "recall": pc.recall_mean, "recall std": pc.recall_std,
                                       "F1": pc.f1_mean, "F1 std": pc.f1_std}))
    per_class = pd.concat(per_class, ignore_index=True)

    summ = pd.concat([
        backbones.assign(loại="backbone")[["loại", "exp_id", "backbone", "macro-F1 val", "top-1 val",
                                            "độ trễ batch-1 sơ bộ (ms)"]].rename(columns={"độ trễ batch-1 sơ bộ (ms)": "p50 b1 (ms)"}),
        training.assign(loại="training")[["loại", "exp_id", "backbone", "macro-F1 val", "top-1 val"]],
        inference.assign(loại="inference", backbone=inference["mô hình/checkpoint"])
        [["loại", "exp_id", "backbone", "macro_f1_val", "top1_val", "p50_ms_b1"]]
        .rename(columns={"macro_f1_val": "macro-F1 val", "top1_val": "top-1 val", "p50_ms_b1": "p50 b1 (ms)"}),
    ], ignore_index=True).sort_values("macro-F1 val", ascending=False).head(10)
    fsum = json.loads((eval_out / f"{final_tag}_summary.json").read_text(encoding="utf-8"))
    bsum = json.loads((eval_out / f"{base_tag}_summary.json").read_text(encoding="utf-8"))
    head = pd.DataFrame([
        {"loại": "CHUNG KẾT (test)", "exp_id": final_tag, "backbone": final_desc,
         "macro-F1 val": "", "top-1 val": "",
         "macro-F1 test": _fmt_ms(fsum["macro_f1"]["mean"], fsum["macro_f1"]["std"]),
         "top-1 test": _fmt_ms(fsum["top1"]["mean"], fsum["top1"]["std"])},
        {"loại": "MỐC (test)", "exp_id": base_tag, "backbone": "T00 + I00",
         "macro-F1 val": "", "top-1 val": "",
         "macro-F1 test": _fmt_ms(bsum["macro_f1"]["mean"], bsum["macro_f1"]["std"]),
         "top-1 test": _fmt_ms(bsum["top1"]["mean"], bsum["top1"]["std"])},
        {"loại": "Δ chung kết - mốc", "exp_id": "", "backbone": "",
         "macro-F1 test": f"{fsum['macro_f1']['mean'] - bsum['macro_f1']['mean']:+.4f}",
         "top-1 test": f"{fsum['top1']['mean'] - bsum['top1']['mean']:+.4f}"},
    ])
    summary = pd.concat([head, pd.DataFrame([{}]), summ.assign(**{"ghi chú": "top 10 theo macro-F1 val (1 seed)"})],
                        ignore_index=True)

    sheets = {"Summary": summary, "Backbones": backbones, "Training": training, "Inference": inference,
              "Final": final, "PerClass": per_class, "Latency": latency}
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    _style_xlsx(path, {"Backbones": "macro-F1 val", "Training": "macro-F1 val", "Inference": "macro_f1_val"})


def _read_val(path):
    p = pd.read_csv(path)
    return p.y_true.to_numpy(), p[[f"p{i}" for i in range(9)]].to_numpy()


def _style_xlsx(path, best_cols: dict) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = load_workbook(path)
    hl = PatternFill("solid", fgColor="FFF2CC")
    for ws in wb.worksheets:
        ws.freeze_panes = "B2"
        for c in ws[1]:
            c.font = Font(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(10, width + 2), 60)
            for c in col[1:]:
                if isinstance(c.value, float):
                    c.number_format = "0.0000"
        key = best_cols.get(ws.title)
        if key:
            hdr = [c.value for c in ws[1]]
            if key in hdr:
                j = hdr.index(key) + 1
                vals = [(ws.cell(i, j).value, i) for i in range(2, ws.max_row + 1)
                        if isinstance(ws.cell(i, j).value, (int, float))]
                if vals:
                    best_row = max(vals)[1]
                    for c in ws[best_row]:
                        c.fill = hl
    wb.save(path)
