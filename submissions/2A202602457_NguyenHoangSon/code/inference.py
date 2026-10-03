"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, [logits_view_k])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _forward_views(model, loader, device, views, amp: bool = True):
    model.eval()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    names, ys, outs = [], [], [[] for _ in views]
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            for k, view in enumerate(views):
                xv = view(x) if view is not None else x
                with torch.autocast(dev_type, dtype=torch.float16, enabled=amp and dev_type == "cuda"):
                    outs[k].append(model(xv).float().cpu())
            ys.append(torch.as_tensor(y))
            names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Logit theo đúng thứ tự loader. `view` biến đổi batch ảnh (ví dụ view_hflip) hoặc None."""
    names, y, outs = _forward_views(model, loader, device, [view], amp)
    return names, y, outs[0]


def predict_views(model, loader, device, views, amp: bool = True):
    """Chạy K view trong một lượt đọc dữ liệu. Mỗi view là hàm x -> x' (cùng batch, cùng thứ tự)."""
    return _forward_views(model, loader, device, list(views), amp)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch x; flip=True thêm bản lật của mỗi crop (10 view)."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError("crop lớn hơn ảnh")
    tops = [0, 0, h - crop, h - crop, (h - crop) // 2]
    lefts = [0, w - crop, 0, w - crop, (w - crop) // 2]
    out = [x[..., t:t + crop, l:l + crop] for t, l in zip(tops, lefts)]
    if flip:
        out += [view_hflip(v) for v in out]
    return out


def multicrop_view_fns(crop: int, flip: bool = False):
    """Danh sách hàm view tương ứng views_multicrop (dùng với predict_views)."""
    n = 10 if flip else 5
    return [(lambda x, i=i: views_multicrop(x, crop, flip)[i]) for i in range(n)]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bicubic). CNN có global pooling nhận mọi kích thước;
    ViT/DeiT cần nội suy position embedding (timm: dynamic_img_size=True), Swin cần kích thước chia hết
    cho cửa sổ: ở bài này chỉ áp dụng multi-scale cho CNN."""
    return [F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False) for s in sizes]


def multiscale_view_fns(sizes):
    return [(lambda x, s=s: views_multiscale(x, [s])[0]) for s in sizes]


def softmax_np(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob") -> np.ndarray:
    """Gộp K view: "prob" = trung bình softmax; "logit" = softmax của trung bình logit (I03)."""
    stack = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        return np.stack([softmax_np(l) for l in stack]).mean(0)
    if space == "logit":
        return softmax_np(stack.mean(0))
    raise ValueError(f"space không hợp lệ: {space}")


def ensemble_probs(list_of_probs) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    shapes = {np.shape(p) for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình khác kích thước đầu ra: {shapes}")
    return np.mean(np.stack(list_of_probs), axis=0)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """T > 0 cực tiểu NLL của softmax(logit / T) trên VAL (slide trang 69).

    Tối ưu log T bằng LBFGS (đảm bảo T > 0), khởi tạo từ điểm tốt nhất của lưới thô 0.05..20.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 60))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float) -> np.ndarray:
    """softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def probs_to_logits(probs) -> np.ndarray:
    """log(probs): dùng khi cần khớp T cho xác suất đã gộp (TTA/ensemble) - softmax(log p / T)."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None))


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(conv.weight.device)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused.bias.copy_(bn.bias + (b - bn.running_mean) * scale)
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (EfficientNet, MobileNetV3) có kèm drop + act:
    phải giữ lại activation, nếu không model sai."""
    act = getattr(bn, "act", None)
    drop = getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop if drop is not None else nn.Identity(), act if act is not None else nn.Identity())


def _fuse_children(module: nn.Module) -> int:
    n = 0
    names = list(module._modules.keys())
    for a, b in zip(names, names[1:]):
        conv, bn = module._modules[a], module._modules[b]
        if (type(conv) is nn.Conv2d and isinstance(bn, nn.BatchNorm2d) and bn.track_running_stats
                and bn.affine and conv.out_channels == bn.num_features):
            module._modules[a] = _fuse(conv, bn)
            module._modules[b] = _bn_replacement(bn)
            n += 1
    for child in module.children():
        n += _fuse_children(child)
    return n


@torch.no_grad()
def fuse_conv_bn(model: nn.Module, check_input: torch.Tensor | None = None, inplace: bool = False):
    """Gộp mọi cặp (Conv2d, BatchNorm2d) LIỀN KỀ trong cùng module cha (slide trang 71, 75).

    Trả về model đã gộp (bản sao nếu inplace=False). Thuộc tính `fused_pairs` = số cặp đã gộp;
    `fuse_max_abs_err` = sai số lớn nhất của đầu ra so với trước khi gộp (nếu có check_input).
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm): fused_pairs = 0, không áp dụng.
    """
    model.eval()
    ref = model(check_input) if check_input is not None else None
    fused = model if inplace else copy.deepcopy(model)
    fused.eval()
    fused.fused_pairs = _fuse_children(fused)
    if ref is not None:
        fused.fuse_max_abs_err = float((fused(check_input) - ref).abs().max())
    return fused
