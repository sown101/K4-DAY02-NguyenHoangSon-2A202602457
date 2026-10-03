"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ BN đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
from torch import nn

# Tag trọng số cụ thể (timm 1.x). Ghi đúng tag này vào results.xlsx; model.pretrained_cfg xác nhận lại.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",
}


def resolve_name(name: str) -> str:
    return SUGGESTED_BACKBONES.get(name, name)


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0) -> nn.Module:
    """Tạo model 9 lớp. init (trục A): "scratch" | "frozen" | "finetune"."""
    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    pretrained = pretrained and init != "scratch"
    kw = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    model = timm.create_model(resolve_name(name), **kw)
    model.weight_tag = model.pretrained_cfg.get("tag") if pretrained else "random-init"
    model.arch_name = resolve_name(name)
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def head_parameters(model: nn.Module) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model: nn.Module) -> None:
    """requires_grad=False cho mọi tham số trừ head; đánh dấu để set_train_mode giữ backbone ở eval."""
    head = head_parameters(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model: nn.Module) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì backbone (BN, dropout) ở eval, chỉ head ở train.

    Lý do: BN ở train mode vẫn cập nhật running_mean/var dù trọng số bị đóng băng, làm thống kê
    pretrained bị trôi theo batch DeepWeeds và kết quả lúc eval khó hiểu (GUIDE mục 3.2).
    """
    if getattr(model, "frozen_backbone", False):
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float,
                 layer_decay: float | None = None):
    """3 nhóm (slide trang 52): backbone ndim>1 (có wd) | norm+bias backbone (wd=0) | head (lr_head).

    Bias của head cũng không weight decay (ndim<=1). layer_decay (trục E, tuỳ chọn) chưa dùng.
    """
    head = head_parameters(model)
    groups = {"backbone": [], "backbone_no_wd": [], "head": [], "head_no_wd": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        in_head = id(p) in head
        no_wd = p.ndim <= 1
        key = ("head" if in_head else "backbone") + ("_no_wd" if no_wd else "")
        groups[key].append(p)
    spec = {"backbone": (lr_backbone, weight_decay), "backbone_no_wd": (lr_backbone, 0.0),
            "head": (lr_head, weight_decay), "head_no_wd": (lr_head, 0.0)}
    return [{"params": ps, "lr": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm conv, linear VÀ matmul của attention),
    MAC = FLOPs / 2. Có thể lệch vài % so với fvcore/ptflops (không đếm phép theo phần tử).
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9
