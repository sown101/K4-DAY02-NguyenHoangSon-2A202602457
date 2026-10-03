"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn tiền xử lý (ghi vào báo cáo):
  - Train "basic": RandomResizedCrop(img_size, scale=(0.25, 1)) + lật ngang. Không dùng scale mặc định
    0.08 vì ảnh cỏ dại 256x256 đã là ảnh cận cảnh; crop quá nhỏ dễ cắt mất đặc trưng loài với 12 epoch.
  - Val/test: Resize(round(img_size / crop_pct)) + CenterCrop(img_size), crop_pct = 0.875
    (img_size 224 -> resize 256 = giữ nguyên ảnh gốc rồi cắt giữa 224).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
DEFAULT_CROP_PCT = 0.875


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv nguyên bản (S1)."""
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        df["Label"] = df["Label"].astype(int)
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Lỗi thì raise AssertionError."""
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    ratio = {k: n[k] / total for k in n}

    per_class = pd.DataFrame({k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for k, v in splits.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(1)

    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert v["Filename"].is_unique, f"{k}: có Filename trùng trong cùng một tập"
    overlap = {"train∩val": len(names["train"] & names["val"]),
               "train∩test": len(names["train"] & names["test"]),
               "val∩test": len(names["val"] & names["test"])}
    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    union = len(names["train"] | names["val"] | names["test"])
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"

    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.iterdir()} if images_dir.is_dir() else set()
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}"

    for k, (lo, hi) in {"train": (0.59, 0.61), "val": (0.19, 0.21), "test": (0.19, 0.21)}.items():
        if not lo <= ratio[k] <= hi:
            print(f"CẢNH BÁO: tỉ lệ {k} = {ratio[k]:.3f} lệch khỏi 60/20/20 hơn 1 điểm %, báo giảng viên")

    report = {"n": n, "ratio": {k: round(v, 4) for k, v in ratio.items()}, "union": union,
              "overlap": overlap, "missing_files": len(missing),
              "per_class": per_class.to_dict(orient="index")}
    if verbose:
        print("Số ảnh:", n, "| tỉ lệ:", report["ratio"], "| hợp:", union)
        print("Giao theo Filename:", overlap, "| file thiếu trên đĩa:", len(missing))
        print(per_class.to_string())
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = DEFAULT_CROP_PCT):
    """Transform cho train (theo `aug`) hoặc val/test (không ngẫu nhiên).

    aug (trục B): "basic" (crop + lật ngang) | "color" (+ ColorJitter) | "trivial" (+ TrivialAugmentWide)
                  | "randaug" (+ RandAugment(2, 9)) | "flipv" (+ lật dọc: ảnh chụp từ trên xuống nên
                  hướng không mang nghĩa, đây là thí nghiệm kiểm chứng giả thuyết đó).
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    """
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = int(round(img_size / crop_pct))
        return T.Compose([T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                          T.CenterCrop(img_size), *normalize])

    ops = [T.RandomResizedCrop(img_size, scale=(0.25, 1.0), interpolation=T.InterpolationMode.BICUBIC),
           T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "flipv":
        ops.append(T.RandomVerticalFlip())
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*ops, *normalize])


class DeepWeedsDataset(Dataset):
    """Đọc ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, nhãn int, tên file).

    cache=True nạp trước BYTES JPEG vào RAM (~490 MB cho cả dataset) để tránh nghẽn đọc đĩa;
    vẫn giải mã mỗi lần nên augmentation ngẫu nhiên giữ nguyên.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self._bytes = None
        if cache:
            self._bytes = [(self.images_dir / f).read_bytes() for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def load_image(self, i: int) -> Image.Image:
        if self._bytes is not None:
            import io
            return Image.open(io.BytesIO(self._bytes[i])).convert("RGB")
        with Image.open(self.images_dir / self.filenames[i]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random trong mỗi worker từ seed torch của worker đó (tái lập augmentation)."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def balanced_weights(labels) -> torch.Tensor:
    """Trọng số mỗi mẫu = 1 / (số ảnh của lớp) để WeightedRandomSampler bốc đều các lớp (trục D)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    return torch.as_tensor(1.0 / counts[labels], dtype=torch.double)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: bool = False, dataset: Dataset | None = None):
    """DataLoader. train=False giữ nguyên thứ tự df (để ghép logit với Filename)."""
    ds = dataset if dataset is not None else DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, shuffle=False, drop_last=False, **kw)
    if sampler == "balanced":
        w = balanced_weights(ds.labels)
        smp = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, sampler=smp, drop_last=True, **kw)
    if sampler is not None:
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, shuffle=True, drop_last=True, **kw)


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Giải chuẩn hoá (C,H,W) hoặc (N,C,H,W) về [0,1] để vẽ ảnh sau augmentation."""
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)
