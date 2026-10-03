"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy trên CPU, không cần dataset:
    python -m unittest tests_code -v          (trong thư mục code/)
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark  # noqa: E402
import inference as inf  # noqa: E402
import losses  # noqa: E402
import model as mdl  # noqa: E402
import train  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(32, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (32,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.logits, self.y)
        self.assertLess(abs(fl.item() - F.cross_entropy(self.logits, self.y).item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(losses.FocalLoss(2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_focal_alpha_matches_weighted_formula(self):
        a = torch.linspace(0.5, 1.5, 9)
        fl = losses.FocalLoss(0.0, alpha=a)(self.logits, self.y)
        ref = (F.cross_entropy(self.logits, self.y, reduction="none") * a[self.y]).mean()
        self.assertLess(abs(fl.item() - ref.item()), 1e-6)

    def test_label_smoothing(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        ours = losses.LabelSmoothingCE(0.1)(self.logits, self.y).item()
        torch_ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item()
        self.assertLess(abs(ours - torch_ref), 1e-6)

    def test_class_weights(self):
        counts = [675, 638, 618, 613, 637, 605, 644, 609, 5464]
        w = losses.class_weights(counts)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())  # lớp nhiều ảnh có trọng số nhỏ
        wb = losses.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)
        self.assertLess(wb[8].item(), wb[0].item())

    def test_cutmix_lambda_is_true_area(self):
        rng = np.random.default_rng(0)
        x = torch.zeros(8, 3, 32, 32)
        x2 = torch.ones(8, 3, 32, 32)
        for _ in range(50):
            xa = torch.cat([x[:4], x2[:4]])
            y = torch.arange(8)
            xm, (ya, yb, lam) = losses.mix_batch(xa, y, 1.0, "cutmix", rng)
            perm_changed = (ya != yb)
            # với mỗi ảnh, tỉ lệ pixel KHÔNG bị thay = lam (khi ảnh nguồn khác ảnh đích)
            for i in range(8):
                if xa[i].mean() != xa[yb[i]].mean():
                    kept = (xm[i] == xa[i]).float().mean().item()
                    self.assertAlmostEqual(kept, lam, places=6)
            self.assertTrue(0.0 <= lam <= 1.0)
            self.assertTrue(torch.equal(ya, y))
            _ = perm_changed

    def test_mixup_and_mixed_loss(self):
        rng = np.random.default_rng(1)
        x = torch.randn(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 3])
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 0.4, "mixup", rng)
        perm = [int((y == v).nonzero()) for v in yb]
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[perm], atol=1e-6))
        ce = nn.CrossEntropyLoss()
        logits = torch.randn(4, 9)
        ml = losses.mixed_loss(ce, logits, (ya, yb, lam))
        self.assertAlmostEqual(ml.item(), (lam * ce(logits, ya) + (1 - lam) * ce(logits, yb)).item(), places=6)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.down = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 9)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        return self.fc(self.pool(self.down(x)).flatten(1))

    def get_classifier(self):
        return self.fc


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        torch.manual_seed(0)
        net = TinyNet()
        for m in net.modules():  # thống kê BN không tầm thường
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-1, 1)
                m.running_var.uniform_(0.5, 2)
                m.weight.data.uniform_(0.5, 1.5)
                m.bias.data.uniform_(-0.5, 0.5)
        x = torch.randn(4, 3, 16, 16)
        fused = inf.fuse_conv_bn(net, check_input=x)
        self.assertEqual(fused.fused_pairs, 2)
        self.assertLess(fused.fuse_max_abs_err, 1e-5)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in fused.modules()))
        self.assertTrue(any(isinstance(m, nn.BatchNorm2d) for m in net.modules()))  # bản gốc không đổi

    def test_fuse_timm_bnact_keeps_activation(self):
        net = mdl.build_model("mobilenetv3_large_100", pretrained=False)
        net.eval()
        x = torch.randn(2, 3, 64, 64)
        fused = inf.fuse_conv_bn(net, check_input=x)
        self.assertGreater(fused.fused_pairs, 10)
        self.assertLess(fused.fuse_max_abs_err, 1e-3)

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(4000, 9)) * 3
        true_T = 2.5
        p = inf.softmax_np(z / true_T)
        y = np.array([rng.choice(9, p=row) for row in p])
        T = inf.fit_temperature(z, y)
        self.assertLess(abs(T - true_T), 0.2)
        np.testing.assert_array_equal(inf.apply_temperature(z, T).argmax(1), z.argmax(1))

    def test_aggregate_views(self):
        a, b = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            p = inf.aggregate_views([a, b], space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(inf.aggregate_views([a], "prob"), inf.softmax_np(a))
        np.testing.assert_allclose(inf.ensemble_probs([inf.softmax_np(a)] * 3), inf.softmax_np(a))

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        self.assertTrue(torch.equal(inf.view_hflip(inf.view_hflip(x)), x))
        self.assertEqual(inf.view_hflip(x)[0, 0, 0, 0], x[0, 0, 0, -1])
        crops = inf.views_multicrop(x, 6, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertEqual(crops[0].shape, (2, 3, 6, 6))
        self.assertEqual(inf.views_multiscale(x, [4, 12])[1].shape, (2, 3, 12, 12))


class TestModelAndTrain(unittest.TestCase):
    def test_param_groups_no_wd_on_norm_bias(self):
        net = TinyNet()
        groups = {g["name"]: g for g in mdl.param_groups(net, 1e-4, 1e-3, 0.05)}
        self.assertEqual(groups["backbone_no_wd"]["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_wd"]["params"]))
        self.assertEqual(groups["head"]["lr"], 1e-3)
        self.assertEqual(groups["head_no_wd"]["weight_decay"], 0.0)
        n = sum(p.numel() for g in groups.values() for p in g["params"])
        self.assertEqual(n, sum(p.numel() for p in net.parameters()))

    def test_freeze_keeps_bn_eval(self):
        net = TinyNet()
        mdl.freeze_backbone(net)
        mdl.set_train_mode(net)
        self.assertFalse(net.bn1.training)
        self.assertTrue(net.fc.training)
        self.assertEqual([p.requires_grad for p in net.fc.parameters()], [True, True])
        self.assertFalse(net.conv1.weight.requires_grad)

    def test_lr_schedule_shape(self):
        total, warm = 100, 10
        f = [train.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[warm - 1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(f[:warm], f[1:warm])))   # tăng khi warmup
        self.assertTrue(all(a >= b for a, b in zip(f[warm:], f[warm + 1:])))  # giảm khi cosine
        self.assertLess(f[-1], 0.01)

    def test_ema(self):
        net = nn.Linear(2, 2)
        ema = train.EMA(net, 0.9)
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        d = min(0.9, 2 / 11)
        self.assertFalse(torch.equal(ema.module.weight, net.weight))
        self.assertTrue(torch.allclose(ema.module.weight, net.weight - d * 1.0, atol=1e-6))

    def test_parse_overrides(self):
        o = train.parse_overrides(["seed=2", "ema_decay=none", "sampler=balanced", "amp=false",
                                   "class_weight_beta=0.999", "lr_head=1e-4", "mix=none"])
        self.assertEqual(o, {"seed": 2, "ema_decay": None, "sampler": "balanced", "amp": False,
                             "class_weight_beta": 0.999, "lr_head": 1e-4, "mix": None})
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co=1"])

    def test_initial_loss_near_ln9(self):
        torch.manual_seed(0)
        net = mdl.build_model("resnet18", pretrained=False)
        net.eval()
        with torch.no_grad():
            loss = F.cross_entropy(net(torch.randn(16, 3, 64, 64)), torch.randint(0, 9, (16,)))
        self.assertLess(abs(loss.item() - math.log(9)), 0.5)

    def test_gmacs_resnet50(self):
        net = mdl.build_model("resnet50", pretrained=False)
        self.assertAlmostEqual(mdl.count_params(net), 23.5, delta=0.3)   # head 9 lớp thay vì 1000
        self.assertAlmostEqual(mdl.count_gmacs(net, 224), 4.1, delta=0.15)


class TestBenchmark(unittest.TestCase):
    def test_bench_and_report(self):
        r = benchmark.bench(lambda: sum(range(1000)), warmup=10, iters=50)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        rep = benchmark.latency_report(TinyNet(), 1, 32, "fp32", "cpu", iters=50)
        for k in ("gpu", "dtype", "batch", "p50", "p95", "p99", "images_per_s", "torch"):
            self.assertIn(k, rep)
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, iters=10)


if __name__ == "__main__":
    unittest.main()
