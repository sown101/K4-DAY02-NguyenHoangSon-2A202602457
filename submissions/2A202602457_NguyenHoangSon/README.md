# Lab Day 2 — DeepWeeds · Nguyễn Hoàng Sơn (2A202602457)

Kết quả chung kết (test fold 0, 3 seed): **macro-F1 0.9614 ± 0.0040**, top-1 0.9692 ± 0.0029, ECE 0.0069.
Phân tích đầy đủ trong [`report.md`](report.md), bảng số liệu trong [`results.xlsx`](results.xlsx).

## Link chạy lại

| Nền tảng | Link | Ghi chú |
|---|---|---|
| Google Colab | [Mở `code/lab_day2.ipynb` trên Colab](https://colab.research.google.com/github/sown101/K4-DAY02-NguyenHoangSon-2A202602457/blob/main/submissions/2A202602457_NguyenHoangSon/code/lab_day2.ipynb) | Runtime → T4 GPU; kết quả ghi vào Google Drive `MyDrive/K4_DAY02` |
| Kaggle | https://www.kaggle.com/code/sowncs/notebook60bb9e3d04 | Bản đã chạy dùng để nộp: Version 1 (Bước 2), Version 2 (T11/T12 + Bước 3), Version 3 (chung kết). Bật GPU T4 và Internet |

Cùng một notebook chạy được trên cả hai nền tảng (tự nhận biết môi trường). Checkpoint (`best.pt`) không commit
vào git; chúng nằm trong output của Kaggle Version 3.

## Môi trường

GPU NVIDIA Tesla T4 · Python 3.13.15 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · timm 1.0.29 · numpy, pandas,
matplotlib, openpyxl (bản có sẵn của Colab/Kaggle). `eval.py` của repo gốc, không sửa.

Bộ trọng số `timm` đã dùng: `resnet50.a1_in1k`, `resnet50.tv2_in1k`, `convnext_tiny.fb_in1k`,
`deit_small_patch16_224.fb_in1k`, `efficientnet_b0.ra_in1k`, `mobilenetv3_large_100.ra_in1k`.

## Thứ tự chạy

Notebook `code/lab_day2.ipynb`, biến `STAGE` ở ô đầu tiên quyết định chạy đến bước nào (các ô của bước sau được
bỏ qua). Mọi thí nghiệm đi qua một hàm `train.run(Config(...))`. Chạy lại notebook không làm hỏng kết quả: thí nghiệm
đã có `summary.json` được bỏ qua, thí nghiệm dở dang được resume từ checkpoint epoch cuối, và `finalize` bỏ qua
seed đã có file test (không tính lại, không ghi đè — test chỉ chạy một lần mỗi seed). Các ô Bước 0, Bước 4 (score,
grade, đo độ trễ) và Bước 5 thì luôn chạy lại và ghi đè kết quả của chúng.

1. **Cài đặt + dữ liệu:** clone repo, tải `images.zip` từ Zenodo (kiểm tra MD5), tải CSV fold 0 từ GitHub của tác giả.
2. **Bước 0:** kiểm tra chia dữ liệu, EDA, unit test (`tests_code.py`), loss ban đầu, overfit 1 batch, ảnh sau augmentation.
3. **Bước 1** (`STAGE ≥ 2`): B01–B05 + B06.
4. **Bước 2:** T00–T10 (một yếu tố mỗi lần), tổ hợp T11, T12.
5. **Bước 3** (`STAGE ≥ 3`): `experiments.run_inference_suite` trên val → `step3/`.
6. **Bước 4** (`STAGE ≥ 4`): F01 và T00 × 3 seed; `experiments.finalize` ghi `predictions/` (test **một lần** mỗi seed);
   `eval.py score` và `eval.py grade`.
7. **Bước 5** (`STAGE = 5`): ma trận nhầm lẫn, ảnh lỗi, `results.xlsx`.

Chạy kiểm tra không cần GPU: `cd code && python -m unittest tests_code -v`.

Kiểm tra lại chỉ số từ file dự đoán (chạy từ thư mục gốc repo, `data/labels/` là CSV fold 0 của tác giả):

```
python eval.py grade --final "submissions/2A202602457_NguyenHoangSon/predictions/F01_seed*_test.csv" \
  --baseline "submissions/2A202602457_NguyenHoangSon/predictions/T00_seed*_test.csv" \
  --uncal "submissions/2A202602457_NguyenHoangSon/predictions/F01uncal_seed*_test.csv" \
  --final-val "submissions/2A202602457_NguyenHoangSon/predictions/F01_seed*_val.csv" \
  --latency-p95-ms 26.85 --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv \
  --labels data/labels/labels.csv
```

## Seed

Sàng lọc (B01–B06, T00–T12): seed 0. Chung kết F01 và mốc T00: seed 0, 1, 2. `set_seed` cố định `random`, numpy,
torch (CPU, CUDA) và seed từng DataLoader worker; `cudnn.benchmark=True` nên không tái lập từng bit (lệch đo được
~0.0007 macro-F1 giữa hai lần chạy cùng cấu hình B03/T00).

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `code/` | `dataset.py`, `model.py`, `losses.py`, `train.py`, `inference.py`, `benchmark.py` (bộ khung starter đã hoàn thiện), `experiments.py` (Bước 3–5), `tests_code.py` (20 test), `lab_day2.ipynb` |
| `results.xlsx` | 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency |
| `report.md` | Báo cáo kết luận |
| `curves/` | Đường cong training của mọi thí nghiệm B, T, F (loss train/val, macro-F1/top-1 val, LR theo bước) |
| `predictions/` | Dự đoán test/val của chung kết `F01`, bản chưa hiệu chuẩn `F01uncal`, mốc `T00` (3 seed) và dự đoán val của mọi thí nghiệm sàng lọc |
| `eval_out/` | Kết quả `eval.py score/grade` (`*_summary.json`, `grade_I.json`), độ trễ chung kết `F01_latency_b1.json`, ma trận nhầm lẫn, ảnh lỗi |
| `eda/` | Kiểm tra chia dữ liệu (`split_report.json`), phân bố lớp, ảnh mẫu, ảnh sau augmentation |
| `step3/` | Bảng suy luận và độ trễ, biểu đồ đánh đổi, nhiệt độ T |
| `logs/<exp_id>/seed<k>/` | `config.json` (mọi siêu tham số + phiên bản thư viện), `history.csv` (log theo epoch), `summary.json` |
