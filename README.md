# CSC4007 — Lab 2: Preprocessing, Vectorization & Baseline ML
## IMDB Core Lab + VietNewsSense Vietnamese Transfer Challenge

Lab 2 nối tiếp Lab 1 theo mạch:

`audit → preprocess → vectorize → baseline ML → evaluate → error analysis`

Bài lab có **hai phần liên kết**:

- **Part A — Core Lab (IMDB):** học pipeline cổ điển trên review tiếng Anh.
- **Part B — Vietnamese Transfer Challenge (VietNewsSense):** kiểm chứng xem preprocessing và representation có thể chuyển nguyên xi sang tin tức tiếng Việt hay không.

IMDB vẫn là case study chính để giữ mạch liên tục sang Lab 3 (RNN). VietNewsSense không thay thế IMDB mà đóng vai trò transfer challenge.

## Mục tiêu

Sau Lab 2, sinh viên cần:

1. Audit dữ liệu trước/sau preprocessing.
2. So sánh BoW và TF-IDF.
3. Huấn luyện Logistic Regression hoặc Linear SVM.
4. Đánh giá bằng Accuracy, Macro-F1 và Confusion Matrix.
5. Phân tích lỗi thay vì chỉ nhìn metric tổng.
6. Kiểm chứng ảnh hưởng của **Vietnamese word segmentation** và **word/character TF-IDF** trên VietNewsSense.
7. Phân biệt được phần nào của pipeline có thể giữ nguyên và phần nào phụ thuộc ngôn ngữ/dataset.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# Part A — Core Lab: IMDB

Chạy baseline:

```bash
python run_lab2.py \
  --dataset imdb \
  --seed 42 \
  --vectorizer tfidf \
  --model logreg \
  --analyzer word \
  --ngram_min 1 \
  --ngram_max 2 \
  --output_dir outputs/imdb_baseline
```

Sinh viên cần thực hiện tối thiểu **2 cấu hình IMDB**, ví dụ:

- BoW vs TF-IDF;
- Logistic Regression vs Linear SVM;
- unigram vs unigram + bigram;
- giữ dấu câu vs bỏ dấu câu.

Phân tích tối thiểu **10 mẫu sai**.

---

# Part B — Vietnamese Transfer Challenge: VietNewsSense

Dataset tiếng Việt được nạp qua `local_csv`.

Giả sử file VietNewsSense có:

- cột nội dung: `content`
- cột nhãn: `category`

Các metadata khác như `title`, `source`, `url`, `publish_date` sẽ được **giữ lại** trong output để hỗ trợ error analysis.

## V0 — Word TF-IDF, chưa word segmentation

```bash
python run_lab2.py \
  --dataset local_csv \
  --data_path data/raw/vietnewssense.csv \
  --text_col content \
  --label_col category \
  --language vi \
  --vi_segment none \
  --vectorizer tfidf \
  --model linearsvm \
  --analyzer word \
  --ngram_min 1 \
  --ngram_max 2 \
  --seed 42 \
  --output_dir outputs/vietnews_v0
```

## V1 — Word TF-IDF + Vietnamese word segmentation

```bash
python run_lab2.py \
  --dataset local_csv \
  --data_path data/raw/vietnewssense.csv \
  --text_col content \
  --label_col category \
  --language vi \
  --vi_segment underthesea \
  --vectorizer tfidf \
  --model linearsvm \
  --analyzer word \
  --ngram_min 1 \
  --ngram_max 2 \
  --seed 42 \
  --output_dir outputs/vietnews_v1_segmented
```

## V2 — BONUS: Character TF-IDF

```bash
python run_lab2.py \
  --dataset local_csv \
  --data_path data/raw/vietnewssense.csv \
  --text_col content \
  --label_col category \
  --language vi \
  --vi_segment none \
  --vectorizer tfidf \
  --model linearsvm \
  --analyzer char_wb \
  --ngram_min 3 \
  --ngram_max 5 \
  --seed 42 \
  --output_dir outputs/vietnews_v2_char
```

### Nguyên tắc so sánh V0 ↔ V1

Để đánh giá riêng tác động của word segmentation:

- giữ cùng dataset;
- giữ cùng `seed`;
- giữ cùng model;
- giữ cùng vectorizer;
- giữ cùng n-gram;
- chỉ thay `--vi_segment`.

**Không so sánh trực tiếp IMDB F1 với VietNewsSense F1 để kết luận “tiếng Việt khó hơn”.** Hai dataset có task, số lớp và phân phối khác nhau.

---

# Các tham số mới quan trọng

| Tham số | Ý nghĩa |
|---|---|
| `--language en|vi` | Chọn ngữ cảnh error analysis |
| `--vi_segment none|underthesea` | Bật/tắt Vietnamese word segmentation |
| `--analyzer word|char|char_wb` | Chọn word-level hoặc character-level features |
| `--ngram_min` / `--ngram_max` | Khoảng n-gram |
| `--output_dir` | Tách output từng run, tránh ghi đè |

Unicode được chuẩn hóa về **NFC**. Pipeline không xóa dấu tiếng Việt.

---

# Output của mỗi run

Ví dụ với `--output_dir outputs/vietnews_v1_segmented`:

```text
outputs/vietnews_v1_segmented/
├── logs/
│   ├── data_audit.md
│   ├── audit_before.md
│   ├── audit_after.md
│   └── run_summary.json
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── metrics/
│   ├── metrics_summary.json
│   └── metrics_summary.md
├── figures/
│   └── confusion_matrix.png
├── error_analysis/
│   ├── error_analysis.csv
│   └── error_analysis_summary.md
├── pipeline/
│   └── model_pipeline.joblib
└── predictions/
    └── test_predictions.csv
```

---

# Yêu cầu bắt buộc

Sinh viên cần:

1. Chạy **ít nhất 2 cấu hình IMDB**.
2. Chạy VietNewsSense **V0 và V1**.
3. Phân tích **ít nhất 10 lỗi IMDB**.
4. Phân tích **ít nhất 5 lỗi VietNewsSense**.
5. Hoàn thành `reports/analysis_report.md`.
6. Giải thích kết quả transfer bằng bằng chứng metric/error analysis, không kết luận chỉ từ trực giác.

**V2 character TF-IDF là bonus.**

---

# Cấu trúc repo

```text
csc4007_lab2/
├── .github/workflows/ci.yml
├── data/raw/
│   ├── README.md
│   └── sample_vietnews_tiny.csv
├── notebooks/
├── reports/
│   └── analysis_report.md
├── requirements.txt
├── run_lab2.py
└── src/
    ├── audit_core.py
    ├── error_analysis.py
    ├── evaluate.py
    ├── load_data.py
    ├── modeling.py
    ├── preprocess.py
    ├── split.py
    └── utils.py
```

## CI

CI gồm:

- smoke test IMDB;
- smoke test local Vietnamese CSV;
- kiểm tra artefact bắt buộc.

File `sample_vietnews_tiny.csv` chỉ dùng để kiểm tra code, **không dùng để báo cáo kết quả học thuật**.
