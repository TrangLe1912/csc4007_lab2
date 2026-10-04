# CSC4007 — Lab 2 Analysis Report
## IMDB Core Lab + VietNewsSense Vietnamese Transfer Challenge

## 1. Thông tin sinh viên

- Họ và tên:
- Mã sinh viên:
- Lớp:
- Repo GitHub:

---

## 2. Part A — IMDB Data Audit

- Số lượng mẫu đã dùng:
- Phân bố nhãn positive / negative:
- Độ dài review điển hình (median, p95):
- Có missing / empty text không?
- Có duplicate không?

### Ba quan sát đáng chú ý

1.
2.
3.

---

## 3. Part A — IMDB Preprocessing Design

- Bạn đã dùng những bước làm sạch nào?
- Bạn giữ lại dấu câu hay bỏ đi? Vì sao?
- Bạn có thay số bằng `<NUM>` không? Vì sao?
- Có bước nào bạn cố tình **không làm** để tránh mất tín hiệu cảm xúc?

---

## 4. Part A — IMDB Experiment Comparison

| Run | Text version | Vectorizer | Analyzer | Model | n-gram | Macro-F1 | Accuracy | Ghi chú |
|---|---|---|---|---|---|---:|---:|---|
| I0 |  |  |  |  |  |  |  |  |
| I1 |  |  |  |  |  |  |  |  |

### Nhận xét

- Pipeline nào tốt hơn?
- Chênh lệch có lớn không?
- Thay đổi nào có khả năng tạo ra khác biệt?

---

## 5. Part A — IMDB Error Analysis (>= 10 lỗi)

Chọn ít nhất 10 mẫu từ `error_analysis.csv`.

Gợi ý:

- phủ định / tương phản;
- mixed sentiment;
- sarcasm / irony;
- review rất dài;
- entity/domain-specific wording;
- confident but wrong.

| ID | True | Pred | Nhóm lỗi | Giải thích | Hướng cải thiện |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

---

# PART B — VIETNAMESE TRANSFER CHALLENGE

## 6. VietNewsSense Data Check

- Số lượng mẫu:
- Số lớp:
- Phân bố các lớp:
- Độ dài văn bản:
- Metadata có sẵn:
- Vấn đề dữ liệu đáng chú ý:

### Câu hỏi

Những điểm nào của VietNewsSense khác IMDB và có thể ảnh hưởng preprocessing/representation?

---

## 7. VietNewsSense Experiment Comparison

Giữ cùng seed/split/model khi so sánh V0 và V1.

| Run | Preprocessing | Representation | Analyzer | Model | Macro-F1 | Accuracy |
|---|---|---|---|---|---:|---:|
| V0 | Light clean, không segmentation | Word TF-IDF | word | Linear SVM |  |  |
| V1 | + Vietnamese word segmentation | Word TF-IDF | word | Linear SVM |  |  |
| V2* | Light clean | Character TF-IDF | char_wb | Linear SVM |  |  |

`V2*` là bonus.

### Delta chính cần báo cáo

- Δ Macro-F1 V1 − V0:
- Δ Accuracy V1 − V0:

### Nhận xét

- Word segmentation làm kết quả tăng, giảm hay gần như không đổi?
- Kết quả có đồng đều trên mọi lớp không?
- Confusion matrix cho thấy lớp nào thường bị nhầm với lớp nào?

---

## 8. VietNewsSense Error Analysis (>= 5 lỗi)

Gợi ý nhóm lỗi:

- multi-topic article;
- ambiguous / overlapping category;
- named-entity dominated;
- short / low-information text;
- cross-category vocabulary;
- confident but wrong.

| ID | True | Pred | Title/Source (nếu có) | Nhóm lỗi | Giải thích |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

---

## 9. Vietnamese Transfer Reflection

Trả lời ngắn gọn, dựa trên kết quả thực nghiệm:

### 9.1 Thành phần nào của pipeline IMDB có thể giữ nguyên?

Ví dụ: split protocol, fit vectorizer chỉ trên train, classifier, metric...

**Trả lời:**

### 9.2 Thành phần nào cần xem xét lại khi chuyển sang tiếng Việt?

Tập trung vào normalization, tokenization/word segmentation và representation.

**Trả lời:**

### 9.3 Word segmentation ảnh hưởng kết quả như thế nào?

Không trả lời bằng trực giác; dẫn Macro-F1/Accuracy và ít nhất một bằng chứng từ error analysis.

**Trả lời:**

### 9.4 Nếu character TF-IDF tốt hơn word TF-IDF, ta có thể và không thể kết luận gì?

**Có thể kết luận:**

**Không thể kết luận:**

---

## 10. Kết luận Lab 2

Viết 5–7 dòng:

- Điều quan trọng nhất học được về preprocessing.
- Điều quan trọng nhất học được về representation.
- Vì sao một pipeline chạy tốt trên tiếng Anh chưa chắc chuyển nguyên xi sang tiếng Việt.
- Một câu hỏi bạn muốn tiếp tục kiểm chứng ở Lab 3 (RNN).
