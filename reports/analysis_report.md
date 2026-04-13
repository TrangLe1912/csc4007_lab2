# Lab 2 Analysis Report (IMDB)

## 1. Data audit
- Số lượng mẫu đã dùng:
- Phân bố nhãn positive / negative:
- Độ dài review điển hình (median, p95):
- Có missing / empty text không?
- Có duplicate không?
- 3 quan sát đáng chú ý về dữ liệu IMDB:

## 2. Preprocessing design
- Bạn đã dùng những bước làm sạch nào?
- Bạn giữ lại dấu câu hay bỏ đi? Vì sao?
- Bạn có thay số bằng `<NUM>` không? Vì sao?
- Có bước nào bạn cố tình **không** làm để tránh mất tín hiệu cảm xúc?

## 3. Experiment comparison
| Run | Text version | Vectorizer | Model | ngram | Macro-F1 | Accuracy | Ghi chú |
|---|---|---|---|---|---:|---:|---|
| 1 | raw / cleaned |  |  |  |  |  |  |
| 2 | raw / cleaned |  |  |  |  |  |  |
| 3 | raw / cleaned |  |  |  |  |  |  |

## 4. Error analysis (>= 10 lỗi)
- Chọn ít nhất 10 mẫu trong `outputs/error_analysis/error_analysis.csv`.
- Gom thành 2–4 nhóm lỗi.
- Gợi ý nhóm lỗi trên IMDB:
  - phủ định / tương phản (`not good`, `although ... still`)
  - cảm xúc trộn lẫn
  - mỉa mai / châm biếm
  - review quá dài, nhiều chi tiết không liên quan
  - mô hình dự đoán rất tự tin nhưng sai

### Bảng ghi lỗi
| ID | True | Pred | Nhóm lỗi | Giải thích ngắn |
|---|---|---|---|---|
|  |  |  |  |  |

## 5. Reflection
- Pipeline nào tốt nhất trên IMDB? Vì sao?
- Trên IMDB, accuracy và macro-F1 có chênh nhau nhiều không?
- Nếu chuyển sang dữ liệu lệch lớp hơn, bạn kỳ vọng metric nào sẽ phản ánh tốt hơn? Vì sao?
- Một cải tiến bạn muốn thử ở Lab 3 là gì?
