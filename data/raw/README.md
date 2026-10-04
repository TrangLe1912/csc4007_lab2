# data/raw

- `sample_vietnews_tiny.csv` chỉ là dữ liệu nhỏ tổng hợp để smoke test code/CI.
- Không dùng file này để báo cáo kết quả học thuật.
- Với VietNewsSense thật, đặt CSV tại `data/raw/vietnewssense.csv` hoặc dùng đường dẫn khác qua `--data_path`.
- Ví dụ schema:
  - `content`: nội dung bài báo
  - `category`: nhãn chủ đề
  - metadata tùy chọn: `title`, `source`, `url`, `publish_date`
