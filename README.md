# Hệ thống RAG & Truy xuất Thông tin Y tế

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![Milvus](https://img.shields.io/badge/Milvus-2.0-lightgrey.svg)
![Gemini](https://img.shields.io/badge/AI-Gemini_Flash-yellow.svg)

> **Một hệ thống truy xuất thông tin y tế toàn diện, hiệu năng cao, kết hợp truy xuất hai giai đoạn (Bi-Encoder + Cross-Encoder) với Retrieval-Augmented Generation (RAG) để cung cấp câu trả lời y khoa chính xác, dựa trên bằng chứng thực tế.**

---

## Tổng quan

Dự án này triển khai một công cụ tìm kiếm y khoa mạnh mẽ được thiết kế để tìm các bài báo khoa học liên quan (PubMed) dựa trên mô tả bệnh nhân hoặc các câu hỏi y tế. Hệ thống sử dụng pipeline hai giai đoạn để cân bằng giữa tốc độ và độ chính xác, giải quyết vấn đề "ảo giác" (hallucination) trong các mô hình ngôn ngữ lớn (LLM) bằng cách căn cứ câu trả lời vào các tài liệu y văn thực tế đã được tìm kiếm.

### Tính năng nổi bật
- **Truy xuất Hai Giai đoạn (Two-Stage Retrieval)**:
  - **Giai đoạn 1 (Bi-Encoder)**: Tìm kiếm vector tốc độ cao sử dụng Milvus để lấy ra top-k ứng viên tiềm năng (Recall cao).
  - **Giai đoạn 2 (Cross-Encoder)**: Sắp xếp lại (Re-ranking) chính xác sử dụng PubMedBERT để chấm điểm mức độ liên quan (Precision cao).
- ** RAG Chatbot**: Tích hợp `Gemini-2.5-flash` để tổng hợp câu trả lời từ các bài báo đã tìm được với yêu cầu trích dẫn nguồn (PMID) nghiêm ngặt.
- **API Hiệu năng cao**: Được xây dựng với FastAPI, sẵn sàng cho việc mở rộng (scalable).
- **Đánh giá Toàn diện**: Cung cấp các công cụ để phân tích MRR, nDCG và Precision.

---

## Đánh giá Hiệu năng

Chúng tôi đã đánh giá hệ thống so với các baseline (kiểm chứng cơ sở) tiên tiến nhất trên các bộ benchmark truy xuất y tế.

### Bảng Xếp hạng Truy xuất (Retrieval Leaderboard)

| Mô hình | MRR (%) | P@10 (%) | nDCG@10 (%) | R@1k (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Baselines** (Tham khảo từ Leaderboard) | | | | |
| DPR (SciMult-MHAExpert) [3] | **29.89** | **9.35** | **13.79** | **53.71** |
| RRF (Reciprocal Rank Fusion) [4] | 29.86 | 8.86 | 13.36 | 49.45 |
| DPR (PubMedBERT) [4] | 19.83 | 6.51 | 8.87 | 46.23 |
| DPR (BioLinkBERT) [4] | 19.06 | 6.11 | 8.26 | 45.79 |
| DPR (SPECTER) [4] | 17.92 | 5.49 | 7.66 | 42.46 |
| BM25 (Lexical Baseline) [4] | 18.71 | 3.84 | 7.38 | 21.89 |
| bge-base-en-v1.5 [2] | 15.88 | 4.27 | 6.44 | 30.43 |
| MedCPT-d [1] | 13.06 | 2.67 | 4.95 | 19.94 |
| **Hệ thống của Chúng tôi** | | | | |
| 🔹 **Cross-Encoder (Stage 2)** | **19.80** | **6.10** | **8.30** | **45.30** |
| 🔸 Bi-Encoder (Stage 1) | 6.92 | 1.88 | 2.33 | 45.30 |

> **Phân tích**: **Cross-Encoder (Giai đoạn 2)** của chúng tôi cải thiện đáng kể hiệu năng xếp hạng so với kết quả thô từ Bi-Encoder, đạt hiệu năng cạnh tranh với các baseline mạnh dựa trên BERT như DPR (PubMedBERT). Cụ thể, MRR tăng ấn tượng từ ~6.9% lên ~19.8% và độ chính xác (P@10) tăng từ ~1.9% lên ~6.1%.

---

## Kiến trúc Hệ thống

1.  **Xử lý Truy vấn**: Vector hóa câu hỏi của người dùng.
2.  **Tìm kiếm Dense (Milvus)**: Tìm kiếm trong hơn 1 triệu tóm tắt PubMed đã được đánh chỉ mục để lấy Top 100 ứng viên.
3.  **Sắp xếp lại (Re-Ranking)**: Mô hình Cross-Encoder chuyên biệt (PubMedBERT) chấm điểm lại từng cặp (Câu hỏi, Tài liệu).
4.  **Sinh câu trả lời (Generative Answer)**: Top 5 tài liệu tốt nhất được gửi vào LLM (Gemini) làm ngữ cảnh để trả lời câu hỏi.

---

## Cấu trúc Dự án

```bash
DataWarehouse/
├── api_server.py             # Backend API Chính (Các endpoint truy xuất)
├── rag_chatbot_api.py        # API RAG Chatbot riêng biệt
├── retrieve.py               # Class xử lý logic tìm kiếm cốt lõi
├── train_cross_encoder.py    # Script huấn luyện Cross-Encoder
├── insert_to_milvus.py       # Pipeline đánh chỉ mục vào Vector DB
├── corpus_loader.py          # Quản lý & tải dữ liệu Corpus hiệu quả
└── requirements.txt          # Các thư viện phụ thuộc
```

## Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu tiên quyết
- Python 3.10+
- GPU hỗ trợ CUDA (Khuyên dùng cho các mô hình Neural)
- Milvus (Đã cài đặt và đang chạy)

### Cài đặt

1.  **Clone repository**
    ```bash
    git clone https://github.com/yourusername/DataWarehouse.git
    cd DataWarehouse
    ```

2.  **Cài đặt các thư viện**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Thiết lập Môi trường**
    Tạo file `.env`:
    ```env
    GOOGLE_API_KEY=your_gemini_key_here
    MILVUS_URI=...
    ```

### Sử dụng

**Chạy RAG API Server:**
```bash
python rag_chatbot_api.py
```
> Server sẽ khởi chạy tại `http://localhost:8001`

**Chạy Đánh giá Truy xuất:**
```bash
python rerank_results.py
```

---
*Dự án Nghiên cứu Truy xuất Thông tin Y tế*
