---

# 4. Abstractive Summarization

## 4.1 Sequence-to-Sequence Models - Theory

### Nguyên lý / Principle:

**Abstractive Summarization** sử dụng mô hình **Sequence-to-Sequence (Seq2Seq)** để tạo ra bản tóm tắt mới thay vì chọn câu có sẵn.

### Kiến trúc Encoder-Decoder:

**Encoder:**
- Nhận văn bản đầu vào
- Chuyển thành vector biểu diễn (context vector)
- Capture toàn bộ ngữ nghĩa của văn bản

**Decoder:**
- Nhận context vector từ encoder
- Sinh ra từng từ một của bản tóm tắt
- Sử dụng thông tin từ encoder và từ đã sinh trước đó

### Attention Mechanism:

**Vấn đề / Problem:**
- Context vector cố định không thể chứa toàn bộ thông tin
- Với văn bản dài, thông tin bị mất

**Giải pháp / Solution:**
- **Attention** cho phép decoder "tập trung" vào phần khác nhau của input
- Tại mỗi bước sinh từ, decoder xem lại toàn bộ encoder outputs
- Tính trọng số quan trọng cho từng vị trí trong input

### Generation Strategies:

#### 1. Greedy Decoding
- Chọn từ có xác suất cao nhất tại mỗi bước
- **Ưu điểm**: Nhanh
- **Nhược điểm**: Không tối ưu toàn cục

#### 2. Beam Search
- Giữ K candidates tốt nhất (beam width = K)
- **Ưu điểm**: Chất lượng cao hơn greedy
- **Nhược điểm**: Chậm hơn (K lần)

#### 3. Top-k Sampling
- Chọn ngẫu nhiên từ K từ có xác suất cao nhất
- **Ưu điểm**: Đa dạng hơn
- **Nhược điểm**: Có thể không ổn định

#### 4. Top-p (Nucleus) Sampling
- Chọn ngẫu nhiên từ các từ có tổng xác suất >= p
- **Ưu điểm**: Cân bằng giữa đa dạng và chất lượng
- **Nhược điểm**: Cần điều chỉnh p cẩn thận

Trong TextRank, việc chọn số câu N (top N sentences) thường dựa trên các nguyên lý sau:

1. Compression Ratio (Tỷ lệ nén)
Nguyên lý phổ biến nhất:

Chọn N sao cho tóm tắt chiếm 20-30% độ dài văn bản gốc
Ví dụ: Văn bản gốc 20 câu → N = 4-6 câu

N = int(total_sentences * compression_ratio)  # compression_ratio = 0.2-0.3
2. Fixed Length (Độ dài cố định)
Chọn số câu cố định dựa trên yêu cầu:
Short summary: N = 3-5 câu
Medium summary: N = 5-10 câu
Long summary: N = 10+ câu
3. Score Threshold (Ngưỡng điểm)
Chọn tất cả câu có PageRank score > threshold
Ví dụ: Chọn câu có score > mean + std_deviation

threshold = mean_score + k * std_score  # k thường là 0.5-1.0
selected = sentences[scores > threshold]
4. Information Coverage (Độ phủ thông tin)
Chọn N sao cho:
Đạt ngưỡng information coverage nhất định (ví dụ: 80% thông tin quan trọng)
Sử dụng cumulative score để xác định

sorted_scores = sort_descending(scores)
cumsum = cumulative_sum(sorted_scores)
N = find_index(cumsum >= 0.8 * total_sum)
5. Dynamic Selection (Lựa chọn động)
Kết hợp nhiều yếu tố:

Độ dài văn bản gốc
Độ phức tạp của chủ đề
Yêu cầu của người dùng
Trong thực tế phổ biến nhất: Sử dụng compression ratio kết hợp với minimum/maximum constraints:


N = max(min_sentences, min(max_sentences, 
        int(total_sentences * compression_ratio)))
Điều này đảm bảo:

Tóm tắt không quá ngắn (min_sentences)
Không quá dài (max_sentences)
Tỷ lệ hợp lý với văn bản gốc
diễn giải công thức
TextRank Algorithm - Theory
 PhoBERT for Vietnamese:
- PhoBERT: BERT model được huấn luyện cho tiếng Việt
- Tạo ra embeddings chất lượng cao cho câu tiếng Việt
- Hiểu được ngữ nghĩa và ngữ cảnh

PageRank Formula: PR(Vi) = (1-d) + d * Σ(PR(Vj) / |Out(Vj)|)
Trong đó:
- `PR(Vi)`: PageRank score của câu i
- `d`: Damping factor (thường là 0.85)
- `Vj`: Các câu có liên kết đến Vi
- `Out(Vj)`: Số câu mà Vj liên kết tới
Diễn giải chi tiết công thức PageRank trong TextRank
Công thức gốc:

PR(Vi) = (1-d) + d * Σ(PR(Vj) / |Out(Vj)|)
Phân tích từng thành phần:
1. PR(Vi) - PageRank score của câu i
Điểm số quan trọng của câu thứ i
Giá trị càng cao → câu càng quan trọng
Khởi tạo ban đầu: tất cả câu có điểm = 1/N (N là tổng số câu)
2. (1-d) - Base Score (Điểm cơ bản)
d = damping factor (thường là 0.85)
(1-d) = 0.15 - điểm tối thiểu mỗi câu nhận được
Ý nghĩa: Ngay cả câu không được câu nào liên kết tới vẫn có điểm cơ bản
Giống như: "Random surfer" trong PageRank - xác suất người đọc nhảy ngẫu nhiên đến câu này
3. d - Damping Factor (Hệ số giảm chấn)
d = 0.85 (giá trị chuẩn)
Xác suất người đọc tiếp tục đi theo liên kết (thay vì nhảy ngẫu nhiên)
85%: đi theo liên kết
15%: nhảy ngẫu nhiên
4. Σ - Tổng (Summation)
Tính tổng đóng góp từ TẤT CẢ các câu Vj có liên kết đến Vi
Liên kết = có độ tương tự (similarity) cao giữa 2 câu
5. Vj - Các câu liên kết ĐẾN Vi
Tập hợp các câu có edge (cạnh) trỏ đến câu Vi
Trong TextRank: Vj là các câu có cosine similarity với Vi > threshold
Ví dụ: Nếu câu 2, 5, 7 tương tự câu 1 → V2, V5, V7 liên kết đến V1
6. PR(Vj) - Điểm của câu Vj
Điểm PageRank của câu j (câu đang liên kết đến Vi)
Câu Vj càng quan trọng → đóng góp nhiều điểm cho Vi
Ý nghĩa: "Vote" từ câu quan trọng có giá trị cao hơn
7. |Out(Vj)| - Số câu mà Vj liên kết TỚI
Số lượng câu khác mà Vj có edge đến
Chia đều điểm: Vj chia điểm của nó cho tất cả câu mà nó liên kết tới
Ví dụ:
Vj liên kết đến 4 câu → mỗi câu nhận PR(Vj)/4
Vj chỉ liên kết đến 1 câu → câu đó nhận toàn bộ PR(Vj)
Ví dụ minh họa:
Giả sử:
Câu 1 (V1) được liên kết bởi câu 2 (V2) và câu 3 (V3)
PR(V2) = 0.3, V2 liên kết đến 2 câu → |Out(V2)| = 2
PR(V3) = 0.5, V3 liên kết đến 5 câu → |Out(V3)| = 5
d = 0.85
Tính PR(V1):

PR(V1) = (1 - 0.85) + 0.85 * [PR(V2)/|Out(V2)| + PR(V3)/|Out(V3)|]

PR(V1) = 0.15 + 0.85 * [0.3/2 + 0.5/5]

PR(V1) = 0.15 + 0.85 * [0.15 + 0.1]

PR(V1) = 0.15 + 0.85 * 0.25

PR(V1) = 0.15 + 0.2125

PR(V1) = 0.3625
Ý nghĩa tổng thể:
Câu quan trọng khi:
Nhiều câu liên kết đến (nhiều Vj)
Được liên kết bởi câu quan trọng (PR(Vj) cao)
Câu liên kết tới ít câu khác (|Out(Vj)| nhỏ - không chia sẻ điểm nhiều)
Giống như đánh giá uy tín:
Nhiều người giới thiệu bạn = tốt
Được người nổi tiếng giới thiệu = rất tốt
Người đó chỉ giới thiệu ít người = bạn nhận nhiều "uy tín" hơn
Thuật toán lặp (Iterative):
Khởi tạo: Tất cả câu có PR = 1/N
Lặp lại công thức cho đến khi hội tụ (thay đổi < threshold)
Thường 10-20 vòng lặp là đủ
So sánh với PageRank gốc:
Khía cạnh	PageRank (Web)	TextRank (Văn bản)
Nodes	Trang web	Câu
Edges	Hyperlinks	Độ tương tự (similarity)
Weight	Có/không link	Cosine similarity score
Direction	Directed	Undirected (thường)
Trong TextRank, thường sử dụng undirected graph (đồ thị vô hướng) nên nếu V1 tương tự V2 → cả 2 chiều đều có edge.