# 🚨 HƯỚNG DẪN FIX LỖI TRAINING LOSS = 0

## 📌 TÌNH HUỐNG HIỆN TẠI

Bạn đang gặp lỗi:
```
Training Loss: 0.000000
Validation Loss: nan
ROUGE: 0.000000
Generated: <0x03>  ← Garbage output!
```

**Đây là lỗi NGHIÊM TRỌNG** - model hoàn toàn không học được gì.

---

## 🔍 BƯỚC 1: CHẨN ĐOÁN (BẮT BUỘC)

### Chạy script chẩn đoán:

1. Upload file `diagnostic_script.py` vào môi trường của bạn
2. Đảm bảo folder `data/` có đầy đủ files CSV
3. Chạy:
```bash
python diagnostic_script.py
```

### Script này sẽ kiểm tra:
- ✅ Model có load đúng không?
- ✅ Data có hợp lệ không?
- ✅ Tokenization có đúng không?
- ✅ Forward pass có tính được loss không?
- ✅ Labels có bị toàn -100 không?
- ✅ Gradients có được tính không?

### Đọc kết quả:

Script sẽ in ra **DIAGNOSIS** ở cuối. Tìm các dòng:
- ❌ Màu đỏ = LỖI NGHIÊM TRỌNG
- ⚠️  Màu vàng = CẢNH BÁO
- ✅ Màu xanh = OK

**QUAN TRỌNG:** Nếu có BẤT KỲ dòng ❌ nào, đừng tiếp tục train!

---

## 🔧 BƯỚC 2: FIX THEO CHẨN ĐOÁN

### Vấn đề 1: "Training loss is 0"

**Nguyên nhân có thể:**
1. All labels are -100 (padding)
2. Model weights frozen
3. Wrong loss computation

**Fix:**
- Dùng notebook `mt5_emergency_fix.ipynb`
- Notebook này có check từng bước
- Sẽ DỪNG NGAY nếu phát hiện vấn đề

### Vấn đề 2: "Generated sentinel tokens" 

**Triệu chứng:** 
```
Generated: <extra_id_0> <extra_id_37>
```

**Nguyên nhân:**
- Tokenizer không match model
- Model không được init đúng

**Fix:**
```python
# Đảm bảo load đúng cách:
MODEL_NAME = "google/mt5-small"  # Viết TRƯỚC!
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
```

### Vấn đề 3: "All labels are -100"

**Triệu chứng:**
```
Valid labels: 0/128 (0.0%)
```

**Nguyên nhân:**
- Data collator sai cấu hình
- Preprocessing function sai

**Fix:**
```python
# Đảm bảo dùng text_target:
labels = tokenizer(
    text_target=examples["summary"],  # ← text_target!
    max_length=128,
    truncation=True,
    padding=False
)
```

### Vấn đề 4: "Loss is NaN"

**Nguyên nhân:**
- FP16 precision issues
- Learning rate quá cao
- Gradient explosion

**Fix:**
```python
training_args = Seq2SeqTrainingArguments(
    fp16=False,  # ← TẮT FP16!
    learning_rate=5e-5,  # Không quá cao
    gradient_clip_norm=1.0,  # Clip gradients
)
```

### Vấn đề 5: "No gradients computed"

**Nguyên nhân:**
- Model trong eval mode
- Parameters bị freeze

**Fix:**
```python
# Explicitly unfreeze:
for param in model.parameters():
    param.requires_grad = True

# Ensure training mode:
model.train()
```

---

## 🚀 BƯỚC 3: SỬ DỤNG EMERGENCY FIX NOTEBOOK

File: `mt5_emergency_fix.ipynb`

### Đặc điểm:
1. ✅ **Tích hợp diagnostic** - check mọi thứ trước khi train
2. ✅ **FP16 disabled** - tránh numerical issues
3. ✅ **Higher learning rate** - 1e-4 thay vì 1e-5
4. ✅ **Explicit checks** - DỪNG NGAY nếu có vấn đề
5. ✅ **Frequent logging** - log mỗi 10 steps

### Cách dùng:

1. Upload `mt5_emergency_fix.ipynb`
2. Chạy từng cell theo thứ tự
3. **ĐỌC KỸ OUTPUT** của mỗi cell
4. Nếu thấy ❌ → DỪNG và báo lỗi
5. Chỉ train khi thấy "ALL CHECKS PASSED"

### Cell quan trọng nhất:

**"FINAL CHECK Before Training"** - Cell này sẽ:
- Test forward pass với batch thật
- Kiểm tra loss > 0
- Kiểm tra gradients
- **DỪNG NGAY nếu loss = 0 hoặc NaN**

```
Expected output:
   Loss: 5.2341  ← PHẢI > 0!
   ✅ Loss is normal!
   ✅ Gradients OK
   ✅ ALL CHECKS PASSED - READY TO TRAIN
```

Nếu thấy:
```
   Loss: 0.0000
   ❌ CRITICAL ERROR: Loss is 0!
   DO NOT START TRAINING!
```

→ **DỪNG NGAY**, không train!

---

## 📊 BƯỚC 4: THEO DÕI TRAINING

### Bước đầu tiên (step 0-10):

**QUAN TRỌNG:** Loss ở 10 bước đầu là CHỈ SỐ QUAN TRỌNG NHẤT!

✅ **Normal:**
```
Step 1: Loss 7.234
Step 2: Loss 6.891
Step 3: Loss 6.543
...
Step 10: Loss 5.123
```

❌ **Abnormal:**
```
Step 1: Loss 0.000  ← DỪNG NGAY!
```

hoặc

```
Step 1: Loss nan  ← DỪNG NGAY!
```

### Sau 500 steps:

✅ **Good:**
```
Step 500:
  Training Loss: 2.543
  Validation Loss: 2.891
  ROUGE-1: 0.2531
  ROUGE-2: 0.1234
```

❌ **Bad:**
```
Step 500:
  Training Loss: 0.000000  ← Vẫn lỗi!
  Validation Loss: nan
  ROUGE: 0.000000
```

Nếu vẫn thấy 0.000000 sau 500 steps → **DỪN NGAY**, có vấn đề căn bản!

---

## 🎯 KỲ VỌNG SAU KHI FIX ĐÚNG

### Epoch 1:
- Training loss: 5-8 → 2-3
- Val loss: 3-4
- ROUGE-1: 25-35%
- Generated text: Có nghĩa nhưng chưa tốt

### Epoch 2:
- Training loss: 2-3 → 1.5-2
- Val loss: 2.5-3
- ROUGE-1: 40-55%
- Generated text: Tốt hơn

### Epoch 3:
- Training loss: 1.5-2
- Val loss: 2-2.5
- ROUGE-1: 50-70%
- ROUGE-2: 30-50%
- Generated text: Tốt

---

## 🔍 DEBUGGING CHECKLIST

Trước khi train, check:
- [ ] `diagnostic_script.py` chạy không có lỗi ❌
- [ ] Test loss > 0 (thường 2-8)
- [ ] Test generation không ra sentinel tokens
- [ ] Labels không phải toàn -100
- [ ] Model parameters có requires_grad=True

Trong khi train:
- [ ] Step 1 loss > 0
- [ ] Loss giảm dần
- [ ] Không có NaN
- [ ] ROUGE > 0 sau eval đầu

Nếu fail bất kỳ check nào → **DỪNG VÀ DEBUG**

---

## 💡 QUICK FIXES

### Fix 1: Restart mọi thứ
```python
# Kill all processes
!pkill -9 python

# Clear GPU
torch.cuda.empty_cache()
gc.collect()

# Reload everything từ đầu
```

### Fix 2: Reduce complexity
```python
# Train với subset nhỏ để test
small_train = tokenized_datasets["train"].select(range(100))
small_val = tokenized_datasets["validation"].select(range(20))
```

### Fix 3: Simplify settings
```python
training_args = Seq2SeqTrainingArguments(
    # Minimal settings
    output_dir="./test",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    fp16=False,  # Disable
    gradient_checkpointing=False,  # Disable
)
```

---

## 📞 NẾU VẪN KHÔNG ĐƯỢC

### Thu thập thông tin:

1. Chạy `diagnostic_script.py`, copy toàn bộ output
2. Screenshot 10 bước training đầu tiên
3. Copy thông tin:
   - GPU model
   - CUDA version: `torch.version.cuda`
   - PyTorch version: `torch.__version__`
   - Transformers version: `transformers.__version__`

### Thử model khác:

Nếu mT5 vẫn lỗi, thử:
```python
# ViT5 - specifically for Vietnamese
MODEL_NAME = "VietAI/vit5-base"
```

hoặc

```python
# Smaller mT5
MODEL_NAME = "google/mt5-base"
```

---

## 📦 FILES SUMMARY

1. **diagnostic_script.py** - Chạy ĐẦU TIÊN để tìm lỗi
2. **mt5_emergency_fix.ipynb** - Notebook với tất cả fix và checks
3. **LỖI_VÀ_GIẢI_PHÁP.md** - Chi tiết về từng lỗi
4. **HƯỚNG_DẪN_FIX.md** - File này

---

## 🎓 HIỂU VỀ LOSS = 0

**Tại sao loss = 0 là nghiêm trọng?**

1. Loss = 0 nghĩa là model nghĩ nó đã "perfect"
2. Nhưng ROUGE = 0 chứng tỏ output là garbage
3. Điều này chứng tỏ:
   - Loss không được tính đúng
   - Labels bị sai
   - Model không thực sự train

**Normal loss should be:**
- Initial: 5-8
- After training: 1.5-2
- **NEVER 0!**

---

## ✅ SUCCESS CRITERIA

Bạn đã fix xong khi thấy:

```
Step 1: Loss 6.234 ✅
Step 50: Loss 4.567 ✅
Step 100: Loss 3.456 ✅
Step 500:
  Training Loss: 2.345 ✅
  Validation Loss: 2.891 ✅
  ROUGE-1: 0.2891 ✅ (NOT 0!)
  
[EVAL] Sample prediction: Hà Nội công bố kết quả... ✅ (Vietnamese!)
```

KHÔNG PHẢI:
```
Step 1: Loss 0.000 ❌
Step 500: Loss 0.000, ROUGE 0.000 ❌
[EVAL] Sample prediction: <0x03> ❌
```

---

Chúc may mắn! 🍀
