# step3_prepare_finetune.py
# Chuẩn bị dữ liệu 4 chiều + chạy fine-tune
# Chạy: python step3_prepare_finetune.py

import os
import re
import json
import time
import unicodedata
import numpy as np
import pandas as pd
import torch
import evaluate
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_NAME     = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR     = "./nllb-4way-finetuned"
MAX_LENGTH     = 256
BATCH_SIZE     = 8     # ← Giảm xuống 2 nếu Out of Memory
GRAD_ACCUM     = 4       # effective batch = 8 × 4 = 32
LEARNING_RATE  = 5e-5
NUM_EPOCHS     = 3
MAX_TRAIN_SAMPLES = 50000   # Giới hạn số câu train (để chạy nhanh hơn)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

LANG_CODES = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
}

print("=" * 65)
print("BƯỚC 3: CHUẨN BỊ DỮ LIỆU + FINE-TUNE 4 CHIỀU")
print("=" * 65)
print(f"  Device: {DEVICE.upper()}")
print(f"  Batch:  {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Max train samples: {MAX_TRAIN_SAMPLES:,}")

# ============================================================
# PHẦN A: CHUẨN BỊ DỮ LIỆU 4 CHIỀU
# ============================================================
print("\n" + "─" * 65)
print("📂 PHẦN A: Gộp và làm sạch dữ liệu")
print("─" * 65)

def clean(text):
    """Làm sạch text"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()

def is_valid(src, tgt):
    """Kiểm tra cặp câu hợp lệ"""
    if not src or not tgt:
        return False
    s_len, t_len = len(src.split()), len(tgt.split())
    if s_len < 3 or t_len < 3:
        return False
    if s_len > 100 or t_len > 100:
        return False
    ratio = max(s_len, t_len) / max(min(s_len, t_len), 1)
    if ratio > 5:
        return False
    return True

all_rows = []

# Đọc Vi-En
for csv_file in ["finetune_data/vi_en_train.csv", "finetune_data/vi_en_val.csv"]:
    if not os.path.exists(csv_file):
        continue
    df = pd.read_csv(csv_file)
    count = 0
    for _, row in df.iterrows():
        vi = clean(str(row.get("vi", "")))
        en = clean(str(row.get("en", "")))
        if is_valid(vi, en):
            # Chiều Vi → En
            all_rows.append({
                "source": vi, "target": en,
                "src_lang": "vie_Latn", "tgt_lang": "eng_Latn"
            })
            # Chiều ngược En → Vi
            all_rows.append({
                "source": en, "target": vi,
                "src_lang": "eng_Latn", "tgt_lang": "vie_Latn"
            })
            count += 1
    print(f"  ✅ {csv_file}: {count:,} cặp → {count*2:,} dòng (×2 chiều)")

# Đọc Vi-Zh
for csv_file in ["finetune_data/vi_zh_train.csv", "finetune_data/vi_zh_val.csv"]:
    if not os.path.exists(csv_file):
        continue
    df = pd.read_csv(csv_file)
    count = 0
    for _, row in df.iterrows():
        vi = clean(str(row.get("vi", "")))
        zh = clean(str(row.get("zh", "")))
        if vi and zh and len(vi) >= 5 and len(zh) >= 2:
            all_rows.append({
                "source": vi, "target": zh,
                "src_lang": "vie_Latn", "tgt_lang": "zho_Hans"
            })
            all_rows.append({
                "source": zh, "target": vi,
                "src_lang": "zho_Hans", "tgt_lang": "vie_Latn"
            })
            count += 1
    print(f"  ✅ {csv_file}: {count:,} cặp → {count*2:,} dòng (×2 chiều)")

# Chuyển thành bảng và xóa các câu bị trùng lặp
combined_all = pd.DataFrame(all_rows)
combined_all = combined_all.drop_duplicates(subset=["source", "target"])

# --- BẮT ĐẦU ĐOẠN SỬA: LỌC TRÁNH CÁ LỚN NUỐT CÁ BÉ ---
# Tách riêng 2 cụm ngôn ngữ
df_vi_en = combined_all[(combined_all["src_lang"] == "eng_Latn") | (combined_all["tgt_lang"] == "eng_Latn")]
df_vi_zh = combined_all[(combined_all["src_lang"] == "zho_Hans") | (combined_all["tgt_lang"] == "zho_Hans")]

# Lấy ngẫu nhiên tối đa 45,000 câu Anh, và giữ LẠI TOÀN BỘ câu Trung
df_vi_en = df_vi_en.sample(min(45000, len(df_vi_en)), random_state=42)

# Gộp 2 cụm lại với nhau và xóc bài (trộn ngẫu nhiên)
combined = pd.concat([df_vi_en, df_vi_zh]).sample(frac=1, random_state=42).reset_index(drop=True)
# --- KẾT THÚC ĐOẠN SỬA ---

# Chia train / val (90% / 10%)
split = int(len(combined) * 0.9)
train_df = combined[:split]
val_df   = combined[split:]

print(f"\n📊 Tổng kết:")
print(f"   Tổng sau xử lý: {len(combined):,} dòng")
print(f"   Train: {len(train_df):,}  |  Val: {len(val_df):,}")
print(f"\n   Phân phối theo chiều:")
for (sl, tl), g in train_df.groupby(["src_lang", "tgt_lang"]):
    print(f"     {sl} → {tl}: {len(g):>8,} câu")

# ============================================================
# PHẦN B: FINE-TUNE
# ============================================================
print("\n" + "─" * 65)
print("🏋️ PHẦN B: Fine-tune NLLB-200")
print("─" * 65)

# Load tokenizer & model
print(f"\n📥 Đang tải model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
)
model.gradient_checkpointing_enable()  # Tiết kiệm VRAM
print(f"✅ Model đã tải! Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

# Tạo HuggingFace Dataset
dataset = DatasetDict({
    "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
})

# Tokenize function — xử lý 4 chiều bằng cách đọc src_lang/tgt_lang từ dữ liệu
def tokenize_fn(examples):
    """Tokenize từng câu với đúng src_lang và tgt_lang của nó"""
    input_ids_all      = []
    attention_mask_all = []
    labels_all         = []

    for src, tgt, sl, tl in zip(
        examples["source"], examples["target"],
        examples["src_lang"], examples["tgt_lang"]
    ):
        # 1. Khai báo rõ ngôn ngữ nguồn và ngôn ngữ đích cho NLLB
        tokenizer.src_lang = sl
        tokenizer.tgt_lang = tl 
        
        # 2. Xử lý chuẩn mới: Dùng tham số text_target, KHÔNG dùng as_target_tokenizer()
        enc = tokenizer(src, text_target=tgt, max_length=MAX_LENGTH, truncation=True)

        input_ids_all.append(enc["input_ids"])
        attention_mask_all.append(enc["attention_mask"])
        labels_all.append(enc["labels"])

    return {
        "input_ids":      input_ids_all,
        "attention_mask": attention_mask_all,
        "labels":         labels_all,
    }

print("\n🔄 Tokenize dữ liệu...")
tokenized = dataset.map(
    tokenize_fn, batched=True, batch_size=64,
    remove_columns=["source", "target", "src_lang", "tgt_lang"],
    desc="Tokenizing"
)
print("✅ Tokenize xong!")

# Metric
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    dec_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
    dec_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    result = bleu_metric.compute(predictions=dec_preds, references=dec_labels)
    return {"bleu": round(result["score"], 2)}

# Training Arguments
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=2,
    logging_steps=100,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# BẮT ĐẦU TRAINING!
print("\n" + "=" * 65)
print("🚀 BẮT ĐẦU FINE-TUNE!")
print("   Sau mỗi epoch sẽ in BLEU trên val set")
print("   Model tốt nhất được lưu tự động")
estimated_gpu = NUM_EPOCHS * 20
estimated_cpu = NUM_EPOCHS * 120
print(f"\n⏳ Thời gian ước tính:")
print(f"   GPU T4 (Colab): ~{estimated_gpu} phút")
print(f"   CPU:            ~{estimated_cpu} phút")
print("=" * 65 + "\n")

result = trainer.train()

# Lưu model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/training_results.json", "w") as f:
    json.dump(result.metrics, f, indent=2)

print(f"\n✅ Fine-tune hoàn thành!")
print(f"💾 Model đã lưu tại: {OUTPUT_DIR}")
print(f"📊 Training metrics: {result.metrics}")
print(f"\nTiếp tục: python step4_benchmark_after.py")