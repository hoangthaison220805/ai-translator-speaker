import os
import json
import pandas as pd
from datasets import load_dataset
import huggingface_hub

print("=" * 60)
print("BƯỚC 2: THU THẬP DỮ LIỆU TRAINING (ĐÃ FIX LỖI)")
print("=" * 60)

# ⚠️ THAY TOKEN CỦA BẠN VÀO ĐÂY (Bắt đầu bằng hf_...)
HF_TOKEN = "hf_YOUR_TOKEN_HERE" 

# Đăng nhập ngầm vào Hugging Face để tải FLORES
huggingface_hub.login(token=HF_TOKEN)

os.makedirs("finetune_data", exist_ok=True)

# ============================================================
# A. CORPUS Vi-En (từ OPUS-100)
# ============================================================
print("\n📥 [1/3] Tải corpus Vi-En từ OPUS-100...")
try:
    opus = load_dataset("Helsinki-NLP/opus-100", "en-vi", trust_remote_code=True)

    vi_en_train = [{"vi": item["translation"].get("vi", "").strip(), "en": item["translation"].get("en", "").strip()} 
                   for item in opus["train"] 
                   if len(item["translation"].get("vi", "").split()) >= 3]

    vi_en_val = [{"vi": item["translation"].get("vi", "").strip(), "en": item["translation"].get("en", "").strip()} 
                 for item in opus["validation"]]

    # ĐÃ FIX: Thêm encoding='utf-8-sig' để Excel không bị lỗi font
    pd.DataFrame(vi_en_train).to_csv("finetune_data/vi_en_train.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(vi_en_val).to_csv("finetune_data/vi_en_val.csv", index=False, encoding='utf-8-sig')
    print(f"   ✅ Vi-En train: {len(vi_en_train):,} cặp câu")
    print(f"   ✅ Vi-En val:   {len(vi_en_val):,} cặp câu")

except Exception as e:
    print(f"   ❌ Lỗi tải OPUS-100: {e}")

# ============================================================
# B. CORPUS Vi-Zh (Đã Fix ID ngôn ngữ thành vie-cmn_Hans)
# ============================================================
print("\n📥 [2/3] Tải corpus Vi-Zh từ Tatoeba...")
try:
    # Đổi ID thành vie-cmn_Hans theo chuẩn của Tatoeba
    tatoeba = load_dataset("Helsinki-NLP/tatoeba_mt", "vie-cmn_Hans", trust_remote_code=True)
    
    vi_zh_data = []
    # Dữ liệu Tatoeba thường nằm trong split 'train'
    for item in tatoeba["test"]: 
        vi = item.get("sourceString", "").strip()
        zh = item.get("targetString", "").strip()
        if vi and zh:
            vi_zh_data.append({"vi": vi, "zh": zh})

    split_idx = int(len(vi_zh_data) * 0.9)
    
    pd.DataFrame(vi_zh_data[:split_idx]).to_csv("finetune_data/vi_zh_train.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame(vi_zh_data[split_idx:]).to_csv("finetune_data/vi_zh_val.csv", index=False, encoding='utf-8-sig')
    print(f"   ✅ Vi-Zh Tatoeba Train: {len(vi_zh_data[:split_idx]):,} cặp câu")
    print(f"   ✅ Vi-Zh Tatoeba Val:   {len(vi_zh_data[split_idx:]):,} cặp câu")
    
except Exception as e:
    print(f"   ❌ Lỗi tải Vi-Zh: {e}")

# ============================================================
# C. FLORES-200 TEST SET (Đã Fix cách ghép đôi Vi-Zh)
# ============================================================
print("\n📥 [3/3] Tải FLORES-200 (test set chuẩn)...")
try:
    # 1. Xử lý Vi-En (Giữ nguyên vì đã chạy tốt)
    flores_vi_en = load_dataset("facebook/flores", "eng_Latn-vie_Latn", trust_remote_code=True, token=HF_TOKEN)
    flores_vi_en_data = [{"vi": item["sentence_vie_Latn"], "en": item["sentence_eng_Latn"]} for item in flores_vi_en["devtest"]]
    pd.DataFrame(flores_vi_en_data).to_csv("finetune_data/flores_test_vi_en.csv", index=False, encoding='utf-8-sig')
    print(f"   ✅ FLORES Vi-En: {len(flores_vi_en_data)} câu")

    # 2. Xử lý Vi-Zh (Ghép đôi 2 bộ dữ liệu)
    # Tải gói tiếng Trung (kết nối qua tiếng Anh)
    flores_en_zh = load_dataset("facebook/flores", "eng_Latn-zho_Hans", trust_remote_code=True, token=HF_TOKEN)
    
    # Sử dụng hàm zip() để ghép dòng thứ i của tiếng Việt với dòng thứ i của tiếng Trung
    flores_vi_zh_data = [
        {"vi": item_vi["sentence_vie_Latn"], "zh": item_zh["sentence_zho_Hans"]} 
        for item_vi, item_zh in zip(flores_vi_en["devtest"], flores_en_zh["devtest"])
    ]
    pd.DataFrame(flores_vi_zh_data).to_csv("finetune_data/flores_test_vi_zh.csv", index=False, encoding='utf-8-sig')
    print(f"   ✅ FLORES Vi-Zh: {len(flores_vi_zh_data)} câu (Đã ghép đôi thành công)")

except Exception as e:
    print(f"   ⚠️ Lỗi FLORES: {e}")

print("\n" + "=" * 60)
print("📂 DỮ LIỆU ĐÃ TẢI THÀNH CÔNG:")
for f in sorted(os.listdir("finetune_data")):
    if f.endswith(".csv"):
        path = os.path.join("finetune_data", f)
        df = pd.read_csv(path)
        print(f"   {f:<35} {len(df):>8,} cặp câu")