import os
import json
import time
import sacrebleu
import ctranslate2
import transformers
import pandas as pd

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_translate"))

def load_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    translator = ctranslate2.Translator(path, device="cpu", compute_type="int8")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    return translator, tokenizer

def translate(text, translator, tokenizer):
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch([source_tokens])
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(results[0].hypotheses[0]))

def run_benchmark():
    
    print("⏳ Đang nạp các model OPUS-MT...")
    t_vi_en, tok_vi_en = load_model("opus-mt-vi-en")
    t_en_vi, tok_en_vi = load_model("opus-mt-en-vi")
    t_zh_vi, tok_zh_vi = load_model("opus-mt-zh-vi")
    t_en_zh, tok_en_zh = load_model("opus-mt-en-zh")

    try:
        with open("milestone6\\test_data_50.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc file JSON: {e}")
        return

    print("\n🚀 BẮT ĐẦU BENCHMARK TRÊN TẬP DỮ LIỆU...\n")

    directions = ["vi_en", "en_vi", "zh_vi", "vi_zh"]
    # Thêm mảng "srcs" để lưu câu gốc xuất ra Excel
    results = {d: {"srcs": [], "preds": [], "refs": [], "latencies": []} for d in directions}

    if "vi_en" in data:
        print(f"🔄 Đang chấm điểm chiều Vi -> En ({len(data['vi_en'])} câu)...")
        for item in data["vi_en"]:
            t0 = time.time()
            pred = translate(item["src"], t_vi_en, tok_vi_en)
            results["vi_en"]["latencies"].append(time.time() - t0)
            results["vi_en"]["preds"].append(pred)
            results["vi_en"]["refs"].append(item["ref"])
            results["vi_en"]["srcs"].append(item["src"])

    if "en_vi" in data:
        print(f"🔄 Đang chấm điểm chiều En -> Vi ({len(data['en_vi'])} câu)...")
        for item in data["en_vi"]:
            t0 = time.time()
            pred = translate(item["src"], t_en_vi, tok_en_vi)
            results["en_vi"]["latencies"].append(time.time() - t0)
            results["en_vi"]["preds"].append(pred)
            results["en_vi"]["refs"].append(item["ref"])
            results["en_vi"]["srcs"].append(item["src"])

    if "zh_vi" in data:
        print(f"🔄 Đang chấm điểm chiều Zh -> Vi ({len(data['zh_vi'])} câu)...")
        for item in data["zh_vi"]:
            t0 = time.time()
            pred = translate(item["src"], t_zh_vi, tok_zh_vi)
            results["zh_vi"]["latencies"].append(time.time() - t0)
            results["zh_vi"]["preds"].append(pred)
            results["zh_vi"]["refs"].append(item["ref"])
            results["zh_vi"]["srcs"].append(item["src"])

    if "vi_zh" in data:
        print(f"🌉 Đang chấm điểm chiều Vi -> Zh [Bắc cầu] ({len(data['vi_zh'])} câu)...")
        for item in data["vi_zh"]:
            t0 = time.time()
            temp_en = translate(item["src"], t_vi_en, tok_vi_en)
            pred_zh = translate(temp_en, t_en_zh, tok_en_zh)
            
            results["vi_zh"]["latencies"].append(time.time() - t0)
            results["vi_zh"]["preds"].append(pred_zh)
            results["vi_zh"]["refs"].append(item["ref"])
            results["vi_zh"]["srcs"].append(item["src"])

    # ================= ĐÁNH GIÁ & XUẤT EXCEL =================
    print("\n📊 BẢNG KẾT QUẢ ĐÁNH GIÁ (OPUS-MT INT8)")
    print("-" * 65)
    print(f"{'Chiều Dịch':<10} | {'Độ trễ TB (s/câu)':<20} | {'Điểm BLEU Score':<15}")
    print("-" * 65)

    summary_data = [] # Mảng lưu dữ liệu tổng hợp cho Excel

    for d in directions:
        if not results[d]["latencies"]: continue
            
        avg_latency = sum(results[d]["latencies"]) / len(results[d]["latencies"])
        if d.endswith("zh"):
            bleu = sacrebleu.corpus_bleu(results[d]["preds"], [results[d]["refs"]], tokenize="zh").score
        else:
            bleu = sacrebleu.corpus_bleu(results[d]["preds"], [results[d]["refs"]]).score
            
        dir_name = d.replace('_', '-')
        print(f"{dir_name:<10} | {avg_latency:<20.4f} | {bleu:<15.2f}")
        
        # Thêm vào bảng tóm tắt
        summary_data.append({
            "Chiều Dịch": dir_name,
            "Độ trễ TB (s/câu)": round(avg_latency, 4),
            "Điểm BLEU Score": round(bleu, 2)
        })
    print("-" * 65)

    # Khối lệnh tạo file Excel
    print("\n💾 Đang xuất dữ liệu ra file Excel...")
    excel_path = "milestone6\\benchmark_results.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Ghi Sheet Tóm tắt
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            
            # Ghi các Sheet chi tiết cho từng chiều dịch
            for d in directions:
                if results[d]["srcs"]:
                    df_detail = pd.DataFrame({
                        "Câu gốc (Source)": results[d]["srcs"],
                        "Câu chuẩn (Reference)": results[d]["refs"],
                        "Máy dịch (Prediction)": results[d]["preds"],
                        "Độ trễ (s)": [round(lat, 4) for lat in results[d]["latencies"]]
                    })
                    df_detail.to_excel(writer, sheet_name=d.replace('_', '-'), index=False)
                    
        print(f"✅ Đã lưu thành công tại: {excel_path}")
    except Exception as e:
        print(f"❌ Lỗi khi xuất Excel: {e}")

if __name__ == "__main__":
    run_benchmark()