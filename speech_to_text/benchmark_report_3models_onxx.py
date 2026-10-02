import os
import time
import datetime
import pandas as pd
import re
import wave
import numpy as np
import zhconv
import sherpa_onnx
import traceback
import sys
# ==========================================
# 0. CẤU HÌNH BÀI TEST
# ==========================================
# Đảm bảo đường dẫn tới 3 thư mục chứa file ONNX là chính xác
DIR_MODEL_VI = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/milestone2/model_VI"
DIR_MODEL_EN = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/milestone2/model_EN"
DIR_MODEL_ZH = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/milestone2/model_ZH"

# File Excel chứa 150 file âm thanh (cần có các cột: file_path, language, ground_truth)
# Ngôn ngữ trong cột 'language' nên là 'vi', 'en', hoặc 'zh'
EXCEL_INPUT = "D:\\Project\\AIOT_training_khoahoc\\python_code_begin\\AI_translator\\milestone1\\benmark_miletones1_filled.xlsx"

thoi_gian = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXCEL_OUTPUT = f"D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/milestone2/Report_Benchmark_3Models_onxx_{thoi_gian}.xlsx"

# ==========================================
# 1. HÀM CHUẨN HÓA & ĐỌC ÂM THANH
# ==========================================
def clean_text(text):
    text = str(text).lower()
    if re.search(r'[\u4e00-\u9fff]', text):
        text = zhconv.convert(text, 'zh-hans')
        text = re.sub(r'[^\w]', '', text)
        text = " ".join(list(text))
    else:
        text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def read_audio_for_sherpa(file_path):
    with wave.open(file_path, 'rb') as f:
        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return audio, f.getframerate()

# ==========================================
# 2. HÀM KHỞI TẠO MÔ HÌNH THÔNG MINH
# ==========================================
def load_sherpa_model(model_path, lang_name):
    # Tự động xác định Streaming hay Offline dựa vào ngôn ngữ
    is_streaming = False if lang_name.lower() == 'việt' else True
    
    print(f"Đang nạp mô hình [Tiếng {lang_name}] INT8 (Streaming={is_streaming})...")
    
    try:
        encoder_file, decoder_file, joiner_file = "", "", ""
        
        for file in os.listdir(model_path):
            if file.startswith("encoder") and file.endswith(".int8.onnx"):
                encoder_file = os.path.join(model_path, file)
            elif file.startswith("decoder") and file.endswith(".int8.onnx"):
                decoder_file = os.path.join(model_path, file)
            elif file.startswith("joiner") and file.endswith(".int8.onnx"):
                joiner_file = os.path.join(model_path, file)
                
        if not encoder_file or not decoder_file or not joiner_file:
            raise FileNotFoundError(f"Không tìm thấy đủ 3 file int8.onnx trong {model_path}.")

        # Phân luồng khởi tạo
        if is_streaming:
            model = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=f"{model_path}/tokens.txt",
                encoder=encoder_file,
                decoder=decoder_file,
                joiner=joiner_file,
                num_threads=4, 
                sample_rate=16000,
                feature_dim=80
            )
        else:
            model = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=f"{model_path}/tokens.txt",
                encoder=encoder_file,
                decoder=decoder_file,
                joiner=joiner_file,
                num_threads=4, 
                sample_rate=16000,
                feature_dim=80
            )
            
        print(f"✅ Nạp thành công [Tiếng {lang_name}]!")
        
        # Đóng gói mô hình và trạng thái streaming để trả về cho vòng lặp
        return {"model": model, "is_streaming": is_streaming}
        
    except Exception as e:
        print(f"❌ LỖI nạp mô hình {lang_name}: {e}")
        sys.exit(1)

# Nạp 3 mô hình (Code gọn gàng hơn rất nhiều)
print("\n--- BẮT ĐẦU NẠP CÁC MÔ HÌNH ---")
models = {
    'vi': load_sherpa_model(DIR_MODEL_VI, "Việt"),
    'en': load_sherpa_model(DIR_MODEL_EN, "Anh"),
    'zh': load_sherpa_model(DIR_MODEL_ZH, "Trung")
}

# ==========================================
# 3. VÒNG LẶP BENCHMARK 150 FILE
# ==========================================
try:
    df = pd.read_excel(EXCEL_INPUT)
except Exception as e:
    print(f"❌ LỖI đọc file Excel: {e}")
    sys.exit(1)

ket_qua_test = []
from jiwer import wer

print(f"\n--- BẮT ĐẦU BENCHMARK {len(df)} FILE ÂM THANH ---")

for index, row in df.iterrows():
    audio_file = str(row['file_path']).strip()
    # Ép kiểu ngôn ngữ về chữ thường và loại bỏ khoảng trắng (vd: ' vi ' -> 'vi')
    lang = str(row['language']).strip().lower() 
    truth_text = clean_text(row['ground_truth'])
    
    print(f"[{index+1}/{len(df)}] Đang giải mã [{lang.upper()}]: {audio_file}...")
    
    # Bỏ qua nếu cột ngôn ngữ bị ghi sai
    if lang not in models:
        print(f"   ⚠️ Bỏ qua: Ngôn ngữ '{lang}' không hợp lệ (Chỉ nhận vi, en, zh).")
        continue
        
    try:
        start_time = time.time()
        
        # 1. Đọc file âm thanh
        audio_data, sample_rate = read_audio_for_sherpa(audio_file)
        # --- THÊM ĐOẠN NÀY ĐỂ CHỐNG NUỐT CHỮ ---
        # Tạo 0.5 giây khoảng lặng (dữ liệu mảng toàn số 0)
        silence_duration = 1.0
        silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
        
        # Nối khoảng lặng vào đầu file âm thanh thật
        audio_data = np.concatenate((silence, audio_data))
        # ---------------------------------------
        # 2. Chọn đúng mô hình theo ngôn ngữ (Định tuyến)
        active_model = models[lang]["model"]
        is_streaming = models[lang]["is_streaming"]
        
        # 3. Tạo luồng và nạp dữ liệu
        stream = active_model.create_stream()
        stream.accept_waveform(sample_rate, audio_data)
        
        # 4. RẼ NHÁNH: XỬ LÝ THEO KIỂU MÔ HÌNH (HYBRID DECODING)
        if is_streaming:
            # --- Dành cho Tiếng Anh & Trung (Streaming Model) ---
            stream.input_finished() # Bắt buộc báo hiệu hết file
            while active_model.is_ready(stream):
                active_model.decode_stream(stream)
                
            # Xử lý kết quả trả về của Streaming Model (tùy version Sherpa-ONNX)
            result_raw = active_model.get_result(stream)
            pred_text_raw = result_raw if isinstance(result_raw, str) else getattr(result_raw, 'text', '')
            
        else:
            # --- Dành cho Tiếng Việt (Offline Model) ---
            active_model.decode_stream(stream)
            
            # Xử lý kết quả trả về của Offline Model
            result_raw = active_model.get_result(stream)
            pred_text_raw = result_raw.text

        # 5. Tính toán Latency và WER
        pred_text = clean_text(pred_text_raw)
        latency = time.time() - start_time
        audio_duration = len(audio_data) / sample_rate
        rtf = latency / audio_duration if audio_duration > 0 else 0
        
        loi_wer = wer(truth_text, pred_text) if truth_text and pred_text else 1.0
        
        # Đưa vào danh sách kết quả
        ket_qua_test.append({
            "File": audio_file,
            "Language": lang.upper(),
            "Ground Truth": truth_text,
            "AI Predict": pred_text,
            "WER": round(loi_wer, 4),
            "Latency (s)": round(latency, 2),
            "Audio Duration (s)": round(audio_duration, 2),
            "RTF": round(rtf, 3)
        })
        
    except Exception as e:
        print(f"\n[LỖI] Xảy ra tại file {audio_file}: {e}")
        traceback.print_exc()
        print("-" * 30)

# ==========================================
# 4. XUẤT BÁO CÁO TOÀN DIỆN
# ==========================================
df_results = pd.DataFrame(ket_qua_test)

if not df_results.empty:
    df_results.to_excel(EXCEL_OUTPUT, index=False)
    print("\n" + "="*50)
    print(f"✅ Đã hoàn tất! File báo cáo lưu tại:\n {EXCEL_OUTPUT}")

    print("\n--- TỔNG QUAN HIỆU NĂNG 3 MÔ HÌNH ---")
    tong_quan = df_results.groupby('Language').agg(
        Total_Files=('File', 'count'),
        Avg_WER=('WER', 'mean'),
        Avg_Latency=('Latency (s)', 'mean'),
        Avg_RTF=('RTF', 'mean')
    ).reset_index()
    print(tong_quan.to_string(index=False))
else:
    print("\n" + "="*50)
    print("[CẢNH BÁO TỪ HỆ THỐNG]")
    print("Bảng dữ liệu trống! Không có file nào được nhận diện thành công.")