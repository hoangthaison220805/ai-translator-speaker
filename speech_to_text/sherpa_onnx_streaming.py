import os
import sys
import time
import numpy as np
import sounddevice as sd
import keyboard
import sherpa_onnx
import re
import zhconv
from number_parser import parse

# ==========================================
# 0. CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
DIR_MODEL_VI = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/sherpa-onnx-zipformer-vi-int8-2025-04-20"
DIR_MODEL_EN = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/sherpa-onnx-zipformer-gigaspeech-2023-12-12"
DIR_MODEL_ZH = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator/sherpa-onnx-zipformer-zh-en-2023-11-22"
VAD_MODEL = "silero_vad.onnx" # Đảm bảo file này nằm cùng thư mục với script chạy

# ==========================================
# 1. BỘ CHUẨN HÓA VĂN BẢN (ROUTER)
# ==========================================
def clean_text_router(text, lang):
    text = str(text).lower()
    if lang == 'en':
        text = parse(text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
    elif lang == 'zh':
        text = zhconv.convert(text, 'zh-hans')
        text = re.sub(r'[^\w]', '', text)
    elif lang == 'vi':
        text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH & VAD (PRE-LOADING)
# ==========================================
def load_offline_transducer(model_path, lang_name):
    print(f"[*] Đang nạp {lang_name}...")
    encoder_file = next((f for f in os.listdir(model_path) if f.startswith("encoder") and f.endswith(".onnx")), None)
    decoder_file = next((f for f in os.listdir(model_path) if f.startswith("decoder") and f.endswith(".onnx")), None)
    joiner_file = next((f for f in os.listdir(model_path) if f.startswith("joiner") and f.endswith(".onnx")), None)
            
    if not all([encoder_file, decoder_file, joiner_file]):
        print(f"❌ Lỗi: Thiếu file .onnx trong {model_path}")
        sys.exit(1)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=f"{model_path}/tokens.txt",
        encoder=os.path.join(model_path, encoder_file),
        decoder=os.path.join(model_path, decoder_file),
        joiner=os.path.join(model_path, joiner_file),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80
    )

print("="*50)
print("🚀 HỆ THỐNG PHIÊN DỊCH REAL-TIME (SILERO VAD)")
print("="*50)
model_vi = load_offline_transducer(DIR_MODEL_VI, "Tiếng Việt")
model_en = load_offline_transducer(DIR_MODEL_EN, "Tiếng Anh")
model_zh = load_offline_transducer(DIR_MODEL_ZH, "Tiếng Trung")

# Khởi tạo Silero VAD
if not os.path.exists(VAD_MODEL):
    print(f"❌ Lỗi: Không tìm thấy {VAD_MODEL}. Hãy wget tải về!")
    sys.exit(1)
    
vad_config = sherpa_onnx.VadModelConfig(
    silero_vad=sherpa_onnx.SileroVadModelConfig(
        model=VAD_MODEL,
        min_silence_duration=0.5  # Chuyển tham số này vào trong Silero!
    ),
    sample_rate=16000
)
print("✅ Nạp thành công Silero VAD!")
print("✅ Hệ thống đã sẵn sàng!\n")

# ==========================================
# 3. THU ÂM & REAL-TIME STREAMING (MÚA CHỮ LIÊN TỤC)
# ==========================================
def record_vad_and_stream(active_model, lang_code, key='space'):
    RATE = 16000
    frames = []

    def audio_callback(indata, frames_count, time_info, status):
        # Hứng dữ liệu liên tục từ micro
        frames.append(indata.copy())
        
    print(f"\n👉 Hãy bấm và GIỮ phím '{key.upper()}' để nói...")
    while not keyboard.is_pressed(key):
        time.sleep(0.01)
        
    print("🔴 ĐANG NGHE... (Nhả phím ra để kết thúc)")
    last_text = ""
    
    with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', callback=audio_callback):
        while keyboard.is_pressed(key):
            if len(frames) > 0:
                # 1. Gom toàn bộ âm thanh TỪ LÚC BẤM ĐẾN HIỆN TẠI
                current_audio = np.concatenate(frames, axis=0).flatten()
                
                # 2. Đợi có ít nhất 0.2s âm thanh mới bắt đầu dịch để tránh đoán mò
                if len(current_audio) > RATE * 0.2:
                    # 3. Ép AI dịch ngay lập tức luồng âm thanh đang lớn dần
                    stream = active_model.create_stream()
                    stream.accept_waveform(RATE, current_audio)
                    active_model.decode_stream(stream)
                    
                    text = clean_text_router(stream.result.text, lang_code)
                    
                    if text != last_text and text.strip() != "":
                        display_text = text.upper()
                        
                        # [TRỊ LỖI TRÀN VIỀN]: Nếu câu dài hơn 80 ký tự, chỉ lấy 80 chữ cuối cùng
                        if len(display_text) > 80:
                            display_text = "..." + display_text[-77:]
                            
                        # Dùng mã \033[K để quét sạch rác ở cuối dòng một cách triệt để nhất
                        print(f"\r   🗣️ Real-time: {display_text}\033[K", end="", flush=True)
                        last_text = text
            # Tốc độ quét 150ms/lần giúp chữ múa liên tục và mượt mà
            time.sleep(0.15) 
            
    print("\n\n⏹️ Đang chốt kết quả cuối cùng...")
    
    if len(frames) == 0:
        return None
        
    # [QUAN TRỌNG] Chèn thêm 0.5s khoảng lặng ở đuôi để AI xả nốt từ cuối cùng
    final_audio = np.concatenate(frames, axis=0).flatten()
    tail_silence = np.zeros(int(RATE * 0.5), dtype=np.float32)
    final_audio = np.concatenate((final_audio, tail_silence))
    
    # Dịch lần cuối để lấy kết quả hoàn hảo
    stream = active_model.create_stream()
    stream.accept_waveform(RATE, final_audio)
    active_model.decode_stream(stream)
    
    return clean_text_router(stream.result.text, lang_code)

# ==========================================
# 4. GIAO DIỆN ĐIỀU KHIỂN
# ==========================================
while True:
    print("="*40)
    print("CHỌN NGÔN NGỮ ĐỂ DỊCH:")
    print(" [1] Tiếng Việt")
    print(" [2] Tiếng Anh")
    print(" [3] Tiếng Trung")
    print(" [Q] Thoát")
    print("="*40)
    
    choice = input("Lựa chọn: ").strip().lower()
    if choice == 'q': break
    
    lang_map = {'1': ('vi', model_vi), '2': ('en', model_en), '3': ('zh', model_zh)}
    if choice not in lang_map:
        print("⚠️ Không hợp lệ!")
        continue
        
    lang_code, active_model = lang_map[choice]
    
    # Bấm giữ phím SPACE để thu âm
    final_result = record_vad_and_stream(active_model, lang_code, key='space')
    
    if final_result:
        print(f"✅ KẾT QUẢ CHUẨN: {final_result.upper()}\n")
    else:
        print("⚠️ VAD báo: Không có tiếng người được phát hiện.\n")