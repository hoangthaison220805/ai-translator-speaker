"""
stt_module.py — Module nhận dạng giọng nói (STT)
=================================================
Bọc lại code Sherpa-ONNX từ Milestone 2 thành các hàm gọn gàng.

Cách hoạt động:
  1. STTEngine.__init__() : Nạp model vi/en/zh vào RAM (chỉ làm 1 lần)
  2. STTEngine.listen()   : Bắt đầu thu âm → giữ SPACE để nói → nhả SPACE để dừng
  3. Trả về chuỗi text đã nhận dạng được
"""

import os
import sys
import time
import re
import numpy as np
import sounddevice as sd
import keyboard
import sherpa_onnx

# --- Import cấu hình từ file config.py ---
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import STT_MODELS, VAD_MODEL_PATH


class STTEngine:
    # def __init__(self):
    #     """Nạp tất cả model STT vào RAM một lần duy nhất khi khởi động."""
    #     print("[STT] Đang nạp model nhận dạng giọng nói...")
    #     self.models = {}
    #     for lang, model_path in STT_MODELS.items():
    #         self.models[lang] = self._load_model(model_path, lang)
    #     print("[STT] ✅ Nạp xong tất cả model STT.\n")
    def __init__(self):
        """Chỉ khởi tạo, chưa nạp model nào."""
        self._model = None   # Model đang active
        self._lang  = None   # Ngôn ngữ đang active
        print("[STT] ✅ Khởi tạo xong (chưa nạp model).\n")

    def load(self, lang):
        """Nạp model cho ngôn ngữ được chọn."""
        print(f"[STT]   → Nạp model STT '{lang}'...")
        model_path = STT_MODELS.get(lang)
        if not model_path or not os.path.exists(model_path):
            print(f"[STT] ❌ Không tìm thấy model '{lang}' tại: {model_path}")
            sys.exit(1)
        self._model = self._load_model(model_path, lang)
        self._lang  = lang
        print(f"[STT] ✅ Đã nạp STT '{lang}'.")

    def unload(self):
        """Giải phóng RAM của model hiện tại."""
        if self._model is not None:
            del self._model
            self._model = None
            self._lang  = None
            print("[STT] 🗑️  Đã giải phóng model STT.")
    
    def _load_model(self, model_path, lang_name):
        """Hàm nội bộ: tìm và nạp file .onnx của Sherpa-ONNX"""
        if not os.path.exists(model_path):
            print(f"[STT] ⚠️  Cảnh báo: Không tìm thấy model '{lang_name}' tại {model_path}")
            return None

        # Tự động tìm 3 file encoder/decoder/joiner trong thư mục model
        encoder = next((f for f in os.listdir(model_path) if f.startswith("encoder") and f.endswith(".onnx")), None)
        decoder = next((f for f in os.listdir(model_path) if f.startswith("decoder") and f.endswith(".onnx")), None)
        joiner  = next((f for f in os.listdir(model_path) if f.startswith("joiner")  and f.endswith(".onnx")), None)

        if not all([encoder, decoder, joiner]):
            print(f"[STT] ❌ Lỗi: Thiếu file .onnx trong {model_path}")
            return None

        print(f"[STT]   → Nạp model '{lang_name}'...")
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=os.path.join(model_path, "tokens.txt"),
            encoder=os.path.join(model_path, encoder),
            decoder=os.path.join(model_path, decoder),
            joiner=os.path.join(model_path, joiner),
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )

    def _clean_text(self, text, lang):
        """Chuẩn hóa text sau khi nhận dạng (loại bỏ ký tự lạ)"""
        text = str(text).strip()
        if lang == "en":
            text = re.sub(r"[^a-zA-Z0-9\s',.!?]", "", text)
        elif lang == "zh":
            text = re.sub(r"[^\w\s，。！？]", "", text)
        elif lang == "vi":
            text = re.sub(r"[^\w\s,.!?àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", "", text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def listen(self, lang, push_to_talk_key="space"):
        """
        Thu âm từ Micro và trả về chuỗi text.
        
        Cách dùng: Giữ phím SPACE để nói, nhả ra để dừng.
        
        Args:
            lang: Ngôn ngữ cần nhận dạng ("vi", "en", "zh")
            push_to_talk_key: Phím giữ để thu âm (mặc định: "space")
        
        Returns:
            str: Văn bản đã nhận dạng, hoặc "" nếu không nghe thấy gì
        """
        model = self._model
        if model is None:
            print(f"[STT] ❌ Chưa nạp model! Hãy gọi load() trước.")
            return ""
        # if model is None:
        #     print(f"[STT] ❌ Model '{lang}' chưa được nạp!")
        #     return ""

        RATE = 16000
        frames = []

        def audio_callback(indata, frames_count, time_info, status):
            frames.append(indata.copy())

        # --- Chờ người dùng bấm phím ---
        print(f"\n  👉 Giữ phím [SPACE] để nói, nhả ra để dừng...")
        while not keyboard.is_pressed(push_to_talk_key):
            time.sleep(0.01)

        print("  🔴 Đang lắng nghe... (Nhả phím SPACE để kết thúc)")

        # --- Thu âm trong khi giữ phím ---
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=audio_callback):
            while keyboard.is_pressed(push_to_talk_key):
                time.sleep(0.05)

        print("  ⏹️  Đã dừng thu âm. Đang xử lý...")

        if not frames:
            return ""

        # --- Chốt kết quả cuối cùng ---
        final_audio = np.concatenate(frames, axis=0).flatten()
        # Thêm 0.5 giây im lặng ở cuối để AI xử lý từ cuối câu
        silence = np.zeros(int(RATE * 0.5), dtype=np.float32)
        final_audio = np.concatenate((final_audio, silence))

        stream = model.create_stream()
        stream.accept_waveform(RATE, final_audio)
        model.decode_stream(stream)

        raw_text = stream.result.text
        return self._clean_text(raw_text, lang)
