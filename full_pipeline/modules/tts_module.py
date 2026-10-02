"""
tts_module.py — Module chuyển văn bản thành giọng nói (TTS)
=============================================================
Bọc lại code Piper TTS từ Milestone 5 thành hàm gọn gàng.

Cách hoạt động:
  1. TTSEngine.__init__() : Nạp model TTS vào RAM (chỉ làm 1 lần)
  2. TTSEngine.speak()    : Nhận text + ngôn ngữ → tạo file WAV → phát ra loa
"""

import os
import sys
import wave
import winsound
from piper import PiperVoice, voice
import pypinyin
# --- Import cấu hình ---
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TTS_MODELS, TTS_TEMP_WAV


class TTSEngine:
    def __init__(self):
        """Chỉ khởi tạo, chưa nạp model nào."""
        self._voice = None   # Voice đang active
        self._lang  = None   # Ngôn ngữ đang active
        os.makedirs(os.path.dirname(TTS_TEMP_WAV), exist_ok=True)
        print("[TTS] ✅ Khởi tạo xong (chưa nạp model).\n")

    def load(self, lang):
        """Nạp model TTS cho ngôn ngữ được chọn."""
        path = TTS_MODELS.get(lang)
        if not path or not os.path.exists(path):
            print(f"[TTS] ❌ Không tìm thấy model '{lang}' tại: {path}")
            sys.exit(1)
        print(f"[TTS]   → Nạp giọng TTS '{lang}'...")
        self._voice = PiperVoice.load(path)
        self._lang  = lang
        print(f"[TTS] ✅ Đã nạp TTS '{lang}'.")

    def unload(self):
        """Giải phóng RAM của model TTS hiện tại."""
        if self._voice is not None:
            del self._voice
            self._voice = None
            self._lang  = None
            print("[TTS] 🗑️  Đã giải phóng model TTS.")

    def speak(self, text, lang):
        """
        Tổng hợp giọng nói từ văn bản và phát ra loa.

        Args:
            text : Văn bản cần đọc
            lang : Ngôn ngữ của văn bản ("vi", "en", "zh")
        """
        if not text or not text.strip():
            return

        voice = self._voice
        if voice is None:
            print(f"[TTS] ❌ Chưa nạp model! Hãy gọi load() trước.")
            return
        if lang == "zh":
            # Chuyển "你能说得更慢吗?" -> "ni2 neng2 shuo1 de5 geng4 man4 ma5 ?"
            pinyin_list = pypinyin.pinyin(text, style=pypinyin.Style.TONE3)
            # Ghép các âm lại với nhau bằng dấu cách
            text = " ".join([p[0] for p in pinyin_list])
        # Bước 1: Tổng hợp giọng nói và ghi ra file WAV tạm
        with wave.open(TTS_TEMP_WAV, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(voice.config.sample_rate)
            voice.synthesize_wav(text, wav_file)

        # Bước 2: Phát file WAV ra loa (winsound tích hợp sẵn trong Windows)
        winsound.PlaySound(TTS_TEMP_WAV, winsound.SND_FILENAME)
