"""
config.py — File cấu hình trung tâm cho toàn bộ pipeline
=========================================================
Tất cả đường dẫn model, cài đặt ngôn ngữ đều nằm ở đây.
Khi cần thay đổi đường dẫn, chỉ cần sửa file này, không cần động vào code khác.
"""

import os

# ==========================================
# ĐỊA CHỈ GỐC CỦA DỰ ÁN
# ==========================================
BASE_DIR = "D:/Project/AIOT_training_khoahoc/python_code_begin/AI_translator"

# ==========================================
# CẤU HÌNH STT (Speech-to-Text) — Dùng Sherpa-ONNX từ Milestone 2
# ==========================================
STT_MODELS = {
    "vi": os.path.join(BASE_DIR, "sherpa-onnx-zipformer-vi-int8-2025-04-20"),
    "en": os.path.join(BASE_DIR, "sherpa-onnx-zipformer-gigaspeech-2023-12-12"),
    "zh": os.path.join(BASE_DIR, "sherpa-onnx-zipformer-zh-en-2023-11-22"),
}

VAD_MODEL_PATH = os.path.join(BASE_DIR, "silero_vad.onnx")

# ==========================================
# CẤU HÌNH DỊCH THUẬT (Translation) — Dùng NLLB từ Milestone 3
# ==========================================

# TRANSLATION_MODEL_DIR = os.path.join(BASE_DIR, "nllb-4way-finetuned-ct2")
# TOKENIZER_NAME = "facebook/nllb-200-distilled-600M"

# # Mã ngôn ngữ theo chuẩn NLLB-200
# NLLB_LANG_CODES = {
#     "vi": "vie_Latn",
#     "en": "eng_Latn",
#     "zh": "zho_Hans",
# }

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_translate"))
OPUS_MT_MODELS = {
    "vi-en": os.path.join(MODELS_DIR, "opus-mt-vi-en"),
    "en-vi": os.path.join(MODELS_DIR, "opus-mt-en-vi"),
    "zh-vi": os.path.join(MODELS_DIR, "opus-mt-zh-vi"),
    "en-zh": os.path.join(MODELS_DIR, "opus-mt-en-zh"),
}
# ==========================================
# CẤU HÌNH TTS (Text-to-Speech) — Dùng Piper từ Milestone 5
# ==========================================
TTS_MODEL_DIR = os.path.join(BASE_DIR, "milestone5-tts", "models")

TTS_MODELS = {
    "vi": os.path.join(TTS_MODEL_DIR, "vi_VN-vais1000-medium.onnx"),
    "en": os.path.join(TTS_MODEL_DIR, "en_US-lessac-low.onnx"),
    "zh": os.path.join(TTS_MODEL_DIR, "zh_CN-huayan-x_low.onnx"),
}

# File WAV tạm thời để TTS ghi ra rồi phát
TTS_TEMP_WAV = os.path.join(BASE_DIR, "milestone6", "temp_tts.wav")

# ==========================================
# CẤU HÌNH 4 CHIỀU DỊCH
# ==========================================
# Mỗi chiều: (ngôn_ngữ_nói_vào, ngôn_ngữ_nghe_ra, nhãn hiển thị)
TRANSLATION_MODES = {
    "1": ("vi", "en", "🇻🇳 Tiếng Việt  →  🇺🇸 Tiếng Anh"),
    "2": ("en", "vi", "🇺🇸 Tiếng Anh   →  🇻🇳 Tiếng Việt"),
    "3": ("zh", "vi", "🇨🇳 Tiếng Trung →  🇻🇳 Tiếng Việt"),
    "4": ("vi", "zh", "🇻🇳 Tiếng Việt  →  🇨🇳 Tiếng Trung"),
}
