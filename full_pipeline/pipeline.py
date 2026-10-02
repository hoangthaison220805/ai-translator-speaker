"""
pipeline.py — File CHÍNH của Milestone 6: Voice Translator V1
==============================================================
Pipeline hoàn chỉnh: 🎤 Mic → STT → Dịch → TTS → 🔊 Loa

Cách chạy:
  python pipeline.py

Cách dùng:
  1. Chọn chiều dịch (1-4)
  2. Giữ phím SPACE và nói
  3. Nhả phím SPACE → nghe bản dịch phát ra loa
  4. Nhấn Q để thoát
"""

import os
import sys
import time
import gc

# Đảm bảo Python tìm thấy thư mục milestone6 khi chạy từ bên ngoài
sys.path.insert(0, os.path.dirname(__file__))

from config import TRANSLATION_MODES
from modules.stt_module import STTEngine
from modules.translate_module import TranslationEngine
from modules.tts_module import TTSEngine


def print_banner():
    """In màn hình chào mừng"""
    print("\n" + "=" * 55)
    print("   🌐  VOICE TRANSLATOR V1  —  100% OFFLINE")
    print("=" * 55)
    print("   Hỗ trợ 4 chiều dịch | STT + Dịch + TTS")
    print("=" * 55 + "\n")


def print_menu():
    """In menu chọn chiều dịch"""
    print("\n─" * 28)
    print("  CHỌN CHIỀU DỊCH:")
    print("─" * 28)
    for key, (_, _, label) in TRANSLATION_MODES.items():
        print(f"  [{key}] {label}")
    print("  [Q] Thoát chương trình")
    print("─" * 28)


def run_pipeline(stt, translator, tts, src_lang, tgt_lang):
    """
    Chạy một vòng pipeline đầy đủ:
    Mic → STT → Dịch → TTS → Loa

    Args:
        stt        : STTEngine đã được nạp sẵn
        translator : TranslationEngine đã được nạp sẵn
        tts        : TTSEngine đã được nạp sẵn
        src_lang   : Ngôn ngữ nói vào ("vi", "en", "zh")
        tgt_lang   : Ngôn ngữ nghe ra  ("vi", "en", "zh")
    """
    total_start = time.time()

    # ─── BƯỚC 1: Thu âm & Nhận dạng giọng nói ───────────────
    print("\n[1/3] 🎤 STT — Nhận dạng giọng nói...")
    t0 = time.time()
    recognized_text = stt.listen(src_lang)
    stt_time = time.time() - t0

    # Nếu không nhận ra gì thì dừng vòng này lại
    if not recognized_text:
        print("  ⚠️  Không nhận ra được giọng nói. Vui lòng thử lại.")
        return
    recognized_text = recognized_text.lower().capitalize()
    print(f"  ✅ Nghe được ({stt_time:.2f}s): \"{recognized_text}\"")

    # ─── BƯỚC 2: Dịch văn bản ────────────────────────────────
    print("\n[2/3] 🔄 Dịch thuật...")
    t0 = time.time()
    translated_text = translator.translate(recognized_text, src_lang, tgt_lang)
    translate_time = time.time() - t0

    if not translated_text:
        print("  ⚠️  Dịch thất bại. Vui lòng thử lại.")
        return

    print(f"  ✅ Bản dịch ({translate_time:.2f}s): \"{translated_text}\"")

    # ─── BƯỚC 3: Phát âm bản dịch ────────────────────────────
    print("\n[3/3] 🔊 TTS — Đang phát âm thanh...")
    t0 = time.time()
    tts.speak(translated_text, tgt_lang)
    tts_time = time.time() - t0

    # ─── TỔNG KẾT ĐỘ TRỄ ─────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n  ⏱️  Tổng thời gian: {total_time:.2f}s "
          f"(STT {stt_time:.2f}s + Dịch {translate_time:.2f}s + TTS {tts_time:.2f}s)")

    # Đánh giá có đạt mục tiêu < 5 giây không
    if total_time < 5.0:
        print("  🎯 Đạt mục tiêu < 5 giây!")
    else:
        print("  ⚠️  Vượt quá 5 giây (có thể do câu nói quá dài)")


def main():
    print_banner()

    print("⚙️  Đang khởi động hệ thống...\n")
    print("─" * 55)
    stt        = STTEngine()       # Chỉ import lib, chưa nạp model
    translator = TranslationEngine()
    tts        = TTSEngine()
    print("─" * 55)
    print("✅ Hệ thống sẵn sàng. Chọn chế độ để bắt đầu.\n")

    # ─── VÒNG LẶP CHÍNH ──────────────────────────────────────
    current_mode = None
    src_lang     = None
    tgt_lang     = None

    while True:
        # Nếu chưa chọn chiều dịch thì hiện menu
        if current_mode is None:
            print_menu()
            choice = input("  👉 Lựa chọn: ").strip().lower()

            if choice == "q":
                print("\n👋 Đã thoát. Tạm biệt!\n")
                break

            if choice not in TRANSLATION_MODES:
                print("  ⚠️  Lựa chọn không hợp lệ!\n")
                continue

            src_lang, tgt_lang, label = TRANSLATION_MODES[choice]
            # Unload model cũ nếu đang có
            if current_mode is not None:
                print("\n  🔄 Đang giải phóng model cũ...")
                stt.unload()
                translator.unload()
                tts.unload()
                gc.collect()           # Bắt buộc để Python trả RAM về OS
            # Nạp model mới
            print(f"\n  ⏳ Đang nạp model cho [{label}]...")
            print(f"  (Lần đầu chọn chế độ mất ~10-15 giây)")
            stt.load(src_lang)
            translator.load(src_lang, tgt_lang)
            tts.load(tgt_lang)
            current_mode = choice
            print(f"\n  ✅ Sẵn sàng! Chế độ: {label}")
            print("  (Nhập 'M' để quay lại menu, 'Q' để thoát)\n")

        # ─── Sẵn sàng dịch ──────────────────────────────────
        print(f"\n─── Lần dịch mới ─── (Chế độ: {TRANSLATION_MODES[current_mode][2]})")
        action = input("  Nhấn [Enter] để bắt đầu nói | [M] Menu | [Q] Thoát: ").strip().lower()

        if action == "q":
            print("\n👋 Đã thoát. Tạm biệt!\n")
            break
        elif action == "m":
            print("\n  🔄 Đang giải phóng RAM để quay lại menu...")
            stt.unload()
            translator.unload()
            tts.unload()
            gc.collect() # Ép Python trả RAM ngay lập tức
            current_mode = None  # Quay lại menu
            continue

        # ─── Chạy pipeline ───────────────────────────────────
        try:
            run_pipeline(stt, translator, tts, src_lang, tgt_lang)
        except KeyboardInterrupt:
            print("\n\n  ⏸️  Bị gián đoạn. Nhấn Enter để thử lại hoặc Q để thoát.")
        except Exception as e:
            print(f"\n  ❌ Lỗi không mong muốn: {e}")
            print("  Vui lòng thử lại hoặc khởi động lại chương trình.")


if __name__ == "__main__":
    main()
