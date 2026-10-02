"""
translate_module.py — Module dịch thuật (Opus-MT)
===================================================
Dùng 4 model Opus-MT thay vì NLLB-600M.

Routing:
  vi → en : dùng opus-mt-vi-en
  en → vi : dùng opus-mt-en-vi
  zh → vi : dùng opus-mt-zh-vi
  vi → zh : cầu nối vi→en→zh (2 bước)
  en → zh : dùng opus-mt-en-zh
  zh → en : cầu nối zh→vi→en (2 bước)   ← tuỳ bạn có cần không
"""

import os
import sys
import ctranslate2
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OPUS_MT_MODELS


class TranslationEngine:

    def __init__(self):
        """Chỉ khởi tạo, chưa nạp model nào."""
        self._translators = {}
        self._tokenizers  = {}
        print("[Dịch] ✅ Khởi tạo xong (chưa nạp model).\n")

    # ── Xác định model nào cần nạp ────────────────────────────────
    def _get_required_pairs(self, src_lang, tgt_lang):
        """Trả về list các cặp model cần thiết cho chiều dịch này."""
        direct = f"{src_lang}-{tgt_lang}"
        if direct in OPUS_MT_MODELS:
            return [direct]
        # Cầu nối vi → zh : vi-en + en-zh
        if src_lang == "vi" and tgt_lang == "zh":
            return ["vi-en", "en-zh"]
        # Cầu nối zh → en : zh-vi + vi-en
        if src_lang == "zh" and tgt_lang == "en":
            return ["zh-vi", "vi-en"]
        return []

    # ── Load / Unload ──────────────────────────────────────────────
    def load(self, src_lang, tgt_lang):
        """Nạp các model cần thiết cho cặp ngôn ngữ được chọn."""
        pairs = self._get_required_pairs(src_lang, tgt_lang)
        if not pairs:
            print(f"[Dịch] ❌ Không hỗ trợ: {src_lang} → {tgt_lang}")
            sys.exit(1)

        for pair in pairs:
            path = OPUS_MT_MODELS.get(pair)
            if not path or not os.path.exists(path):
                print(f"[Dịch] ❌ Không tìm thấy model '{pair}' tại: {path}")
                sys.exit(1)

            print(f"[Dịch]   → Nạp opus-mt-{pair}...")
            self._translators[pair] = ctranslate2.Translator(
                path,
                device="cpu",
                compute_type="int8",
                inter_threads=1,    # Phù hợp với Pi
                intra_threads=2,    # Dùng 2 core
            )
            # ✅ Nạp CẢ HAI source.spm và target.spm
            self._tokenizers[pair] = transformers.AutoTokenizer.from_pretrained(
            path, local_files_only=True
            )

        print(f"[Dịch] ✅ Đã nạp xong model dịch.")

    def unload(self):
        """Giải phóng RAM của tất cả model dịch hiện tại."""
        for pair in list(self._translators.keys()):
            del self._translators[pair]
            del self._tokenizers[pair]
        self._translators.clear()
        self._tokenizers.clear()
        print("[Dịch] 🗑️  Đã giải phóng model dịch.")

    # ── Dịch thuật ────────────────────────────────────────────────
    def _translate_pair(self, text, pair):
        
        tokenizer     = self._tokenizers[pair]
        translator = self._translators[pair]   
        source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        results       = translator.translate_batch(
            [source_tokens],
            max_decoding_length=256,
            repetition_penalty=2.0,   # ← giữ lại để an toàn
            beam_size=2,
        )
        target_tokens = results[0].hypotheses[0]
        return tokenizer.decode(
            tokenizer.convert_tokens_to_ids(target_tokens)
        ).strip()

    def translate(self, text, src_lang, tgt_lang):
        """Dịch văn bản. Tự động dùng cầu nối nếu cần."""
        if not text or not text.strip():
            return ""
        if src_lang == tgt_lang:
            return text

        pair = f"{src_lang}-{tgt_lang}"

        # Dịch trực tiếp
        if pair in self._translators:
            return self._translate_pair(text, pair)

        # Cầu nối vi → zh
        if src_lang == "vi" and tgt_lang == "zh":
            return self._translate_pair(
                self._translate_pair(text, "vi-en"), "en-zh"
            )

        # Cầu nối zh → en
        if src_lang == "zh" and tgt_lang == "en":
            return self._translate_pair(
                self._translate_pair(text, "zh-vi"), "vi-en"
            )

        print(f"[Dịch] ❌ Không hỗ trợ: {src_lang} → {tgt_lang}")
        return ""