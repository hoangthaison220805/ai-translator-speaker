import os
import time
import json
import wave
from piper import PiperVoice

class TTSBenchmark:
    def __init__(self):
        self.models_dir = "models"
        self.model_files = {
            "vi": "vi_VN-vais1000-medium.onnx",
            "en": "en_US-lessac-low.onnx",
            "zh": "zh_CN-huayan-x_low.onnx"
        }
        self.test_sentences = {
            "vi": "Xin chào, tôi là trợ lý thông minh. Chúc bạn một ngày tốt lành.",
            "en": "Hello, I am a smart assistant. Have a great day.",
            "zh": "你好，我是智能助手。祝你今天过得愉快。"
        }
    
    def get_model_size(self, lang):
        path = os.path.join(self.models_dir, self.model_files[lang])
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024) # MB
        return 0

    def run_benchmark(self):
        results = {}
        for lang, text in self.test_sentences.items():
            print(f"\n--- Đang benchmark tiếng {lang} ---")
            model_path = os.path.join(self.models_dir, self.model_files[lang])
            
            if not os.path.exists(model_path):
                print(f"Bỏ qua (không tìm thấy model {lang})")
                continue
            
            # 1. Đo thời gian nạp model (Cold start latency)
            start_load = time.time()
            voice = PiperVoice.load(model_path)
            load_time = time.time() - start_load
            print(f"Thời gian nạp model: {load_time:.4f}s")
            
            # 2. Đo thời gian tổng hợp giọng nói
            output_file = f"output/benchmark_{lang}.wav"
            os.makedirs("output", exist_ok=True)
            
            start_tts = time.time()
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(voice.config.sample_rate)
                voice.synthesize(text, wav_file)
            tts_time = time.time() - start_tts
            
            # 3. Tính RTF (Real-time Factor)
            with wave.open(output_file, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_duration = frames / float(rate)
                
            if audio_duration == 0:
                print(f"⚠️ Lỗi nghiêm trọng: Model {lang} sinh ra âm thanh rỗng (0 giây). Vui lòng kiểm tra lại file .json!")
                rtf = 0
            else:
                rtf = tts_time / audio_duration
            
            print(f"Độ dài audio: {audio_duration:.2f}s")
            print(f"Thời gian xử lý: {tts_time:.4f}s")
            print(f"RTF (Real-time Factor): {rtf:.4f} (Nhỏ hơn 1 là chạy nhanh hơn thời gian thực)")
            
            results[lang] = {
                "model_size_mb": round(self.get_model_size(lang), 2),
                "load_time_sec": round(load_time, 4),
                "audio_duration_sec": round(audio_duration, 4),
                "tts_processing_time_sec": round(tts_time, 4),
                "real_time_factor": round(rtf, 4)
            }
            
        with open("benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print("\nĐã lưu kết quả vào benchmark_results.json")

if __name__ == "__main__":
    benchmark = TTSBenchmark()
    benchmark.run_benchmark()
