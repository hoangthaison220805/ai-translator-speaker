import os
import time
import queue
import threading
import wave
import winsound
import re
from piper import PiperVoice

class StreamingTTS:
    def __init__(self, lang="vi"):
        print("[Hệ thống] Đang khởi động Streaming TTS...")
        self.temp_wav = "stream_temp.wav"
        
        # Hàng đợi chứa các câu đã sẵn sàng để phát âm
        self.sentence_queue = queue.Queue()
        
        # Nạp model
        model_files = {
            "vi": "vi_VN-vais1000-medium.onnx",
            "en": "en_US-lessac-low.onnx",
            "zh": "zh_CN-huayan-x_low.onnx"
        }
        model_path = os.path.join("models", model_files[lang])
        if not os.path.exists(model_path):
            raise Exception(f"Không tìm thấy file {model_path}. Hãy chạy download_models.py trước.")
        
        self.voice = PiperVoice.load(model_path)
        
        # Bắt đầu luồng công nhân chuyên phát âm thanh
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.worker_thread.start()

    def _tts_worker(self):
        """Luồng chạy ngầm: Chờ lấy từng câu ra để đọc"""
        while self.is_running:
            try:
                # Chờ lấy câu từ hàng đợi (thời gian chờ 1 giây)
                sentence = self.sentence_queue.get(timeout=1.0)
                
                if sentence.strip():
                    print(f"\n🔊 [Đang phát âm]: {sentence}")
                    
                    # Tạo file wav và phát
                    with wave.open(self.temp_wav, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.voice.config.sample_rate)
                        self.voice.synthesize_wav(sentence, wav_file)
                    
                    winsound.PlaySound(self.temp_wav, winsound.SND_FILENAME)
                    
                self.sentence_queue.task_done()
            except queue.Empty:
                continue

    def add_text_stream(self, text_chunk):
        """Hàm này nhận từng cụm từ do AI dịch ra và tách thành câu hoàn chỉnh"""
        # (Trong thực tế, bạn sẽ nối các cụm từ lại và dùng regex để tách câu)
        # Ở ví dụ này, để đơn giản, ta coi text_chunk đưa vào đã là 1 câu.
        self.sentence_queue.put(text_chunk)

    def stop(self):
        self.is_running = False
        self.worker_thread.join()
        if os.path.exists(self.temp_wav):
            try:
                os.remove(self.temp_wav)
            except:
                pass

def simulate_translator():
    """Giả lập một hệ thống AI dịch từng câu mất thời gian"""
    tts = StreamingTTS(lang="vi")
    
    # Giả sử AI dịch ra 3 câu, mỗi câu dịch mất 2 giây
    translated_sentences = [
        "Xin chào, tôi là trợ lý phiên dịch của bạn.",
        "Tôi có thể vừa dịch vừa phát âm thanh cùng một lúc.",
        "Như vậy bạn sẽ không phải đợi dịch xong toàn bộ mới được nghe."
    ]
    
    print("\n--- BẮT ĐẦU GIẢ LẬP DỊCH THUẬT ---")
    for i, sentence in enumerate(translated_sentences):
        print(f"\n[AI Dịch] Đang dịch câu {i+1}...")
        time.sleep(2) # Giả lập độ trễ khi dịch (2 giây)
        
        print(f"[AI Dịch] Dịch xong câu {i+1}: '{sentence}' -> Gửi qua TTS phát luôn!")
        tts.add_text_stream(sentence)
        
    # Đợi cho phát âm xong hết các câu trong hàng đợi
    tts.sentence_queue.join()
    tts.stop()
    print("\n--- HOÀN THÀNH ---")

if __name__ == "__main__":
    simulate_translator()
