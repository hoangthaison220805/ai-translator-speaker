# AI-translator
# 🌐 Nbot - Offline Edge AI Voice Translator

Dự án máy phiên dịch giọng nói toàn trình (End-to-End Voice Translator) chạy **100% Offline** trên các thiết bị nhúng (Edge Devices) như Raspberry Pi 4. Hệ thống hỗ trợ dịch thuật 4 chiều giữa **Tiếng Việt - Tiếng Anh - Tiếng Trung** với độ trễ (Latency) toàn trình dưới 5 giây.

## ✨ Điểm nổi bật (Features)
- **100% Offline & Private:** Không sử dụng bất kỳ API đám mây nào (No Google/Azure). Hoàn toàn bảo mật dữ liệu.
- **Tối ưu hóa tài nguyên (Edge-Optimized):** 
  - Áp dụng kỹ thuật lượng tử hóa **INT8** (Quantization) qua CTranslate2.
  - Sử dụng kiến trúc dịch thuật "Bắc cầu" (Bridge Translation: Vi ➔ En ➔ Zh) để giải quyết bài toán thiếu hụt dữ liệu song ngữ trực tiếp.
  - Quản lý bộ nhớ nghiêm ngặt (Garbage Collection, Lazy-loading), giới hạn Peak RAM luôn **< 600MB**.
- **Kiến trúc Hướng đối tượng (OOP):** Dễ dàng mở rộng, thay thế module hoặc tích hợp vào các dự án Robotics lớn hơn.
- **Benchmarking Tool tích hợp:** Tự động đo lường điểm BLEU Score và Latency, xuất báo cáo trực tiếp ra file Excel.

## 🧠 Các Mô hình AI sử dụng (Model Zoo)

Hệ thống kết hợp 3 lớp mô hình (STT ➔ MT ➔ TTS) độc lập. **Lưu ý:** Bạn cần tải các mô hình dưới đây và đặt đúng đường dẫn đã cấu hình trong file `config.py`.

### 1. Nhận dạng giọng nói (STT) - `Sherpa-ONNX`
Sử dụng kiến trúc Zipformer INT8 để chạy mượt mà trên CPU ARM:
- **Tiếng Việt:** `sherpa-onnx-zipformer-vi-int8-2025-04-20`
- **Tiếng Anh:** `sherpa-onnx-zipformer-gigaspeech-2023-12-12`
- **Tiếng Trung:** `sherpa-onnx-zipformer-zh-en-2023-11-22`

### 2. Dịch thuật (Machine Translation) - `OPUS-MT (Helsinki-NLP)`
Các mô hình đã được ép cân (Pruned & Quantized) xuống định dạng CTranslate2 INT8 (~75MB/model). Bao gồm 4 thư mục:
- `opus-mt-vi-en`
- `opus-mt-en-vi`
- `opus-mt-zh-vi`
- `opus-mt-en-zh` (Đóng vai trò trạm trung chuyển cho chiều Vi-Zh).
*(Bắt buộc phải có file `model.bin` và các file `*.spm` SentencePiece đi kèm).*

### 3. Tổng hợp giọng nói (TTS) - `Piper TTS`
Sử dụng các mô hình `.onnx` siêu nhẹ giữ nguyên được thanh điệu tự nhiên:
- **Tiếng Việt:** `vi_VN-vais1000-medium.onnx`
- **Tiếng Anh:** `en_US-lessac-low.onnx`
- **Tiếng Trung:** `zh_CN-huayan-x_low.onnx`
