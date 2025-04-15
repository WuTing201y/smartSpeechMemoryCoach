import whisperx
import os
import sys

# 1. 設定裝置：CPU
device = "cpu"

# 2. 讀取音檔（替換成你的檔名）
base_dir = os.path.dirname(os.path.abspath(__file__))
# 組合完整路徑
audio_path = os.path.join(base_dir, "..", "data", "ICNALE_SM_TWN_N400", "SM_TWN_PTJ1_001_B1_1.mp3")

if not os.path.isfile(audio_path):
    sys.exit("CANNOT FIND AUDIO")

# 3. 載入 WhisperX 語音模型（有"base", "medium", "small" 可以使用）
# 這是模型的等級（越大越準確，但也越慢、越吃記憶體）
#WhisperX 預設會嘗試用 float16 模式（效能比較好，但只支援高階 GPU）
#所以必須改用 float32 模式，它雖然慢一點，但可以在所有機器上執行
audio = whisperx.load_audio(audio_path)
model = whisperx.load_model("small", device, compute_type="float32")

# 4. 做轉錄（得到文字段落與語言）
result = model.transcribe(audio)
print("language:", result["language"])

# 5. 載入對齊模型（為了獲得每個字的時間戳）
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# 6. 對齊文字與音訊（輸出每個詞的時間區間）
aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

# 7. 顯示轉錄結果
final_text = result.get("text") or " ".join(s["text"].strip() for s in result["segments"])
print("\n--- Final Transcript ---\n", final_text, sep="")

if aligned and "word_segments" in aligned:
    print("\n--- Word‑level Timing ---\n")
    for w in aligned["word_segments"]:
        print(f"{w['word']} | {w['start']:.2f}s → {w['end']:.2f}s")

else:
    print("\nCANNOT GET RESULT")