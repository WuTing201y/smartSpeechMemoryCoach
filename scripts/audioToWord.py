import whisperx

# 1. 設定裝置：CPU
device = "cpu"

# 2. 讀取音檔（替換成你的檔名）
audio_path =  "../data/ICNALE_SM_TWN_N400/SM_TWN_PTJ1_001_B1_1.mp3"
audio = whisperx.load_audio(audio_path)

# 3. 載入 WhisperX 語音模型（有"base", "medium", "small" 可以使用）
# 這是模型的等級（越大越準確，但也越慢、越吃記憶體）
model = whisperx.load_model("medium", device)

# 4. 做轉錄（得到文字段落與語言）
result = model.transcribe(audio)

# 5. 載入對齊模型（為了獲得每個字的時間戳）
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# 6. 對齊文字與音訊（輸出每個詞的時間區間）
aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device)

# 7. 顯示轉錄結果
print("\n--- Final Transcript ---\n")
print(aligned_result["text"])

# 8. 顯示每個詞的時間戳（for delivery 特徵）
print("\n--- Word-level timing ---\n")
for word in aligned_result["word_segments"]:
    print(f"{word['word']} | start: {word['start']:.2f}s | end: {word['end']:.2f}s")
