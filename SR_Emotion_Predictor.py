from flask import Flask, request, jsonify
from pydub import AudioSegment
from pydub.utils import which
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import soundfile as sf
import torchaudio
import torch
import os
import numpy as np

# 手動指定 ffmpeg 路徑
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# 載入模型 - 7 類情緒模型
modelName = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(modelName)
featureExtractor = AutoFeatureExtractor.from_pretrained(modelName)

# 音檔轉換 MP3 → WAV - 16kHz 單聲道
def convertMp3ToWav(mp3Path, wavPath):
    audio = AudioSegment.from_mp3(mp3Path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wavPath, format="wav")

# 載入並預測情緒
def predictEmotion(wavPath):
    speech, sampleRate = sf.read(wavPath)
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)  # 若雙聲道轉單聲道
    if sampleRate != 16000:
        speech = torchaudio.functional.resample(torch.tensor(speech), sampleRate, 16000).numpy()
        sampleRate = 16000
    inputs = featureExtractor(speech, sampling_rate=sampleRate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)[0]
    results = sorted(
        [(model.config.id2label[i].capitalize(), float(probabilities[i])) for i in range(len(probabilities))],
        key=lambda x: x[1], reverse=True
    )
    return results

# 初始化 Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predictRoute():
    if "file" not in request.files:
        return jsonify({"error": "請上傳 MP3 或 WAV 音檔"}), 400

    file = request.files["file"]
    filename = file.filename

    tempWav = "Temp.wav"
    tempMp3 = "Temp.mp3"

    if filename.endswith(".mp3"):
        file.save(tempMp3)
        convertMp3ToWav(tempMp3, tempWav)
        os.remove(tempMp3)
    elif filename.endswith(".wav"):
        file.save(tempWav)
    else:
        return jsonify({"error": "僅支援 MP3 或 WAV 格式"}), 400

    prediction = predictEmotion(tempWav)
    os.remove(tempWav)

    return jsonify({label: round(score, 4) for label, score in prediction})

if __name__ == "__main__":
    # 設為 True 啟動 API - False 則跑 CLI 預測流程
    runAsApi = True

    if runAsApi:
        app.run(port=5000, use_reloader=False)
    else:
        inputPath = "Default2.wav"
        wavPath = "Temp.wav"

        if inputPath.endswith(".mp3"):
            convertMp3ToWav(inputPath, wavPath)
        elif inputPath.endswith(".wav"):
            wavPath = inputPath
        else:
            raise ValueError("請提供 MP3 或 WAV 檔案")

        predictionResults = predictEmotion(wavPath)

        print("\n辨識結果：")
        for label, score in predictionResults:
            print(f"{label}: {score:.4f}")

        if inputPath.endswith(".mp3"):
            os.remove(wavPath)
