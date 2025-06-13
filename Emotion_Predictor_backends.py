from flask import Flask, request, jsonify
from pydub import AudioSegment
from pydub.utils import which
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import soundfile as sf
import torchaudio
import torch
import os
import numpy as np
from deepface import DeepFace
import cv2
from flask_cors import CORS

# 是否使用 TTS 模型
is_tts = False  
if is_tts:
    from TTS.api import TTS
    import sounddevice as sd
    tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False,gpu=True)

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
CORS(app)  # 啟用 CORS 支援

#聲音辨識 API
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
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

#影像辨識 API
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        # 取得前端傳來的圖像
        file = request.files['image']
        if not file:
            return jsonify({'error': '沒有收到圖片'}), 400

        # 將圖像轉換為 numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 分析圖像中的情緒（不強制一定要偵測到臉）
        result = DeepFace.analyze(
            img_path=img,
            actions=['emotion'],
            enforce_detection=False
        )

        # 將 numpy 中的 float32 轉成 Python float，避免 JSON 出錯
        def convert(o):
            if isinstance(o, np.float32) or isinstance(o, np.float64):
                return float(o)
            if isinstance(o, dict):
                return {k: convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [convert(i) for i in o]
            return o

        result = convert(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'DeepFace analyze error: {str(e)}'}), 500

# TTS 語音合成 API
@app.route('/speak', methods=['POST'])
def speak():
    if is_tts:
        text = request.form.get('text', '')
        if text.strip():
            # 產生語音波形 (22050Hz)
            wav = tts.tts(text)
            sd.play(wav, samplerate=22050)
            sd.wait()
            return f"已播放：{text}"
        return "請輸入文字！"
    else:
        return "TTS 未啟用！", 503

if __name__ == "__main__":
    # 設為 True 啟動 API - False 則跑 CLI 預測流程
    runAsApi = True

    if runAsApi:
        app.run(port=5000, use_reloader=False)
    # else:
    #     inputPath = "Default2.wav"
    #     wavPath = "Temp.wav"

    #     if inputPath.endswith(".mp3"):
    #         convertMp3ToWav(inputPath, wavPath)
    #     elif inputPath.endswith(".wav"):
    #         wavPath = inputPath
    #     else:
    #         raise ValueError("請提供 MP3 或 WAV 檔案")

    #     predictionResults = predictEmotion(wavPath)

    #     print("\n辨識結果：")
    #     for label, score in predictionResults:
    #         print(f"{label}: {score:.4f}")

    #     if inputPath.endswith(".mp3"):
    #         os.remove(wavPath)
