- 主程式:ai_main.py
    - ai_t1.py: 介面格式位置
    - ai_t1.ui: 介面原檔
    - Emotion_Predictor_backends.py: 影像+語音辨識
    - style.qss: 樣式調整區(暫時棄用)

- 請安裝requirements裡面的套件
- Emotion_Predictor_backends.py、ai_main.py、LM studio API需分開運行

- 操作:
    - 確認錄影、錄音設備正常運作，LM studio載入模型並開啟API，分別運行Emotion_Predictor_backends.py、ai_main.py
    - 相片功能: 會自動抓取攝影機影像，按下擷取影像會自動儲存至photo_tmp資料夾，並顯示檔名
    - 錄音功能: 按下開始，開始錄音，按下停止，會自動儲存至voice_tmp資料夾
    - 模型回應: 在底下框框輸入訊息，按傳送即可接收模型回應

- 功能進度:
    - 相片功能 OK
    - 錄音功能 OK
    - 回應訊息 OK
    - 接收模型訊息 OK
    - 評分 N
    - 資料庫 N