- 主程式:ai_main.py
    - ai_t1.py: 介面格式位置
    - ai_t1.ui: 介面原檔
    - style.qss: 樣式調整區

- 請安裝requirements裡面的套件
- SR emotion需分開運行

- 操作:
    - 確認錄影、錄音設備正常運作，運行ai_main.py
    - 相片功能: 會自動抓取攝影機影像，按下擷取影像會自動儲存至photo_tmp資料夾，並顯示檔名
    - 錄音功能: 按下開始，開始錄音，按下停止，會自動儲存至voice_tmp資料夾
        - 目前測試可以用wav檔成功連API並回傳

- 功能進度:
    - 相片功能 OK
    - 錄音功能 OK
    - 回應訊息 N
    - 接收模型訊息 N
    - 評分 N
    - 資料庫 N