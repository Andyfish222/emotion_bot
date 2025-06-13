from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtCore import pyqtSignal, QObject
import sys, cv2, threading, datetime, os, wave
from ai_t1 import Ui_MainWindow 
import logging,threading
import pyaudio
import requests,re 
from qt_material import apply_stylesheet
from openai import OpenAI
import sqlite3

#define
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
## 遠端運行
# voice_url = "http://emoback.andyfish2.trade/predict_voice"
# image_url = "http://emoback.andyfish2.trade/predict_image"
# client = OpenAI(base_url="http://emoback.andyfish2.trade/v1", api_key="lm-studio")
## 本地運行
voice_url = "http://localhost:5000/predict_voice"
image_url = "http://localhost:5000/predict_image"
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 中介物件：用來從非Qt執行緒發射訊號回主線程
class StreamSignalEmitter(QObject):
    new_text = pyqtSignal(str)
    del_new_msg = pyqtSignal()

#資料庫相關
def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("已初始化資料庫 chat_history.db")

def save_message(user_id: str, role: str, content: str):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content))
    conn.commit()
    conn.close()
    logging.info(f"已儲存訊息: {role} - {content}")

def get_recent_messages(user_id: str, limit: int = 10) -> list:
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''
        SELECT role, content FROM messages
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()

    final_msg = [{"role": role, "content": content} for role, content in reversed(rows)]
    # 反轉回時間順序
    return final_msg

class MyWidget(QtWidgets.QMainWindow):
    def __init__(self):
        # 建立初始化 UI 類別實體、初始化變數
        super().__init__()
        self.setUpdatesEnabled(True)
        self.ui = Ui_MainWindow()       
        self.ui.setupUi(self)

        #初始化資料
        self.reply_msg = ""               # 初始化模型回應
        self.new_model_message = ""       # 初始化模型新訊息
        init_db()                         # 初始化資料庫
        self.memory_limit = 50            # 設定記憶限制
        self.user_id = "5"                # 設定使用者 ID，預設為 1
        self.think_state = 0              # 設定思考狀態，0 為未思考，1 為正在思考
        self.print_think = 0
        self.mode = "default"  # 設定模式，預設為 "default"
        self.gamemode = ""

        #建立訊號發射器(LLM回應)
        self.signals = StreamSignalEmitter()
        self.signals.new_text.connect(self.update_browser)
        self.signals.del_new_msg.connect(self.del_nmsg) 

        #樣式表
        apply_stylesheet(app, theme='dark_amber.xml')
        with open("style.qss", "r", encoding="utf-8") as f:
            self.setStyleSheet(f.read())
        self.ui.model_response.setLineWrapMode(self.ui.model_response.LineWrapMode.NoWrap)  # 設定不自動換行

        #影像相關
        self.ocv = True                 # 啟用 OpenCV
        self.photo= False               # 拍照狀態

        #音訊相關
        self.chunk = 1024                     # 記錄聲音的樣本區塊大小
        self.sample_format = pyaudio.paInt16  # 樣本格式，可使用 paFloat32、paInt32、paInt24、paInt16、paInt8、paUInt8、paCustomFormat
        self.channels = 1                     # 聲道數量
        self.fs = 44100                       # 取樣頻率，常見值為 44100 ( CD )、48000 ( DVD )、22050、24000、12000 和 11025。
        # self.seconds = 5                    # 錄音秒數
        self.run = False
        self.is_tts=False                     # 是否啟用 TTS 語音合成

        #按鈕動作區
        self.ui.take_pic.clicked.connect(self.take_pic)
        self.ui.start_rec.clicked.connect(self.start_recording)
        self.ui.stop_rec.clicked.connect(self.stop_recording)
        self.ui.send_msg.clicked.connect(self.send_msg)
        self.ui.send_score.clicked.connect(self.feedback)
        self.ui.msg_input.returnPressed.connect(self.send_msg)

    def closeEvent(self):
        self.ocv = False

    #暫存檔名:當下時間
    def rename(self):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    #模型回應區------------------------------------------
    def get_llm_reply(self,user_input,image_result,voice_result,mode="default"):
        #依情境選取系統prompt
        dict_system_msg = {
            "default": '''
                         你是一位溫暖且善解人意的情緒智能助手，能夠理解和回應用戶的情感需求，並總是以繁體中文回應。
                         不要輸出括號內的說明（例如：(我會說...)(模擬思考...) 等），請直接用自然語言對話。
                         用戶輸入可能包含表情狀態、聲音狀態和用戶回應，你的目標是用好朋友的口吻來使用戶開心。
                         ''',
            "happy": '''
                         你是一位超級開朗的朋友，遇到開心的用戶時，請給予熱烈祝福，並總是以繁體中文回應。
                         不要輸出括號內的說明（例如：(我會說...)(模擬思考...) 等），請直接用自然語言對話。
                         用戶輸入可能包含表情狀態、聲音狀態和用戶回應，你的目標是用好朋友的口吻來使用戶更開心。
                         ''',
            "sad": '''
                         你是一位溫柔的朋友，當用戶難過時，請給溫暖的安慰和一個建議，並總是以繁體中文回應。
                         不要輸出括號內的說明（例如：(我會說...)(模擬思考...) 等），請直接用自然語言對話。
                         用戶輸入可能包含表情狀態、聲音狀態和用戶回應，你的目標是用好朋友的口吻來安慰用戶或使用戶變開心。
                         ''',
            "game": {
                "guess_number": '''
                                用戶現在想要玩猜數字，請先在心中想一個數字，但先不要說出來。範例如下:

                                我心中想了一個 1 到 10 的數字，你猜猜看是幾？
                                （假設用戶輸入：5）
                                (助手:再大一點喔！--因為心中想的是 7，5<7)
                                （假設用戶輸入：8）
                                (助手:再小一點喔！--因為心中想的是 7，8>7)

                                （假設用戶輸入：7）
                                (助手:你猜的是 7，哇！答對了 🎉
                                要不要再玩一次？這次我提高難度喔！)
                                 ''',
                "guess_word": '''
                                用戶現在想要玩猜謎語，請先在心中想一個答案，但先不要說出來。範例如下:

                                我們來玩猜謎語吧！我會給你一個謎語，請你猜猜答案是什麼。
                                謎語：圓圓白白，在冰箱裡，打開殼就能吃，常常早餐吃它。你猜是什麼呢？（提示：兩個字）

                                （用戶輸入：雞蛋)
                                (助手:正確答案!你真棒）
                                 ''',
                "pick_card": '''
                                用戶現在想要玩抽卡。範例如下:

                                 我們來抽一張「心情關懷卡」吧，看看今天的指引是什麼 🎴
                                🎴 你抽到的是：「深呼吸卡」

                                內容：停下來，做三次深呼吸。吸氣，停留，吐氣。
                                吸 —— 停 —— 吐。重複三次。

                                💬 做完後你感覺如何呢？
                                 '''
            }
        }
        if self.mode == "game":
            if self.gamemode == "guess_word": prompt_text = dict_system_msg["game"]["guess_word"]
            if self.gamemode == "guess_number": prompt_text = dict_system_msg["game"]["guess_number"]
            if self.gamemode == "pick_card": prompt_text = dict_system_msg["game"]["pick_card"]
        elif image_result == "sad" or voice_result == "sad":
            prompt_text = dict_system_msg["sad"]
        elif image_result == "happy" or voice_result == "happy":
            prompt_text = dict_system_msg["happy"]
        else:
            prompt_text = dict_system_msg["default"]
        sys_msgs = [{"role": "system", "content": prompt_text}]  # 預設系統訊息

        history_msgs = get_recent_messages(user_id=self.user_id, limit=self.memory_limit)  # 獲取最近的對話歷史
        if history_msgs == []: history_msgs = [{"role": "system", "content": "沒有歷史對話"}]  # 如果沒有歷史對話，則使用預設訊息
        if len(history_msgs)==self.memory_limit and history_msgs[0]["role"]=="assistant": history_msgs.pop(0)  # 如果歷史對話超過限制，則刪除最舊的訊息，並確保是偶數條數據
        logging.info(f"歷史運用訊息: {history_msgs}")
        now_msgs = [{"role": "user", "content": f'''表情狀態:{image_result},聲音狀態:{voice_result},用戶回應:{user_input}'''}]  # 當前用戶輸入訊息

        completion = client.chat.completions.create(
            model="model-identifier",
            messages= sys_msgs+ history_msgs + now_msgs,  # 合併系統訊息、歷史訊息和當前訊息
            temperature=0.7,
            stream=True,  # 啟用串流模式
        )
        return completion
    
    def send_msg(self):
        usr_msg = self.ui.msg_input.text()
        logging.info(f"用戶輸入: {usr_msg}")
        save_message(user_id=self.user_id,role="user",content=usr_msg) #user msg
        self.append_user_message(self.ui.model_response, usr_msg)

        if re.findall(r"猜數字",usr_msg) != []:
            self.mode = "game"
            self.gamemode = "guess_number"
        elif re.findall(r"猜謎",usr_msg) != []:
            self.mode = "game"
            self.gamemode = "guess_word"
        elif re.findall(r"抽卡",usr_msg) != []:
            self.mode = "game"
            self.gamemode = "pick_card"

        self.start_stream()             # 啟動背景任務來處理模型回應
        self.ui.msg_input.clear()
        self.ui.msg_input.setFocus()    # 清除輸入框後重新聚焦
    
    def start_stream(self):
        llm_thread = threading.Thread(target=self.stream_task, daemon=True)
        llm_thread.start() # 在 threading.Thread 裡啟動背景任務
        self.ui.model_response.append("""
                                        <div style="padding:8px; border-radius:10px; margin:5px 0;">
                                            <b>助理：</b><br>
                                        </div><br>
                                      """)

    def stream_task(self):
        self.buffer = ""  # 初始化緩衝區
        try:
            self.reply_msg = self.get_llm_reply(
                user_input = self.ui.msg_input.text(),
                image_result = self.ui.face_state.text(),
                voice_result = self.ui.voice_state.text()
            )
            logging.info(f"模型回應: {self.reply_msg}")
        except Exception as e:
            logging.info(f"模型回應失敗: {e}")
        for chunk in self.reply_msg:  
            text = chunk.choices[0].delta.content
            if self.think_state == 0:self.signals.new_text.emit(text)
            if text!= None: self.buffer += text
             # 檢查開始與結束思考
            if "<think>" in self.buffer:
                self.think_state = 1
                self.buffer = self.buffer.replace("<think>", "")  # 移除標籤
            if "</think>" in self.buffer:
                self.think_state = 0
                self.buffer = self.buffer.replace("</think>", "")  # 移除標籤
                self.print_think = 0
            
            if text!= None: self.new_model_message += text
            if chunk.choices[0].finish_reason == "stop":
                try:
                    if self.is_tts:
                        clean_txt = self.remove_emoji_simple(text=self.new_model_message)
                        _ = requests.post("http://localhost:5000/speak",data={"text": f"{clean_txt}"})  # TTS朗讀模型回應
                except Exception as e:
                    logging.error(f"TTS模型回應失敗: {e}")
                save_message(user_id=self.user_id, role="assistant", content=self.new_model_message) #assistant msg
                self.signals.del_new_msg.emit()

    def update_browser(self, text):
        if self.think_state == 1 and self.print_think==0:
            self.ui.model_response.moveCursor(QTextCursor.MoveOperation.End)
            self.ui.model_response.insertHtml('<span style="color: gray;">[思考中...]</span><br>')
            self.print_think = 1
            scroll_bar = self.ui.model_response.verticalScrollBar()
            scroll_bar.setValue(scroll_bar.maximum())
        elif self.think_state == 0:
            # self.new_model_message += text
            self.ui.model_response.insertPlainText(text)
            self.ui.model_response.moveCursor(QTextCursor.MoveOperation.End)  # 滾動到最新的回應
    
    def del_nmsg(self):
        self.new_model_message = ""
        logging.info("清空新訊息變數")
        self.ui.model_response.insertPlainText("\n\n---------------------------------------------------------------------------------------------------------------------------------------------\n\n")
        self.ui.model_response.moveCursor(QTextCursor.MoveOperation.End)  # 滾動到最新的回應
        self.reply_msg = ""            
        self.think_state = 0              
        self.mode = ""
        self.gamemode = ""                    
    
    def remove_emoji_simple(self,text):
        """
        簡單版本：只保留字母、數字、中文和基本標點符號
        """
        # 只保留字母、數字、中文字符和基本標點符號
        clean_text = re.sub(r'[^\u4e00-\u9fff\w\s.,！？，。!?;:()\[\]{}\'\"]+', '', text)
        
        # 移除多餘的空格
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def append_user_message(self,text_browser, message):
        html = f"""
        <div style="color:#00ccff;
                    padding:8px; border-radius:10px; margin:5px 0;">
            <b>你：</b> {message}<br>
        </div>
        """
        text_browser.append(html)
        scroll_bar = text_browser.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def feedback(self):
        score = str(self.ui.score_bar.value())
        fb_dict = {"0": "",
                   "1": "這次的回應讓我感覺更糟，完全沒有幫助",
                   "2": "回覆偏離了我的需求，我感覺沒什麼幫助",
                   "3": "回應還可以，但沒有實質幫助我處理情緒",
                   "4": "內容有些相關，但還不夠貼近我的情況",
                   "5": "用戶覺得心情有變好一點了",
                   "6": "有些建議還不錯，我開始感覺有些改善",
                   "7": "回應對我來說蠻有幫助的，讓我釐清了一些思緒",
                   "8": "你的建議讓我心情明顯改善，感覺好多了",
                   "9": "很棒的回應，真心感受到系統的溫度與理解",
                   "10": "超級棒！這正是我需要的，感覺被理解與支持"}  # 評分字典
        if score!="0": 
            try:
                save_message(user_id=self.user_id, role="user", content=f"{fb_dict[score]}")  # 儲存用戶評分
            except Exception as e:
                logging.error(f"儲存用戶評分失敗: {e}")
        logging.info(f"用戶評語: {fb_dict[score]}")
        
    #影像處理區------------------------------------------
    def take_pic(self):
        self.photo = True
        if not os.path.exists('.//photo_tmp'):
            os.mkdir('.//photo_tmp')
            logging.info("初次建立圖片暫存...")
        logging.info("拍照中...")

    def opencv(self):
        try:
            cap = cv2.VideoCapture(0)
        except Exception as e:
            logging.error(f"找不到攝影機: {e}")
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while self.ocv:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            frame = cv2.resize(frame, (480, 320))
            if self.photo == True:
                name = self.rename()                               # 重新命名檔案
                cv2.imwrite(f'.//photo_tmp//{name}.jpg', frame)    # 儲存圖片
                self.photo = False
                self.ui.pic_state.setText(f"已拍照：{name}.jpg")       

                # 將圖片轉成 JPEG 格式後送出
                _, img_encoded = cv2.imencode('.jpg', frame)
                files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                try:
                    response = requests.post(image_url, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        face_emotions = result[0]['emotion']
                        print("辨識結果：", face_emotions)
                        self.ui.face_state.setText(f"{max(face_emotions, key=face_emotions.get)}")  
                    else:
                        print(f"API 錯誤：{response.status_code}", response.json())
                except Exception as e:
                    print("無法連接 API：", e)   

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerline = channel * width
            qimg = QImage(frame, width, height, bytesPerline, QImage.Format.Format_RGB888)
            self.ui.cam_feed.setPixmap(QPixmap.fromImage(qimg))

    #錄音處理區------------------------------------------
    def start_recording(self):
        self.ui.start_rec.setDisabled(True)
        self.ui.stop_rec.setDisabled(False)
        self.ui.rec_state.setText('錄音中....')
        if not os.path.exists('.//voice_tmp'):
            os.mkdir('.//voice_tmp')
            logging.info("初次建立聲音暫存...")
        event.set()      # 觸發錄音開始事件

    def stop_recording(self):
        self.ui.start_rec.setDisabled(False)
        self.ui.stop_rec.setDisabled(True)
        self.ui.rec_state.setText('停止錄音')
        self.run = False       # 設定 run 為 False 停止錄音迴圈
        event2.set()      # 觸發錄音停止事件

    def recording(self):
        while True:
            event.wait()            # 等待事件被觸發
            event.clear()           # 觸發後將事件回歸原本狀態
            self.run = True              # 設定 run 為 True 表示開始錄音
            logging.info('開始錄音...')
            p = pyaudio.PyAudio()   # 建立 pyaudio 物件
            stream = p.open(format=self.sample_format, channels=self.channels, rate=self.fs, frames_per_buffer=self.chunk, input=True)
            frames = [] 
            while self.run:
                data = stream.read(self.chunk)
                frames.append(data)          # 將聲音記錄到串列中
            logging.info('停止錄音')
            stream.stop_stream()             # 停止錄音
            stream.close()                   # 關閉串流
            p.terminate()
            event2.wait()                    # 等待事件被觸發
            event2.clear()                   # 觸發後將事件回歸原本狀態
            # 如果存檔按下確定，表示要儲存
            tmp_name = self.rename()
            wf = wave.open(f'.//voice_tmp//{tmp_name}.wav', 'wb')   # 開啟聲音記錄檔
            wf.setnchannels(self.channels)             # 設定聲道
            wf.setsampwidth(p.get_sample_size(self.sample_format))  # 設定格式
            wf.setframerate(self.fs)                   # 設定取樣頻率
            wf.writeframes(b''.join(frames))      # 存檔
            wf.close()
            self.ui.rec_state.setText(f'已儲存: {tmp_name}.wav')

            try:
                files = {"file": open(f'.//voice_tmp//{tmp_name}.wav', "rb")}
                response = requests.post(voice_url, files=files)
                voice_emotions = response.json()
                logging.info(f"API 回應: {voice_emotions}")
                self.ui.voice_state.setText(f"{max(voice_emotions, key=voice_emotions.get)}") 
            except Exception as e:
                logging.error(f"API 請求失敗: {e}")
            # finally:
            #     if os.path.exists(f'.//voice_tmp//{tmp_name}.wav'):
            #         os.remove(f'.//voice_tmp//{tmp_name}.wav')

if __name__ == '__main__':
    #介面部分
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()

    #影像部分
    video = threading.Thread(target=Form.opencv)
    video.start()

    #錄音部分
    event = threading.Event()   # 註冊錄音事件
    event2 = threading.Event()  # 註冊停止錄音事件
    record = threading.Thread(target=Form.recording)     # 將錄音的部分放入 threading 裡執行
    record.start()

    Form.show()
    sys.exit(app.exec())