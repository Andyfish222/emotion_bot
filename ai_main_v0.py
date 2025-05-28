from PyQt6 import QtWidgets
from PyQt6.QtGui import *
import sys, cv2, threading, datetime, os, wave, random
from ai_t1 import Ui_MainWindow 
import logging
import pyaudio
import requests
from qt_material import apply_stylesheet
from openai import OpenAI

#define
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
voice_url = "http://localhost:5000/predict_voice"
image_url = "http://localhost:5000/predict_image"
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class MyWidget(QtWidgets.QMainWindow):
    def __init__(self):
        # 建立初始化 UI 類別實體、初始化變數
        super().__init__()
        self.setUpdatesEnabled(True)
        self.ui = Ui_MainWindow()       
        self.ui.setupUi(self)
        self.reply_msg = ""  # 初始化模型回應訊息

        #樣式表
        apply_stylesheet(app, theme='dark_amber.xml')
        # with open("style.qss", "r", encoding="utf-8") as f:
        #     self.setStyleSheet(f.read())

        #影像相關
        self.ocv = True                 # 啟用 OpenCV
        self.photo= False               # 拍照狀態

        #錄音相關
        self.chunk = 1024                     # 記錄聲音的樣本區塊大小
        self.sample_format = pyaudio.paInt16  # 樣本格式，可使用 paFloat32、paInt32、paInt24、paInt16、paInt8、paUInt8、paCustomFormat
        self.channels = 1                     # 聲道數量
        self.fs = 44100                       # 取樣頻率，常見值為 44100 ( CD )、48000 ( DVD )、22050、24000、12000 和 11025。
        # self.seconds = 5                      # 錄音秒數
        self.run = False

        #按鈕動作區
        self.ui.take_pic.clicked.connect(self.take_pic)
        self.ui.start_rec.clicked.connect(self.start_recording)
        self.ui.stop_rec.clicked.connect(self.stop_recording)
        self.ui.send_msg.clicked.connect(self.send_msg)

    def closeEvent(self):
        self.ocv = False

    #暫存檔名:當下時間
    def rename(self):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    #模型回應區------------------------------------------
    def get_llm_reply(self,user_input,image_result,voice_result):
        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": '''你是一位情緒智能助手，能夠理解和回應用戶的情感需求，並總是以繁體中文回應。
                用戶輸入可能包含表情狀態、聲音狀態和文字描述，你的目標是用好朋友的口吻來使用戶開心。'''},
                {"role": "user", "content": 
                    f'''表情狀態:{image_result},
                        聲音狀態:{voice_result},
                        用戶回應:{user_input}'''
                },
                # {"role": "assistant", "content": "當你心情不好時，可以試著做一些讓自己放鬆的事情，比如聽音樂、散步或是和朋友聊天。也可以嘗試寫下你的感受，這樣有助於釐清思緒。最重要的是，給自己一些時間去感受和處理這些情緒。"},
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    def send_msg(self):
        try:
            self.reply_msg = self.get_llm_reply(
                user_input = self.ui.msg_input.text(),
                image_result = self.ui.face_state.text(),
                voice_result = self.ui.voice_state.text()
            )
            logging.info(f"用戶輸入: {self.ui.msg_input.text()}")
            logging.info(f"模型回應: {self.reply_msg}")
        except Exception as e:
            logging.info(f"模型回應失敗: {e}")
        self.ui.model_response.append(self.reply_msg+"\n---------------------------------------------------------------------------------------------------------------------------------------------")  # 將模型回應加入到對話框中
        self.ui.msg_input.clear()
        self.reply_msg = ""             # 清空模型回應變數
        self.ui.msg_input.setFocus()    # 清除輸入框後重新聚焦

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