import glob
import os
import platform
import cv2
import cv2_ext
<<<<<<< HEAD
# import imageio
import imutils
import numpy as np
# import rawpy
=======
import imageio
import imutils
import numpy as np
import rawpy
>>>>>>> c4d82ba (1)
import torch
import pandas as pd
from datetime import timedelta

class Jacket:
    def __init__(self):
        self.model_detect = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/yolov5m6.pt',
                                  source='local')

    def person_filter(self, put,video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put

        if image.shape[0] < image.shape[1]:
            image = imutils.resize(image, height=1280)
        else:
            image = imutils.resize(image, width=1280)

        results = self.model_detect(image)
        df = results.pandas().xyxy[0]
        # df = df.drop(np.where(df['confidence'] < 0.1)[0])
<<<<<<< HEAD
        # print(df)
=======
        print(df)
>>>>>>> c4d82ba (1)
        # ob = pd.DataFrame()
        # ob['class'] = df['name']
        # oblasty = ob.values.tolist()
        # oblasty = sum(oblasty, [])
        return df
        # if cat in oblasty:
        #     os.replace(put + sleh + i, put + sleh + cat + sleh + i)

    def format_timedelta(self,td):
        """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
        исключая микросекунды и сохраняя миллисекунды"""
        result = str(td)
        try:
            result, ms = result.split(".")
        except ValueError:
            return result + ".00".replace(":", "-")
        ms = int(ms)
        ms = round(ms / 1e4)
        return f"{result}.{ms:02}".replace(":", "-")

    def get_saving_frames_durations(self,cap, saving_fps):
        """Функция, которая возвращает список длительностей, в которые следует сохранять кадры."""
        s = []
        # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        # используйте np.arange () для выполнения шагов с плавающей запятой
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s

    def cadre(self,video_file):
        SAVING_FRAMES_PER_SECOND = 10
        filename, _ = os.path.splitext(video_file)
        filename += "-opencv"
        # создаем папку по названию видео файла
        if not os.path.isdir(filename):
            os.mkdir(filename)
        # читать видео файл
        cap = cv2.VideoCapture(video_file)
        # получить FPS видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        # если SAVING_FRAMES_PER_SECOND выше видео FPS, то установите его на FPS (как максимум)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # получить список длительностей для сохранения
        saving_frames_durations = self.get_saving_frames_durations(cap, saving_frames_per_second)
        # запускаем цикл
        count = 0
        cadre = []
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # выйти из цикла, если нет фреймов для чтения
                break
            # получаем продолжительность, разделив количество кадров на FPS
            frame_duration = count / fps
            try:
                # получить самую раннюю продолжительность для сохранения
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # список пуст, все кадры длительности сохранены
                break
            if frame_duration >= closest_duration:
                # если ближайшая длительность меньше или равна длительности кадра,
                # затем сохраняем фрейм
                frame_duration_formatted = self.format_timedelta(timedelta(seconds=frame_duration))

                cadre.append(frame)
                # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame)
                # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # увеличить количество кадров
            count += 1
        return cadre


    def person_filter_video(self, put):
        z = self.cadre(put)
        print(z)
        for i in z:
            print(self.person_filter(i,1))







