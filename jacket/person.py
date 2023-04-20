import cv2
import cv2_ext
import pybboxes as pbx
import imutils
import numpy as np
import torch
import pandas as pd
from datetime import timedelta
import os
from tensorflow.keras.models import load_model


class People:
    def __init__(self):
        self.model_detect_people = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/yolov5m6.pt',
                                  source='local')
        self.model_class = load_model('./model/classification.h5', compile=False)
    def kadr(self, img):
        if img.shape[0] < img.shape[1]:
            image = imutils.resize(img, height=1280)
        else:
            image = imutils.resize(img, width=1280)
        return image

    def person_filter(self, put, cad='1', video=0, classificator=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put

        image = self.kadr(image)
        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        df = df.drop(np.where(df['name'] != 'person')[0])
        if video == 1:
            df['time_cadr'] = cad
        if classificator == 1:
            className = ['JacketAndHat', 'Hat', 'None', 'Jacket']
            visota = 50
            shirina = 50
            clas = []
            for k in df.values.tolist():
                # print(k)
                kad = image[int(k[1]):int(k[3]), int(k[0]):int(k[2])]
                kad = cv2.cvtColor(cv2.resize(kad, (shirina,visota)),cv2.COLOR_BGR2RGB)
                kad = kad / 255
                sp = []
                sp.append(kad)
                sp = np.array(sp)
                prediction = self.model_class.predict(sp)
                # print(className[np.argmax(prediction)])
                clas.append(className[np.argmax(prediction)])
            df['class_people'] = clas
            # print(df)
        return df


class Truck:
    def __init__(self):
        self.model_detect_people = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/truck.pt',
                                  source='local')

    def truck_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        image = self.kadr(image)
        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        print(df)
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df


class Chasha:
    def __init__(self):
        self.model_detect_chasha = torch.hub.load('./yolov5_master', 'custom',
                                                  path='./model/chasha.pt',
                                                  source='local')

    def chasha_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        image = self.kadr(image)
        results = self.model_detect_chasha(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.1)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df


class Kadr:
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
                cad = [frame, frame_duration_formatted]
                cadre.append(cad)
                # print(cadre)
                # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame)
                # удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # увеличить количество кадров
            count += 1
        return cadre


def sav(kadr, name, fil, put, ramka, probability,save_frame,clas = 0):
    if not os.path.exists(put):
        os.makedirs(put)
        os.makedirs(f"{put}/images")
        os.makedirs(f"{put}/save_frame")
        os.makedirs(f"{put}/txt")
        os.makedirs(f"{put}/txt_yolo")
    if not os.path.exists(f"{put}/images"):
        os.makedirs(f"{put}/images")
    if not os.path.exists(f"{put}/txt"):
        os.makedirs(f"{put}/txt")
    if not os.path.exists(f"{put}/txt_yolo"):
        os.makedirs(f"{put}/txt_yolo")
    if not os.path.exists(f"{put}/save_frame"):
        os.makedirs(f"{put}/save_frame")

    colum = ['class', 'xmin', 'ymin', 'xmax', 'ymax']
    if kadr.shape[0] < kadr.shape[1]:
        kadr = imutils.resize(kadr, height=1280)
    else:
        kadr = imutils.resize(kadr, width=1280)

    if save_frame == 1:
        sd = 0
        for k in fil.values.tolist():
            crop_img = kadr[int(k[1]):int(k[3]),int(k[0]):int(k[2])]
            cv2.imwrite(f"{put}/save_frame/frame8_{name}_{sd}.jpg", crop_img)
            sd += 1

    if ramka == 1:
        for k in fil.values.tolist():
            if k[6] != 'person' or clas == 0:
                cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 0, 255), 2)
                if probability == 1:
                    cv2.putText(kadr, str(round(k[4],2)), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(0, 255, 0), thickness=2)
            if k[6] == 'person' and clas == 1:
                if k[8] == 'None':
                    cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 0, 255), 2)
                    if probability == 1:
                        cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(0, 255, 0), thickness=2)

                if k[8] == 'JacketAndHat':
                    cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 255, 0), 2)
                    if probability == 1:
                        cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(0, 255, 0), thickness=2)
                if k[8] == 'Hat' or k[8] == 'Jacket':
                    cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (255, 0, 0), 2)
                    if probability == 1:
                        cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8, color=(0, 255, 0), thickness=2)


    cv2.imwrite(f"{put}/images/frame_{name}.jpg", kadr)

    fil.to_csv(f"{put}/txt/frame_{name}.txt", columns=colum, header=False, sep='\t', index=False)
    yolo = []

    for row in fil.values.tolist():
        W, H = kadr.shape[1], kadr.shape[0]
        y = list(pbx.convert_bbox((row[0], row[1], row[2], row[3]), from_type="voc", to_type="yolo", image_size=(W, H)))
        y.insert(0,row[5])
        yolo.append(y)

    dy = pd.DataFrame(yolo, columns=colum)
    dy.to_csv(f"{put}/txt_yolo/frame_yolo_{name}.txt", columns=colum, header=False, sep='\t', index=False)







