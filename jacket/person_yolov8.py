import cv2
import cv2_ext
import pybboxes as pbx
import numpy as np
import torch
import pandas as pd
import os
import platform
from ultralytics import YOLO

#Детектор и классификатор людей
class People:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model_detect_people = YOLO("./model/person_v8.pt")
        self.model_class = cv2.dnn.readNetFromONNX('./model/classificator.onnx')

    def person_filter(self, put, cad='1', video=0, classificator=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        results = self.model_detect_people(image,imgsz=1280, device=self.device, classes=0)
        for result in results:
            column = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
            df = pd.DataFrame(result.boxes.data.tolist(), columns=column)
            df['name'] = df['class'].apply(lambda x: result.names[x])
        #Установка порога уверености модели
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        if video == 1:
            df['time_cadr'] = cad
        if classificator == 1:
            CLASSES = ['Hat', 'Jacket', 'JacketAndHat', 'None']
            clas = []
            for k in df.values.tolist():
                kad = image[int(k[1]):int(k[3]), int(k[0]):int(k[2])]

                blob = cv2.dnn.blobFromImage(cv2.resize(kad, (32, 32)), scalefactor=1.0 / 32
                                             , size=(32, 32), mean=(128, 128, 128), swapRB=True)
                self.model_class.setInput(blob)
                detections = self.model_class.forward()
                #преобразуем оценки в вероятности softmax
                detections = np.exp(detections) / np.sum(np.exp(detections))
                class_mark = np.argmax(detections)
                clas.append(CLASSES[class_mark])

            df['class_people'] = clas
        return df


class Truck:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model_detect_TRUCK = YOLO("./model/truck_v8.pt")
    def truck_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        results = self.model_detect_TRUCK(image, imgsz=1280, device=self.device, classes=0)
        for result in results:
            column = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
            df = pd.DataFrame(result.boxes.data.tolist(), columns=column)
            df['name'] = df['class'].apply(lambda x: result.names[x])
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df

class STK:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model_detect_STK = YOLO("./model/stk_v8.pt")

    def stk_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put

        results = self.model_detect_STK(image, imgsz=1280, device=self.device, classes=0)
        for result in results:
            column = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
            df = pd.DataFrame(result.boxes.data.tolist(), columns=column)
            df['name'] = df['class'].apply(lambda x: result.names[x])
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df


class Chasha:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model_detect_CASHA = YOLO("./model/chasha_v8.pt")
    def chasha_filter(self, put, cad='1', video=0):

        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        results = self.model_detect_CASHA(image, imgsz=1280, device=self.device, classes=0)
        for result in results:
            column = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
            df = pd.DataFrame(result.boxes.data.tolist(), columns=column)
            df['name'] = df['class'].apply(lambda x: result.names[x])
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
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


def previu_video(kadr, fil, probability, clas_box):
    #проходим по каждой строке из датасета с найденными объектами на кадре
    for k in fil.values.tolist():
        if k[6] != 'person' or clas_box == 0:
            cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 0, 255), 1)
            if probability == 1:
                cv2.putText(kadr, str(k[6]) + ' ' + str(round(k[4], 2)), (int(k[0]), int(k[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(0, 255, 0), thickness=2)
        if k[6] == 'person' and clas_box == 1:
            if k[8] == 'None':
                cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 0, 255), 1)
                if probability == 1:
                    cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(0, 255, 0), thickness=2)

            if k[8] == 'JacketAndHat':
                cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (0, 255, 0), 1)
                if probability == 1:
                    cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(0, 255, 0), thickness=2)
            if k[8] == 'Hat' or k[8] == 'Jacket':
                cv2.rectangle(kadr, (int(k[0]), int(k[1])), (int(k[2]), int(k[3])), (255, 0, 0), 1)
                if probability == 1:
                    cv2.putText(kadr, str(k[8]), (int(k[0]), int(k[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8, color=(0, 255, 0), thickness=2)
    return kadr


def sav(kadr, name, fil, put, ramka, probability,save_frame,clas_box = 0):
    sistem = platform.system()
    if 'Win' in sistem:
        sleh = '\\'
    else:
        sleh = '/'

    if not os.path.exists(put):
        os.makedirs(put)
        os.makedirs(f"{put}{sleh}images")
        os.makedirs(f"{put}{sleh}save_frame")
        os.makedirs(f"{put}{sleh}txt")
        os.makedirs(f"{put}{sleh}txt_yolo")
    if not os.path.exists(f"{put}{sleh}images"):
        os.makedirs(f"{put}{sleh}images")
    if not os.path.exists(f"{put}{sleh}txt"):
        os.makedirs(f"{put}{sleh}txt")
    if not os.path.exists(f"{put}{sleh}txt_yolo"):
        os.makedirs(f"{put}{sleh}txt_yolo")
    if not os.path.exists(f"{put}{sleh}save_frame"):
        os.makedirs(f"{put}{sleh}save_frame")
    name = name.replace(':', '_')
    colum = ['class', 'xmin', 'ymin', 'xmax', 'ymax']

    if save_frame == 1:
        sd = 0
        for k in fil.values.tolist():
            crop_img = kadr[int(k[1]):int(k[3]),int(k[0]):int(k[2])]
            cv2.imwrite(f"{put}{sleh}save_frame{sleh}frame1_{name}_{sd}.jpg", crop_img)
            sd += 1

    if ramka == 1:
        kadr = previu_video(kadr, fil, probability, clas_box)

    cv2.imwrite(f"{put}{sleh}images{sleh}frame1_{name}.jpg", kadr)

    fil.to_csv(f"{put}{sleh}txt{sleh}frame_{name}.txt", columns=colum, header=False, sep='\t', index=False)
    yolo = []

    for row in fil.values.tolist():
        W, H = kadr.shape[1], kadr.shape[0]
        y = list(pbx.convert_bbox((row[0], row[1], row[2], row[3]), from_type="voc", to_type="yolo", image_size=(W, H)))
        y.insert(0,row[5])
        yolo.append(y)

    dy = pd.DataFrame(yolo, columns=colum)
    dy.to_csv(f"{put}{sleh}txt_yolo{sleh}frame_yolo_{name}.txt", columns=colum, header=False, sep='\t', index=False)