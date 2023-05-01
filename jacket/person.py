import cv2
import cv2_ext
import pybboxes as pbx
import imutils
import numpy as np
import torch
import pandas as pd
from datetime import timedelta
import os
from torchvision import transforms as T
import torch.nn.functional as F
import platform

class People:
    def __init__(self):
        self.model_detect_people = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/person.pt',
                                  source='local')
        self.model_class = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/classificator.pt',
                                  source='local',
                                  force_reload=True)

    def person_filter(self, put, cad='1', video=0, classificator=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put


        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        # df = df.drop(np.where(df['name'] != 'person')[0])
        if video == 1:
            df['time_cadr'] = cad
        if classificator == 1:
            IMAGENET_MEAN = 0.485, 0.456, 0.406
            IMAGENET_STD = 0.229, 0.224, 0.225

            def classify_transforms(size=224):
                return T.Compose(
                    [T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
            clas = []
            for k in df.values.tolist():
                # print(k)
                kad = image[int(k[1]):int(k[3]), int(k[0]):int(k[2])]
                # kad = cv2.cvtColor(kad,cv2.COLOR_RGB2BGR)
                # print(kad)
                # cv2.imwrite(f"jacket/test_people2/save_frame/kda{k[1]}.jpg", kad)
                transformations = classify_transforms()
                convert_tensor = transformations(kad)
                convert_tensor = convert_tensor.unsqueeze(0)

                results = self.model_class(convert_tensor)
                pred = F.softmax(results, dim=1)
                k = pred[0].tolist()
                mx = max(k)
                z = k.index(mx)
                clas.append(self.model_class.names[z])
                # print(clas)
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
        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        # print(df)
        df = df.drop(np.where(df['confidence'] < 0.3)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df

class STK:
    def __init__(self):
        self.model_detect_people = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/stk.pt',
                                  source='local')

    def stk_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put

        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        # print(df)
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
        results = self.model_detect_chasha(image)
        # print(results)
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

#
