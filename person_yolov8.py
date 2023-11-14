import cv2
import pybboxes as pbx
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from dop_fun import slesh, time_of_function

@time_of_function
def pandas_frame(nam_cad:list[str], results, classificator:int, cad_list:list, obj_detect:str):
    model_class = cv2.dnn.readNetFromONNX('./model/classificator.onnx')
    df_c:pandas_frame = pd.DataFrame()
    z:int = 0
    CLASSES = ['Hat', 'Jacket', 'JacketAndHat', 'None']
    for res in results:
        result = res[0]
        z += 1
        column:list[str] = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
        df:pandas_frame = pd.DataFrame(result.boxes.data.tolist(), columns=column)
        df['name'] = obj_detect
        nom:int = int(((result.path).split('.')[0]).split('e')[1])
        df['time_cadr'] = nam_cad[nom]
        if classificator == 1:
            clas:list = []
            im:list = cad_list[z-1]
            for k in df.values.tolist():
                kad:list = im[int(k[1]):int(k[3]), int(k[0]):int(k[2])]
                blob:list = cv2.dnn.blobFromImage(cv2.resize(kad, (96, 96)), scalefactor=1.0 / 96
                                             , size=(96, 96), mean=(128, 128, 128), swapRB=True)
                model_class.setInput(blob)
                detections:list = model_class.forward()
                # преобразуем оценки в вероятности softmax
                detections:list = np.exp(detections) / np.sum(np.exp(detections))
                class_mark:int = np.argmax(detections)
                clas.append(CLASSES[class_mark])
            df['class_people'] = clas
        df_c = pd.concat([df_c, df])
    return df_c


@time_of_function
def person_filter(nam_cad, cad_list, classificator=0, device=1):
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_detect_people = YOLO("./model/person_v8.pt")  # ./model/model_scripted.pt")
    results = model_detect_people(cad_list, imgsz=1280, device=device, classes=0, conf=0.5, stream=True)
    return pandas_frame(nam_cad, results, classificator, cad_list, 'person')


@time_of_function
def casha_filter(nam_cad, cad_list, classificator=0, device=1):
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_detect_people = YOLO("./model/chasha_v8.pt")  # ./model/model_scripted.pt")
    results = model_detect_people(cad_list, imgsz=1280, device=device, classes=0, conf=0.5)
    return pandas_frame(nam_cad, results, classificator, cad_list, 'casha')


@time_of_function
def truck_filter(nam_cad, cad_list, classificator=0, device=1):
    model_detect_people = YOLO("./model/truck_v8.pt")  # ./model/model_scripted.pt")
    results = model_detect_people(cad_list, imgsz=1280, device=device, classes=0, conf=0.5)
    return pandas_frame(nam_cad, results, classificator, cad_list, 'truck')


@time_of_function
def stk_filter(nam_cad, cad_list, classificator=0, device=1):
    model_detect_people = YOLO("./model/stk_v8.pt.pt")  # ./model/model_scripted.pt")
    results = model_detect_people(cad_list, imgsz=1280, device=device, classes=0, conf=0.5)
    return pandas_frame(nam_cad, results, classificator, cad_list, 'STK')


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
            # print(k)
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
    sleh = slesh()

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