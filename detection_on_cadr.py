from datetime import timedelta
import cv2
import pandas as pd
import numpy as np
import torch
from dop_fun import resize_img, time_of_function
import time
from person_yolov8 import person_filter, stk_filter, casha_filter, truck_filter ,previu_video


def format_timedelta( td):
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

def get_saving_frames_durations( cap, saving_fps):
    """Функция, которая возвращает список длительностей, в которые следует сохранять кадры."""
    s = []
    # получаем продолжительность клипа, разделив количество кадров на количество кадров в секунду
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # используйте np.arange () для выполнения шагов с плавающей запятой
    for i in np.arange(0, clip_duration,  saving_fps):
        s.append(i)
    return s

@time_of_function
def detection_on_cadr(video_file):
    # читать видео файл
    cap = cv2.VideoCapture(video_file)
    # получить FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    cad = {}
    while True:
        is_read, frame = cap.read()

        if not is_read:
            # выйти из цикла, если нет фреймов для чтения
            break
        frame_duration = count / fps
        frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
        cad[frame_duration_formatted] = resize_img(frame, 1280)
        count += 1
    return cad, fps

def detection(video_file, al, cat, probability, classificator=0, clas_box=0, n=1):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if classificator == 0:
        clas_box = 0
    cad, fps = detection_on_cadr(video_file)

    time_kad = list(cad.keys())
    kadr = list(cad.values())
    frameSize = (kadr[0].shape[1], kadr[0].shape[0])
    out_put_video = video_file[:-4] + '_previu_' + '.mp4'
    out = cv2.VideoWriter(out_put_video, cv2.VideoWriter_fourcc(*'MP4V'), fps, frameSize)
    if len(kadr) > n:
        phot1 = [kadr[i:i + n] for i in range(0, len(kadr), n)]
        nam1 = [time_kad[i:i + n] for i in range(0, len(time_kad), n)]
    df = pd.DataFrame()
    for frame_duration_formatted, frame in zip(nam1, phot1):
        print(frame_duration_formatted)
        if al == 0:
            if cat == 'person':
                strok = person_filter(frame_duration_formatted, frame, classificator, device)
            if cat == 'chasha':
                strok = casha_filter(frame_duration_formatted, frame, device)
            if cat == 'truck':
                strok = truck_filter(frame_duration_formatted, frame, device)
            if cat == 'stk':
                strok = stk_filter(frame_duration_formatted, frame, device)
        if al == 1:
            strok_p = person_filter(frame_duration_formatted, frame, classificator, device)
            strok_c = casha_filter(frame_duration_formatted, frame, device)
            strok_t = truck_filter(frame_duration_formatted, frame, device)
            strok_s = stk_filter(frame_duration_formatted, frame, device)
            strok = pd.concat([strok_p,strok_s,strok_t,strok_c])

        for n, img in zip(frame_duration_formatted, frame):
            st = strok[strok['time_cadr'] == n]
            if len(st.name.unique()) != 0:
                kad = previu_video(img, st, probability, clas_box)
            else:
                kad = img
            # out.write(kad)
            kadr.append(kad)
        for k in kadr:
            out.write(k)
        # формирование финального датафрейма с найденными объектами
        df = pd.concat([df, strok])
    out.release()
    return df


print(detection('./test/test.mp4', 0, 'person', 1, 0, 1, 5))
