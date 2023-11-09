from datetime import timedelta
import cv2
import pandas as pd
# from person import People,  sav,  Chasha, Truck, STK, previu_video
from person_yolov8 import People,  sav,  Chasha, Truck, STK, previu_video
import numpy as np
import imutils
from dop_fun import resize_img
import time

people = People()
chasha = Chasha()
truck = Truck()
stk = STK()

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

def detection(video_file, al, cat, save_catalog, fps, ramka, probability, save_frame, clas=0, clas_box=0, prev_video=0):
            #Выбираем модель которую вызываем
    # cap = cv2.VideoCapture(video_file)
    # if prev_video == 1:
        # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
        # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        # # Конвертируем разрешение под выходное из НС
        # if width > height:
        #     height = height / (width / 1280)
        #     width = 1280
        # else:
        #     width = width / (height / 1280)
        #     height = 1280

    cad, fps = detection_on_cadr(video_file)

    time_kad = list(cad.keys())
    kadr = list(cad.values())

    frameSize = (kadr[0].shape[1], kadr[0].shape[0])

    out_put_video = video_file[:-4] + '_previu_' + '.mp4'
    out = cv2.VideoWriter(out_put_video, cv2.VideoWriter_fourcc(*'MP4V'), fps, frameSize)


    n = 5
    if len(kadr) > n:
        phot1 = [kadr[i:i + n] for i in range(0, len(kadr), n)]
        nam1 = [time_kad[i:i + n] for i in range(0, len(time_kad), n)]
    df = pd.DataFrame()
    for frame_duration_formatted, frame in zip(nam1, phot1):
        if al == 0:
            if cat == 'person':
                strok = people.person_filter(frame, frame_duration_formatted, 1, clas)
            if cat == 'chasha':
                strok = chasha.chasha_filter(frame, frame_duration_formatted, 1)
            if cat == 'truck':
                strok = truck.truck_filter(frame, frame_duration_formatted, 1)
            if cat == 'stk':
                strok = stk.stk_filter(frame, frame_duration_formatted, 1)
        if al == 1:
                strok_p = people.person_filter(frame, frame_duration_formatted, 1, clas)
                strok_c = chasha.chasha_filter(frame, frame_duration_formatted, 1)
                strok_t = truck.truck_filter(frame, frame_duration_formatted, 1)
                strok_s = stk.stk_filter(frame, frame_duration_formatted, 1)
                strok = pd.concat([strok_p,strok_s,strok_t,strok_c])

        #формирование финального датафрейма с найденными объектами
        df = pd.concat([df, strok])
        # print(df)
        #Если необходимо создавать превью

        # if prev_video == 0:
        # if len(strok.name.umnique()) != 0:
        #     sav(frame, frame_duration_formatted, strok, save_catalog, ramka, probability, save_frame, clas_box)
        # else:
        for n, img in zip(frame_duration_formatted, frame):
            st = strok[strok['time_cadr'] == n]
            if len(st.name.unique()) != 0:
                kad = previu_video(img, st, probability, clas_box)
            else:
                kad = img
            out.write(kad)

    return df

start = time.time()
print(detection('./test/test.mp4', 0, 'person', 'person_test', 'fps', 1, 1, 0, 1, 1, 1))
end = time.time()

# print the difference between start
# and end time in milli. secs
print("The time of execution of above program is :",
      (end - start) * 10 ** 3, "ms")

# print((detection_on_cadr('./test/test.mp4','fps',1)))