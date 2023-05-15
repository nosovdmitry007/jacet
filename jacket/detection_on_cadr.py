from datetime import timedelta
import cv2
import pandas as pd
from person import People,  sav,  Chasha, Truck, STK, previu_video
import numpy as np
import imutils

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


def detection_on_cadr(video_file, al, cat, save_catalog, kad, ramka, probability, save_frame, clas=0, clas_box=0, prev_video=0):


    # читать видео файл
    cap = cv2.VideoCapture(video_file)
    # получить FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Подготовка для записи видео, с детекцией
    if prev_video == 1:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        #Конвертируем разрешение под выходное из НС
        if width > height:
            height = height/(width/1280)
            width = 1280
        else:
            width = width/(height/1280)
            height = 1280
        frameSize = (int(width), int(height))
        out_put_video = video_file[:-4]+'_previu_'+'.mp4'
        out = cv2.VideoWriter(out_put_video, cv2.VideoWriter_fourcc(*'MP4V'), fps, frameSize)
    if kad != 'fps':
        SAVING_FRAMES_PER_SECOND = kad
    else:
        SAVING_FRAMES_PER_SECOND = 1/fps
    # если SAVING_FRAMES_PER_SECOND выше видео FPS, то установите его на FPS (как максимум)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # получить список длительностей для сохранения
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # запускаем цикл
    count = 0
    df = pd.DataFrame()
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
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))

            if frame.shape[0] < frame.shape[1]:
                frame = imutils.resize(frame, width=1280)
            else:
                frame = imutils.resize(frame, height=1280)
            #Выбираем модель которую вызываем
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
            #Если необходимо создавать превью
            if prev_video == 0:
                if len(strok.name.unique()) != 0:
                    sav(frame, frame_duration_formatted, strok, save_catalog, ramka, probability, save_frame, clas_box)
            else:
                if len(strok.name.unique()) != 0:
                    kad = previu_video(frame, strok, probability, clas_box)
                else:
                    kad = frame
                out.write(kad)
# удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров

        count += 1
    out.release()
    return df


print(detection_on_cadr('./Данные для обучения нейросети/безымянный.mp4', 1, 'person', 'person_test', 'fps', 1, 1, 0, 1, 1, 1))