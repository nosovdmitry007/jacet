from datetime import timedelta
from tqdm import tqdm
import cv2
import pandas as pd
import os
from person import People,  sav, Jalet, Chasha, Truck
import numpy as np

people = People()
jalet = Jalet()
chasha = Chasha()
truck = Truck()

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
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def detection_on_cadr(video_file,cat, save_catalog, ramka, probability,save_frame):
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
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # запускаем цикл
    count = 0
    cadre = []
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
        #Выбираем модель которую вызываем
            if cat == 'jalet':
                str = jalet.jalet_filter(frame, frame_duration_formatted, 1)
            if cat == 'person':
                str = people.person_filter(frame, frame_duration_formatted, 1)
            if cat == 'chasha':
                str = chasha.chasha_filter(frame, frame_duration_formatted, 1)
            if cat == 'truck':
                str = truck.truck_filter(frame, frame_duration_formatted, 1)
            df = pd.concat([df, str])
            if len(str.name.unique()) != 0:
                sav(frame, frame_duration_formatted, str, save_catalog, ramka, probability,save_frame)
# удалить точку продолжительности из списка, так как эта точка длительности уже сохранена
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # увеличить количество кадров
        count += 1
    return df


print(detection_on_cadr('Данные для обучения нейросети/Каски и жилеты/IMG_4304.MOV', 'person', 'test_people1', 1, 1, 1))