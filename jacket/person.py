
import os
import cv2
import cv2_ext

import imutils
import numpy as np
import torch
import pandas as pd
from datetime import timedelta

class People:
    def __init__(self):
        self.model_detect_people = torch.hub.load('./yolov5_master', 'custom',
                                  path='./model/yolov5m6.pt',
                                  source='local')

    def person_filter(self,put, cad='1', filter='person', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        if image.shape[0] < image.shape[1]:
            image = imutils.resize(image, height=1280)
        else:
            image = imutils.resize(image, width=1280)
        results = self.model_detect_people(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.1)[0])
        df = df.drop(np.where(df['name'] != filter)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df

class Jalet:
    def __init__(self):
        self.model_detect_jalet = torch.hub.load('./yolov5_master', 'custom',
                                                  path='./model/jalet.pt',
                                                  source='local')

    def jalet_filter(self, put, cad='1', video=0):
        if video == 0:
            image = cv2_ext.imread(put)
        else:
            image = put
        if image.shape[0] < image.shape[1]:
            image = imutils.resize(image, height=1280)
        else:
            image = imutils.resize(image, width=1280)
        results = self.model_detect_jalet(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.1)[0])
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
        if image.shape[0] < image.shape[1]:
            image = imutils.resize(image, height=1280)
        else:
            image = imutils.resize(image, width=1280)
        results = self.model_detect_chasha(image)
        df = results.pandas().xyxy[0]
        df = df.drop(np.where(df['confidence'] < 0.1)[0])
        if video == 1:
            df['time_cadr'] = cad
        return df

class Kadr:

    def format_timedelta(self,td):
        """?????????????????? ?????????????? ?????? ?????????????????? ???????????????????????????? ???????????????? timedelta (????????????????, 00:00:20.05)
        ???????????????? ???????????????????????? ?? ???????????????? ????????????????????????"""
        result = str(td)
        try:
            result, ms = result.split(".")
        except ValueError:
            return result + ".00".replace(":", "-")
        ms = int(ms)
        ms = round(ms / 1e4)
        return f"{result}.{ms:02}".replace(":", "-")

    def get_saving_frames_durations(self,cap, saving_fps):
        """??????????????, ?????????????? ???????????????????? ???????????? ??????????????????????????, ?? ?????????????? ?????????????? ?????????????????? ??????????."""
        s = []
        # ???????????????? ?????????????????????????????????? ??????????, ???????????????? ???????????????????? ???????????? ???? ???????????????????? ???????????? ?? ??????????????
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        # ?????????????????????? np.arange () ?????? ???????????????????? ?????????? ?? ?????????????????? ??????????????
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s

    def cadre(self,video_file):
        SAVING_FRAMES_PER_SECOND = 10
        filename, _ = os.path.splitext(video_file)
        filename += "-opencv"
        # ?????????????? ?????????? ???? ???????????????? ?????????? ??????????
        if not os.path.isdir(filename):
            os.mkdir(filename)
        # ???????????? ?????????? ????????
        cap = cv2.VideoCapture(video_file)
        # ???????????????? FPS ??????????
        fps = cap.get(cv2.CAP_PROP_FPS)
        # ???????? SAVING_FRAMES_PER_SECOND ???????? ?????????? FPS, ???? ???????????????????? ?????? ???? FPS (?????? ????????????????)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # ???????????????? ???????????? ?????????????????????????? ?????? ????????????????????
        saving_frames_durations = self.get_saving_frames_durations(cap, saving_frames_per_second)
        # ?????????????????? ????????
        count = 0
        cadre = []
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # ?????????? ???? ??????????, ???????? ?????? ?????????????? ?????? ????????????
                break
            # ???????????????? ??????????????????????????????????, ???????????????? ???????????????????? ???????????? ???? FPS
            frame_duration = count / fps
            try:
                # ???????????????? ?????????? ???????????? ?????????????????????????????????? ?????? ????????????????????
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # ???????????? ????????, ?????? ?????????? ???????????????????????? ??????????????????
                break
            if frame_duration >= closest_duration:
                # ???????? ?????????????????? ???????????????????????? ???????????? ?????? ?????????? ???????????????????????? ??????????,
                # ?????????? ?????????????????? ??????????
                frame_duration_formatted = self.format_timedelta(timedelta(seconds=frame_duration))
                cad = [frame, frame_duration_formatted]
                cadre.append(cad)
                # print(cadre)
                # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame)
                # ?????????????? ?????????? ?????????????????????????????????? ???? ????????????, ?????? ?????? ?????? ?????????? ???????????????????????? ?????? ??????????????????
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # ?????????????????? ???????????????????? ????????????
            count += 1
        return cadre










