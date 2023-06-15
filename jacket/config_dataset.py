import os
import random
import platform

sistem = platform.system()
if 'Win' in sistem:
    sleh = '\\'
else:
    sleh = '/'
def config_dataset_yolo5(put_in,img,labl,tr,put_out):
    # put_in - путь к папке с фйлами
    # img - название папки с изображениями
    # labl - название папки с разметкой
    # tr - процент выборки для тренировки (в формате 80)
    # put_out - куда сохрнять датасет

    put_img = put_in +sleh+ img
    put_txt = put_in + sleh + labl
    img = os.listdir(put_img)
    random.shuffle(img)
    train = int(len(img)*(tr/100))
    train_img = put_out + sleh + 'dataset' + sleh + 'images' + sleh + 'train'
    train_txt = put_out + sleh + 'dataset' + sleh + 'labels' + sleh + 'train'
    val_img = put_out + sleh + 'dataset' + sleh + 'images' + sleh + 'valid'
    val_txt = put_out + sleh + 'dataset' + sleh + 'labels' + sleh + 'valid'

    if not os.path.isdir(put_out + sleh + 'dataset'):
        os.makedirs(train_img)
        os.makedirs(train_txt)
        os.makedirs(val_img)
        os.makedirs(val_txt)

    for i in range(0, train):
        txt = ''
        os.replace(put_img + sleh + img[i], train_img + sleh + img[i])
        txt = img[i][:-4] + '.txt'
        os.replace(put_txt + sleh + txt, train_txt + sleh + txt)

    for j in range(train+1, len(img)):
        txt = ''
        os.replace(put_img + sleh + img[j], val_img + sleh + img[j])
        txt = img[j][:-4] + '.txt'
        os.replace(put_txt + sleh + txt, val_txt + sleh + txt)


config_dataset_yolo5('./Данные для обучения нейросети/person/person', 'img_opt', 'labels', 80, './Данные для обучения нейросети/person')