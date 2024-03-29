# Система классификации объектов

## Установка:

1. Клонируем репозиторий GIT:
```
    https://github.com/nosovdmitry007/jacet/tree/ADD_packet
```
2. Устанавливаем необхоимые библиотеки:
```
    pip install --upgrade pip
    pip install -r requirements.txt
```
3. Необходимо загрузить [веса](https://disk.yandex.ru/d/Ch_yYr4kvGhgEg) моделей в папку `yolo5`
```
├── model 
│   ├── about.txt
│   ├── chasha_v8.pt
│   ├── person_v8.pt
│   ├── stk_v8.pt
|   ├── truck_v8
```
## Пример:

### Детекция и клссификация

Запускаем распознавание данных для видео с возможностью формирования превью видео: 
```
    detection(video_file, al, cat, probability, classificator=0, clas_box=0, n=1)
    video_file -- путь к видеофайлу
    al --  0 - запускать отдельно по категории, 
           1 - запуск НС по всем категориям
    cat -- 'person' , 'chasha', 'truck', 'stk' 
    probability - 0 - не отображать точность модели предсказания на превью видео
                  1 - отображать точность модели предсказания на превью видео
    classificator -- 0 - не классифицировать людей после детекции
                     1 - классификация людей после детекции по 4 категориям ('Hat', 'Jacket', 'JacketAndHat', 'None')
    clas_box -- 0 - не отображать название класса на превью видео (если есть классификация)
                1 - отображать название класса на превью
    n -- кол-во кадров одновременно подоваемых на детекцию (зависит от возможностей компьютера)
    
```    

Пример вызова функции детекции людей, без классификации
```    
detection('./test/test.mp4', 0, 'person', 1, 0, 1, 5)
```
# Результат выполнения функции
На выходе получаем данные в pandas DF. Также формируется превью видео в каталоге с исходным файлом.

```

           xmin         ymin         xmax         ymax  confidence class  \
0   1131.225098  1118.168457  1234.933105  1225.512573    0.737279     0   
0   1131.433105  1118.231323  1234.431519  1224.986450    0.727406     0   
0   1130.979248  1118.059326  1234.557129  1225.303589    0.731094     0   
0   1130.672974  1118.441040  1234.416260  1225.629272    0.728179     0   
0   1131.116333  1118.653931  1234.319214  1225.085693    0.717301     0   
..          ...          ...          ...          ...         ...   ...   
0   1124.108521  1125.171021  1233.567139  1228.800293    0.571940     0   
0   1123.753784  1125.221558  1233.893311  1228.533569    0.557397     0   
0   1123.530640  1124.986084  1233.745605  1228.137329    0.561760     0   
0   1123.635010  1124.851074  1234.006836  1228.033325    0.543853     0   
0   1123.701050  1125.873413  1234.172485  1227.449341    0.496264     0   

    name   time_cadr  
0   person  0:00:00.00  
0   person  0-00-00.13  
0   person  0-00-00.20  
0   person  0-00-00.33  
0   person  0-00-00.40  
..   ...         ...  
0   person  0-00-29.73  
0   person  0-00-29.80  
0   person  0-00-29.93  
0   person  0:00:30.00  
0   person  0-00-30.13 
```