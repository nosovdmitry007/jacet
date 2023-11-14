
import platform
import cv2_ext
import imutils
import time

def time_of_function(function):
    def wrapped(*args):
        start_time = time.time()
        res = function(*args)
        print(f'Function {function.__name__!r}', time.time() - start_time)
        return res
    return wrapped
def slesh():
    sistem = platform.system()
    if 'Win' in sistem:
        sleh = '\\'
    else:
        sleh = '/'
    return sleh


def resize_img(image, size):
    if image.shape[0] > image.shape[1]:
        image = imutils.resize(image, height=size)
    else:
        image = imutils.resize(image, width=size)
    return image