
import pandas as pd
from tqdm import tqdm
from person import People, Kadr, sav, Jalet, Chasha

people = People()
jalet = Jalet()
chasha = Chasha()
kad = Kadr()


test = people.person_filter('2023-03-07_21-23-19.JPG')

print(test)


def person_filter_video(put):
    z = kad.cadre(put)
    df = pd.DataFrame()
    for i in tqdm(z, ncols=100):
        str = people.person_filter(i[0],i[1],'person',1)
        df = pd.concat([df, str])
        if len(str.name.unique()) != 0:
            sav(i[0], i[1], str,'test')
    return df


def jalet_filter_video(put):
    z = kad.cadre(put)
    df = pd.DataFrame()
    for i in tqdm(z, ncols=100):
        str = jalet.jalet_filter(i[0],i[1],1)
        df = pd.concat([df, str])
        if len(str.name.unique()) != 0:
            sav(i[0], i[1], str,'test')
    return df

def chasha_filter_video(put):
    z = kad.cadre(put)
    df = pd.DataFrame()
    for i in tqdm(z, ncols=100):
        str = chasha.chasha_filter(i[0],i[1],1)
        df = pd.concat([df, str])
        if len(str.name.unique()) != 0:
            sav(i[0], i[1], str,'test')

    return df

print(person_filter_video('test_1.mp4'))