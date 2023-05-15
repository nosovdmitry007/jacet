# Inspired from https://github.com/JohannesBuchner/imagehash repository
import os
from PIL import Image
import imagehash
import pandas as pd

def dubli(put):
    df = pd.DataFrame()
    path = os.listdir(put)
    k = path.copy()

    for i in path:
        k.remove(i)
        slov = {}
        filename = []
        hashes = []
        otherhashes = []
        hamming_distance = []
        fil_h = []
        # print(len(k))
        for j in k:
            hash = imagehash.average_hash(Image.open(f'{put}/{i}'))
            otherhash = imagehash.average_hash(Image.open(f'{put}/{j}'))

            hamming_distances = hash - otherhash  # hamming distance
            if hamming_distances < 8:
                filename.append(i)
                fil_h.append(j)
                hashes.append(str(hash))
                otherhashes.append(str(otherhash))
                hamming_distance.append(hamming_distances)

        slov['filename'] = filename
        slov['filename_out'] = fil_h
        slov['hash'] = hashes
        slov['otherhash'] = otherhashes
        slov['hamming_distance'] = hamming_distance
        df_dictionary = pd.DataFrame(slov)
        df = pd.concat([df, df_dictionary], ignore_index=True)
        # print(df)
        df.drop_duplicates(subset=['filename_out'])
    print(df.filename_out.drop_duplicates())
    df.to_csv('./Данные для обучения нейросети/классификация/dataset/baza/dubli_hat.csv', encoding='utf-8')

dubli('./Данные для обучения нейросети/классификация/dataset/baza/Hat')