# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:46:15 2023

@author: Timothy
"""

import matplotlib.pyplot as plt
import PIL
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator


def gray_to_rgb(img):
    return np.repeat(img, 3, 2)

# a = PIL.Image.open('NIH-14/images/train/00000001_000.png')
# aa = a.resize((320,320))
# aa_arr = np.array(aa)

# aaa = np.repeat(aa_arr[..., np.newaxis], 1, -1)
# aaaa= gray_to_rgb(aaa)
# # aaa[:,:,1:] = 0
# aaaa = PIL.Image.fromarray(aaa)
# # aa.save('NIH-14/images/train/00000001_000_r.png')


a = pd.DataFrame(data=[['tr.png', 0], ['tr2.png', 0], ['te.png', 1], [
                 'te2.png', 1],], columns=['filename', 'label'])
atr = a.iloc[[0, 1], :]
ate = a.iloc[[2, 3], :]
# display(aaaa)
traingen = ImageDataGenerator(rescale=1./255,)
trgen = traingen.flow_from_dataframe(
    atr, x_col='filename', y_col='label', directory='NIH-14/dummy/train', target_size=(224, 224), color_mode='rgb', save_to_dir='./NIH-14/dummy', class_mode='raw')

for i in range(0, 1):
    aitem = trgen.next()
