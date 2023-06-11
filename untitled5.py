# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:37:05 2023

@author: Timothy
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm


def divbyX(a, x):
    return a % x


div_grp = 10
a = []

for paths, _, files in os.walk(os.path.join(os.getcwd(), 'NIH-14', 'images', 'train')):
    for file in files:
        a.append(os.path.join(paths, file))


adf = pd.DataFrame(a, columns=['val']).groupby(lambda y: divbyX(y, div_grp))

# print(str(0))

# bdf = adf.get_group(6)

# print(a[0][:-16])
# print(a[0][-16:])

for i in tqdm(range(1, div_grp)):
    bdf = adf.get_group(i)['val']
    for item in bdf:
        shutil.move(item, os.path.join(item[:-16], str(i), item[-16:]))

# bdf = adf.get_group(0)['val']
# print(type(bdf[0]))
# for item in bdf:
#     shutil.move(item, os.path.join(item[:-16],str(0),item[-16:]))

# cwd = os.getcwd()
# for i in range(div_grp):
#     os.mkdir(os.path.join(cwd, 'NIH-14', 'images', 'train', f'{i}'))
