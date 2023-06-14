# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:37:34 2023

@author: Timothy
"""

from PIL import Image
import PIL
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_pt_id(n):
    return n[:8]


nihkagimpath320 = os.path.join(os.getcwd(), 'nih-kaggle', 'images')
# nih14impath = os.path.join(os.getcwd(), 'NIH-14')
df = pd.read_csv('nih-kaggle/NIH_Original label_pp_use this.csv')

# file_list = []
# p = []
# for paths, _, files in os.walk(nihkagimpath320):
#     for file in files:
#         file_list.append(os.path.join(paths, file))

train_df = df[df['is_train_val'] == True]
test_df = df[df['is_train_val'] == False]

train_df.loc[:, 'pt_id'] = train_df.loc[:, 'Image Index'].apply(find_pt_id)
test_df.loc[:, 'pt_id'] = test_df.loc[:, 'Image Index'].apply(find_pt_id)

train_pt_list = pd.unique(train_df['pt_id'])
test_pt_list = pd.unique(test_df['pt_id'])

train_labels = train_df[train_df.columns[-15:-1]]
test_labels = test_df[test_df.columns[-15:-1]]

train_freq = train_labels.sum().to_numpy()
test_freq = test_labels.sum().to_numpy()

train_freq_norm = train_freq/train_freq.max()
test_freq_norm = test_freq/test_freq.max()

# plt.figure(1)
# plt.bar(train_labels.columns, train_freq_norm, fill = False, edgecolor = 'blue')
# plt.bar(test_labels.columns, test_freq_norm, fill = False, edgecolor = 'red')
# plt.xticks(rotation = 90)
# plt.title('Train and Test set Label Distribution')
# plt.legend()
# plt.show()

x = np.arange(len(train_labels.columns))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

# for attribute, measurement in penguin_means.items():
for i in range(len(train_labels.columns)):
    for item in ('train', 'test'):
        offset = width * multiplier
        rects = ax.bar(x + offset, f'{item}_freq_norm'[i], width)  # , label=item)
        ax.bar_label(rects, padding=3)
        multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, train_labels.columns)
ax.legend(loc='upper left')  # , ncols=3)
# ax.set_ylim(0, 250)

# check patient leakage
# for item in train_pt_list:
#     if item in test_pt_list:
#         print('LEAKAGE!')
#         break
# proven no leakage
