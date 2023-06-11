from PIL import Image
import PIL
import os
import shutil
import pandas as pd
import numpy as np

nihkagimpath = os.path.join(os.getcwd(), 'nih-kaggle', 'images', 'images')
nih14impath = os.path.join(os.getcwd(), 'NIH-14')
df = pd.read_csv('nih-kaggle/NIH_Original label_pp_use this.csv')

file_list = []
p = []
for paths, _, files in os.walk(nihkagimpath):
    for file in files:
        file_list.append(os.path.join(paths, file))

Image.open(file_list[0]).save(file_list[0])

# print(file_list[0][:-23], file_list[0][-16:])
for im in file_list:
    new_im = Image.open(im).resize((320, 320))
    new_im.save(f'{im[:-23]}{im[-16:]}.png')

#     # if im.endswith('00000003_000.png'):
#     #     # print(im)
#     #     im2 = Image.open(im).resize((320,320))
#     # for i in range(0,5):

# im1 = Image.open(os.path.join(nih14impath,'images','test','00000003_000.png'))
# # im2 = Image.open(os.path.join(nihkagimpath, 'images', 'images', '00000003_000.png'))


# a = np.equal(im2, im1)
# im2_ar = np.array(im2)/255
# im1_ar = np.array(im1)/255
# print(np.sum(im1_ar-im2_ar))

# print(np.max(im2_ar))
# # im2.show()
# # im1.show()
