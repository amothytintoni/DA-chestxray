from IPython.core.interactiveshell import InteractiveShell
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking
from keras.models import Model, Sequential
from glob import glob
from tqdm import tqdm
from scipy.stats import uniform
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.preprocessing import image
from keras.applications import xception
import os
import datetime as dt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')


InteractiveShell.ast_node_interactivity = "all"
# masking function


def create_mask_for_image(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([0, 0, 250])
    upper_hsv = np.array([250, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# image  deskew function


def deskew_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255

# image  gray  function


def gray_image(image):
    mask = create_mask_for_image(image)
    output = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
    return output/255

# image  thresh  function


def thresh_image(image):
    img = read_img(df['file'][250], (255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)  # +cv.THRESH_OTSU)
    return output


# image  rnoise  function
def rnoise_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255

# image  dilate  function


def dilate_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255


# image  erode  function
def erode_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255


# image  opening  function
def opening_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255

# image canny function


def canny_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255


# image segmentation function
def segment_image(image):
    mask = create_mask_for_image(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output/255


# sharpen the image
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


# function to get an image
def read_img(filepath, size):
    img = image.load_img(os.path.join(data_kaggle, filepath), target_size=size)
    # convert image to array
    img = image.img_to_array(img)
    return img
