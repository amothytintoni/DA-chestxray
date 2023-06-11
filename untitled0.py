import pandas as pd
import matplotlib.pyplot as plotter_lib
import pathlib
import numpy as np

import PIL as image_lib
import sklearn
import tensorflow as tflow

from tensorflow.keras.layers import Flatten

from keras.layers.core import Dense

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

from tensorflow.keras.optimizers import Adam
import os
import sklearn

import pathlib

demo_dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

r50model2 = ResNet50V2(
)

# directory = tflow.keras.utils.get_file('flower_photos', origin=demo_dataset, untar=True)

# data_directory = pathlib.Path(directory)
