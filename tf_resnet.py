
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd

SEED = 1

# tunables
metrics = ['accuracy']
learning_rate = 0.001
output_class = 14
epochs = 10
batch_size = 32
hidden_layer_size = [256,]
input_shape_gs = (224, 224, 1)
input_shape_rgb = (224, 224, 3)

# functions


def gray_to_rgb(img):
    return np.repeat(img, 3, 2)


nih_folder = os.path.join(os.getcwd(), 'NIH-14')
df = pd.read_csv(os.path.join(nih_folder, 'labels', 'NIH_Original label_pp_use this.csv'))
# labels = df.iloc[:, -14:]
train_df = df[df['is_train_val'] == True]
test_df = df[df['is_test'] == True]
train_labels = train_df.iloc[:, -14:]
test_labels = test_df.iloc[:, -14:]


# Load the pre-trained ResNet50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_rgb)

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(output_class, activation='sigmoid'))  # Assuming 14 output classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy', metrics=metrics)

# Data augmentation and preprocessing
# preprocessing_function=gray_to_rgb)
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255,)  # preprocessing_function=gray_to_rgb)

# Set the path to dataset

im_folder = os.path.join(os.getcwd(), 'NIH-14', 'images')

# Set the path to your training and test datasets
train_dir = os.path.join(im_folder, 'train')
test_dir = os.path.join(im_folder, 'test')


train_steps = sum(df['is_train_val']) // batch_size
val_steps = sum(df['is_test']) // batch_size

# # Create the image generators
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(
#     224, 224), batch_size=batch_size, class_mode='multi_label')
# test_generator = test_datagen.flow_from_directory(test_dir, target_size=(
#     224, 224), batch_size=batch_size, class_mode='multi_label')

train_generator = train_datagen.flow_from_dataframe(train_df, directory=train_dir, x_col='Image Index', y_col=list(train_labels.columns), target_size=(
    224, 224), batch_size=batch_size, class_mode='raw', color_mode='rgb', seed=SEED)

test_generator = test_datagen.flow_from_dataframe(test_df, directory=test_dir, x_col='Image Index', y_col=list(test_labels.columns), target_size=(
    224, 224), batch_size=batch_size, class_mode='raw', color_mode='rgb', seed=SEED)

# Train the model
model.fit(train_generator, steps_per_epoch=train_steps, epochs=epochs,
          validation_data=test_generator, validation_steps=val_steps)

# Save the trained model
model.save('resnet50_model.h5')
