from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

datagen = ImageDataGenerator(
        rotation_range=20,   #Random rotation between 0 and 45
        width_shift_range=0.05,   #% shift
        height_shift_range=0.1,
        brightness_range=(0.5, 1),
        shear_range=0.07,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='reflect') #, cval=125)    #Also try nearest, constant, reflect, wrap

i = 0
for batch in datagen.flow_from_directory(directory='Dataset/Dataset3/Side',
                                         batch_size=41,
                                         target_size=(720, 1280),
                                         color_mode="rgb",
                                         save_to_dir='Dataset/Dataset3/Augmented/Side',
                                         save_prefix='Side',
                                         save_format='png'):
    i += 1
    if i > 12:
        break