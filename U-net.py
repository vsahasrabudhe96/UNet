from keras.layers import *
from keras.models import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import cv2
import PIL

IMG_WIDTH = 128
IMG_HEIGHT=128
IMG_CHANNEL = 3


inputs = Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL))
s = Lambda(lambda x:x/255.0)(inputs)


# FIRST STAGE
c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPool2D((2,2))(c1)

c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPool2D((2,2))(c2)

c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.1)(c3)
c3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPool2D((2,2))(c3)

c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.1)(c4)
c4 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPool2D((2,2))(c4)

c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.1)(c5)
c5 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


# EXPANSIVE PATH
u6 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(c5)
u6 = concatenate([u6,c4])
c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.1)(c6)
c6 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(c6)
u7 = concatenate([u7,c3])
c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.1)(c7)
c7 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(c7)
u8 = concatenate([u8,c2])
c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(c8)
u9 = concatenate([u9,c1],axis=3)
c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1,(1,1),activation='sigmoid')(c9)

model = Model(inputs=[inputs],outputs=[outputs])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuarcy'])

print(model.summary())