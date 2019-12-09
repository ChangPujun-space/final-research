from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np
import os




train_img = []
train_label = []
for i in range(1,351):
    temp_path = "video_img_data/"+str(i)
    files= os.listdir(temp_path)
    for file in files[3:-3]:
        if ('.jpg' in file) is True:
            try:
                temp_img = image.load_img(temp_path+'/'+file,target_size=(224,224))
            except:
                print(i,file)
            else:
                temp_img=image.img_to_array(temp_img)
                train_img.append(temp_img)
                train_label.append(i)


train_img = np.array(train_img)
print(train_img[1])
train_img = train_img.astype('float32') / 255
train_img = np.reshape(train_img, (len(train_img), 224, 224, 3))
print(train_img.shape)

input_img = Input(shape=(224,224,3))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)


# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')



autoencoder.fit(train_img, train_img, epochs = 50, batch_size = 64, shuffle = True)

from keras.models import save_model
autoencoder.save("output/CNN_autoencoder.h5")
