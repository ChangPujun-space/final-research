import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.preprocessing import image


import os
train_img = []
train_label = []
for i in range(1,101):
    temp_path = "video_img_data/"+str(i)
    files= os.listdir(temp_path)
    for file in files[:-3]:
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
train_img =train_img.reshape((len(train_img), np.prod(train_img.shape[1:])))
print("train_img:",len(train_img))



encoding_dim = 200  ##compress dimension
input_img = Input(shape=(150528,)) ##input placeholder

#encoder layers
encoded = Dense(500, activation = 'relu')(input_img)
encoded = Dense(300, activation = 'relu')(encoded)
encoded_output = Dense(encoding_dim,)(encoded)

decoded = Dense(300, activation = 'relu')(encoded_output)
decoded = Dense(500, activation = 'relu')(decoded)
decoded = Dense(150528, activation = 'tanh')(decoded)

##construct the autoencoder model
autoencoder = Model(input = input_img, output = decoded)

##compile autoencoder
autoencoder.compile(optimizer = 'adam', loss = 'mse')



autoencoder.fit(train_img, train_img, epochs = 20, batch_size = 256, shuffle = True)

from keras.models import save_model
autoencoder.save("output/autoencoder.h5")
