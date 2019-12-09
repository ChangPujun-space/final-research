from PIL import Image
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
from keras.applications.vgg19 import decode_predictions
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

import os
img = []
label = []
for i in range(1,401):
    temp_path = "video_img_data/"+str(i)
    files= os.listdir(temp_path)
    for file in files:
        if ('.jpg' in file) is True:
            try:
                temp_img = image.load_img(temp_path+'/'+file,target_size=(224,224))
                temp_img=image.img_to_array(temp_img)
            except:
                print(i,file)
            else:
                img.append(temp_img)
                label.append(i)


all_img=np.array(img) 
all_img=preprocess_input(all_img)

all_y=np.asarray(label)
le = LabelEncoder()
all_y = le.fit_transform(all_y)
all_y=to_categorical(all_y)
all_y=np.array(all_y)

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid=train_test_split(all_img,all_y,test_size=0.2, random_state=42)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from keras.models import Model

def vgg19_model(img_rows, img_cols, channel=1, num_classes=None):
    model = VGG19(weights='imagenet', include_top=True)

    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    model.layers[-1].outbound_nodes = []

    x=Dense(num_classes, activation='softmax')(model.output)

    model=Model(model.input,x)

    for layer in model.layers[:8]:

        layer.trainable = False
    # Learning rate is changed to 0.001
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 400 
batch_size = 16 
nb_epoch = 20
model = vgg19_model(img_rows, img_cols, channel, num_classes)

model.fit(X_train, Y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1)
model.save("output/save_model_video_img.h5")

predictions_valid = model.predict(X_valid, batch_size=16, verbose=1)




score = log_loss(Y_valid,predictions_valid)
print(score)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn



predictions_valid = np.argmax(predictions_valid, axis=1)

Y_valid = np.argmax(Y_valid, axis=1)

confmat = confusion_matrix(Y_valid, predictions_valid)
print(accuracy_score(Y_valid, predictions_valid))
print(Y_valid)
print(predictions_valid)
df_cm = pd.DataFrame(confmat)
sn.set(font_scale=1.4)
sn_plot = sn.heatmap(df_cm,annot=True,annot_kws={"size":16})
fig = sn_plot.get_figure()
fig.savefig("output/matrix.png")


