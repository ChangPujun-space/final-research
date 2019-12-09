from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import numpy as np
import os

test_img = []
test_label = []
pic_count = 0
for i in range(1,351):
    temp_path = "video_img_data/"+str(i)
    files= os.listdir(temp_path)
    for file in files[5:15]:
        if ('.jpg' in file) is True:
            try:
                temp_img = image.load_img(temp_path+'/'+file,target_size=(224,224))
            except:
                print(i,file)
            else:
                temp_img=image.img_to_array(temp_img)
                test_img.append(temp_img)
                test_label.append(str(i)+"_"+str(file))
                pic_count = pic_count+1

test_img = np.array(test_img)
test_img = test_img.astype('float32') / 255.
x_test = np.reshape(test_img, (len(test_img), 224, 224, 3))

from keras.models import load_model
model = load_model('output/CNN_autoencoder.h5')

layer_name = 'conv2d_4'
hidden_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
hidden_output = hidden_layer_model.predict(x_test)
print(hidden_output.shape)


list_a = []

for z in range(pic_count):
    L_e = hidden_output[z]
    layer_vector = L_e.reshape(25088,)
    layer_vector.tolist()
    list_a.append(layer_vector)

import pickle
fileObject = open('output/Feature_Vector_List.txt', 'wb')
pickle.dump(list_a,fileObject)
fileObject.close()


fileObject = open('output/Feature_label_List.txt', 'wb')
pickle.dump(test_label,fileObject)
fileObject.close()
