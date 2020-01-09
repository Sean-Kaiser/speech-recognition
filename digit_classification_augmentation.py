import numpy as np
import os
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from useful_functions import acc
from keras.layers import Conv2D
from librosa.feature import mfcc
import useful_functions as uf


#Load Filenames
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"
filenames = os.listdir(path)


#Find Longest segment of data
interval = 2 #window length to predict on
fs = 8000 #sampling of audio
length = int(interval*fs)
    

#Find Silence Noise Data
X_sil = np.zeros((200, int(length), 1))

index_silence = 0
for i, _ in enumerate(filenames):
    _, audio = read(path+filenames[i])
    if filenames[i][0] == 's':        
        X_sil[index_silence, :,0] = audio[:length]
        index_silence += 1


#Format Data and Augment with Silence Data
X = np.zeros((len(filenames)-200, length, 1))
y = np.zeros((len(X)))

index = 0
for i, _ in enumerate(filenames):
    num = filenames[i][0]
    if num != 's': 
        _, audio = read(path+filenames[i]) #Read audio data
        
        X[index,:,0] = uf.augment(audio, length)
        
        y[index] = int(num)
        index += 1


#2D Classification using librosa mfcc
X_new = np.empty((len(X), 20, int(16*interval), 1))
for i, data in enumerate(X):        
    #Assumes being passed X
    data = data[:,0]    
    mf_data = mfcc(data, fs)   
    X_new[i, :, :, 0] = mf_data

                
  
#Classification
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

#Make y categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Classification
classifier = Sequential()
classifier.add(Conv2D(filters=30, kernel_size=(2,2), strides=(1,1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))

print(classifier.output_shape)
classifier.add(Flatten())

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, epochs=15, batch_size=50)    

pred = classifier.predict(X_test)
y_pred = pred.argmax(1)
   
cm = confusion_matrix(y_test.argmax(1), pred.argmax(1)) #Check predictions with confusion matrix
print(cm)
print(acc(cm))







model_json = classifier.to_json()
with open("digit_classifier_aug.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("digit_classifier_aug.h5")
print("Saved model to disk")





