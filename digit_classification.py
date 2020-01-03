import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from keras.layers import Conv1D, MaxPooling1D, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from useful_functions import acc
from keras.models import model_from_json

#Load Filenames
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"
filenames = os.listdir(path)

#Find Longest segment of data
min_index = 0
interval = 2 #window length to predict on
fs = 8000 #sampling of audio
length = interval*fs
min_size = 10000
sz = np.zeros((length))

for i, name in enumerate(filenames):
    sz[i] = os.stat(path+filenames[i]).st_size
    if min_size > sz[i]:
        min_size = sz[i]
        min_index = i
    
fs_aud, data_min = read(path+filenames[min_index])
min_length = len(data_min)

#Format Data
X = np.zeros((len(filenames), length, 1))
for i, _ in enumerate(filenames):
    _, audio = read(path+filenames[i])
    X[i,:len(audio),0] = audio[:length]


#Obtain labels
y = np.zeros((len(X)))
for i,_ in enumerate(filenames):
    y[i] = int(filenames[i][0])







#Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Normalize & Categorical
X_train = X_train / np.max(np.abs(X_train))
X_test = X_test / np.max(np.abs(X_train))

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

classifier = Sequential()
classifier.add(Conv1D(filters=30, kernel_size=2, activation = 'relu', input_shape = (X_train.shape[1], X_train.shape[2])))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Flatten())

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, epochs=20, batch_size=100)    

pred = classifier.predict(X_test)
y_pred = pred.argmax(1)
   
cm = confusion_matrix(y_test.argmax(1), pred.argmax(1)) #Check predictions with confusion matrix
print(cm)
print(acc(cm))



# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




# serialize model to JSON
model_json = classifier.to_json()
with open("digit_classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("digit_classifier.h5")
print("Saved model to disk")













#2D Classification using librosa mfcc
from keras.layers import Conv2D, MaxPooling2D
from librosa.feature import mfcc


X_new = np.empty((len(X), 20, 32, 1))

for i, data in enumerate(X):
        
    #Assumes being passed X, y
    data = data[:,0]
    
    mf_data = mfcc(data, fs)
    
    X_new[i, :, :, 0] = mf_data


#Classification
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

#Normalize & Categorical
#X_train = X_train / np.max(np.abs(X_train))
#X_test = X_test / np.max(np.abs(X_train))

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

classifier = Sequential()
classifier.add(Conv2D(filters=30, kernel_size=(5,5), strides=(2,2), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))

#classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, epochs=20, batch_size=50)    

pred = classifier.predict(X_test)
y_pred = pred.argmax(1)
   
cm = confusion_matrix(y_test.argmax(1), pred.argmax(1)) #Check predictions with confusion matrix
print(cm)
print(acc(cm))














