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

#Load Filenames
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"
filenames = os.listdir(path)

#Find Longest segment of data
min_index = 0
length = 15000
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







plt.plot(sz)







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
classifier.add(Conv1D(filters=30, kernel_size=50, strides=25, input_shape = (X_train.shape[1], X_train.shape[2])))
classifier.add(Activation('relu'))

classifier.add(MaxPooling1D(pool_size=4))

classifier.add(Flatten())

classifier.add(Dense(units=200))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.1))

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, epochs=40, batch_size=20)    

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




