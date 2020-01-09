import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from useful_functions import acc, record_with_vad
from keras.layers import Conv2D
from librosa.feature import mfcc



#Load Filenames
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"
filenames = os.listdir(path)

#Find Longest segment of data
interval = 1 #window length to predict on
fs = 8000 #sampling of audio
length = int(interval*fs)
    

#Format Data
X = np.zeros((len(filenames)-100, length, 1))
y = np.zeros((len(X)))

X_val = np.zeros((100, int(length), 1))
y_val = np.zeros((100))

index=0
indy=0
for i, _ in enumerate(filenames):
    _, audio = read(path+filenames[i])
    num = filenames[i][0]
    if num == 's':
        num = 10
        
    if filenames[i][2] == 's':
        if len(audio) < length:
            X_val[index,:len(audio),0] = audio[:length]
        else:
            X_val[index, :,0] = audio[:length]
        y_val[index] = int(num)
        index += 1
    else:
        X[indy,:len(audio),0] = audio[:length]
        y[indy] = int(num)
        indy+=1
    

#2D Classification using librosa mfcc
X_new = np.empty((len(X), 20, int(16*interval), 1))
for i, data in enumerate(X):        
    #Assumes being passed X
    data = data[:,0]    
    mf_data = mfcc(data, fs)   
    X_new[i, :, :, 0] = mf_data

X_new_val = np.empty((len(X_val), 20, int(16*interval), 1))
for i, data in enumerate(X_val):        
    #Assumes being passed X
    data = data[:,0]    
    mf_data = mfcc(data, fs)   
    X_new_val[i, :, :, 0] = mf_data
 
    
    
    
    
    
    
#Classification
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

#Make y categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Classification
classifier = Sequential()
classifier.add(Conv2D(filters=30, kernel_size=(5,5), strides=(2,2), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])))

classifier.add(Flatten())

classifier.add(Dense(units=100))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(11, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classifier.fit(X_train, y_train, epochs=10, batch_size=50)    

pred = classifier.predict(X_test)
y_pred = pred.argmax(1)
   
cm = confusion_matrix(y_test.argmax(1), pred.argmax(1)) #Check predictions with confusion matrix
print(cm)
print(acc(cm))

#Validation Set
y_val = np_utils.to_categorical(y_val)
pred_val = classifier.predict(X_new_val)

cm = confusion_matrix(y_val.argmax(1), pred_val.argmax(1)) #Check predictions with confusion matrix
print(cm)
print(acc(cm))







model_json = classifier.to_json()
with open("digit_classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("digit_classifier.h5")
print("Saved model to disk")







"""Testing Classifier on Incoming Data"""

#Record test data
duration=5
data, vad_output, t_vad = record_with_vad(duration, fs)
data = data.astype(np.float)

#Graph how outputs change over data
CHUNK = 1000 #Chosen from main script

X_output = np.zeros((1,X_new.shape[1],X_new.shape[2],X_new.shape[3]))
outputs = np.zeros((int((len(data)-fs*interval)/CHUNK), pred.shape[1]))

out_index=0
for i in range(int(fs*interval),len(data),CHUNK):
    pred_data = data[int(i-fs*interval):i]        
    mf_data = mfcc(pred_data, fs)   
    X_output[0, :, :, 0] = mf_data    
    
    preds = classifier.predict(X_output)

    outputs[out_index,:] = preds
    out_index += 1

t = np.linspace(0,len(data), len(data))
x = np.linspace(int(fs*interval), len(data), len(outputs)+1)
x = x[:len(outputs)]



for h in range(len(vad_output)):
    if vad_output[h] != vad_output[h-1]:
        ind = h
        break


#Plot audio, vad, and predictions
plt.figure()
plt.subplot(311)
plt.plot(t, data)

plt.subplot(312)
plt.plot(t_vad, vad_output)
plt.axvline(x=(h*80 + 4000), color='r')

plt.subplot(313)
for i in range(outputs.shape[1]):
    plt.plot(x, outputs[:,i], label=str(i))
    plt.xlim(0,len(data))
plt.axvline(x=(h*80+4000), color='r')
plt.legend()


#Logic to obtain prediction
prediction = -1
sms = np.zeros((outputs.shape[1]-1))
for k in range(outputs.shape[1]-1):
    if np.max(outputs[:,k]) > 0.95:
        prediction=k
    else:
        sms[k] = np.sum(outputs[:,k])
    print("Sum of "+str(k)+" = "+str(np.sum(outputs[:,k])))
if prediction == -1:
    prediction = np.argmax(sms)



print("Method 1 Prediction = ",prediction)


