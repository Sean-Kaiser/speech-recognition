#K nearest neighbors digit classification with time warping

import numpy as np
import os
from scipy.io.wavfile import read
from sklearn.metrics import confusion_matrix
from useful_functions import acc
from librosa.feature import mfcc
from fastdtw import fastdtw


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
    num = filenames[i][0]
    if num == 's':
        num = 10
    y[i] = int(num)


#Obtain Mfccs
X_new = np.empty((len(X), 20, 32))
for i, data in enumerate(X):        
    #Assumes being passed X
    data = data[:,0]    
    mf_data = mfcc(data, fs)   
    X_new[i, :, :] = mf_data    
    

#Get Template Values
num_labels = len(set(y))
labels = np.zeros((num_labels, 2))
labels[:,0] = np.linspace(num_labels-1, 0, num_labels)

X_template = np.zeros((num_labels, X_new.shape[1], X_new.shape[2]))
y_template = np.zeros((num_labels))

index = 0
for i in reversed(range(len(y))):
    if y[i] == labels[index,0] and not labels[index,1]:
        labels[index,1] = 1
        
        X_template[index,:,:] = X_new[i,:,:]
        X_new = np.delete(X_new, i, axis=0)
        
        y_template[index] = y[i]
        y = np.delete(y, i, axis=0)
        
        index += 1
        if index == num_labels:
            break
X_template = np.flip(X_template, axis=0)
y_template = np.flip(y_template, axis=0)        

#Classify in least distance
y_pred = np.zeros((len(X_new)))
    
for s in range(len(X_new)): #For all values of X
    
    distances = np.empty((num_labels))
    
    inter_dists = np.zeros((X_template.shape[0], X_template.shape[1])) #Comparisons for classification   
    
    for i in range(inter_dists.shape[0]): #For all labels
        for j in range(inter_dists.shape[1]): #For all frequency sequences
            dist, _ = fastdtw(X_template[i,j,:], X_new[s,j,:]) #Compute distance
            
            inter_dists[i,j] = dist #Add to list of distances
            
    for k, _ in enumerate(distances): #For all lables
        distances[k] = np.sum(inter_dists[k,:]) #Compute the sum of distances

    y_pred[s] = np.argmin(distances) #Find label with smallest distance
    
    if s % 10 == 0:
        print(s)
    
cm = confusion_matrix(y, y_pred) #Check predictions with confusion matrix
print(cm)
print(acc(cm))    
















