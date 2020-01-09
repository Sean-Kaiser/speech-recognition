import numpy as np
from keras.models import model_from_json
from librosa.feature import mfcc


# load json and create model
json_file = open('digit_classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("digit_classifier.h5")
print("Loaded model from disk")


def predict_1d(data, fs, interval):
    
    X = np.zeros((1,len(data),1))
    X[0,:,0] = data
    
    return classifier.predict(X).argmax(1)
    #return random.choice([0,1,2,3,4,5,6,7,8,9])

def predict_2d(data, fs, interval):
    
    X = np.zeros((1,20,int(16*interval),1)) #Length of mfcc from training
    
    X[0,:,:,0] = mfcc(data, fs)
    
    return classifier.predict(X).argmax(1)