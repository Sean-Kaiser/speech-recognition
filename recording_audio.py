import sounddevice as sd
from scipy.io.wavfile import write, read
import os
import numpy as np

fs = 8000 #sampling freq
sec = 5 #recording duration
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"


for i in range(0,10):
    
    rec = sd.rec(int(sec*fs), samplerate=fs, channels=1)
    print("Speak "+str(i))
    sd.wait()
    write('testing.wav', fs, rec)
    write(path+'9_sean_'+str(i)+'.wav', fs, rec)



"""SNIPPING SEGMENTS"""
noise_thresh = 0.0005
buffer = 150

#Load Filenames
path = "C:\\Users\\Sean K\\Documents\\Python-Projects\\speech-recognition\\free-spoken-digit-dataset\\recordings\\"
filenames = os.listdir(path)

for i, _ in enumerate(filenames):
    if filenames[i][2:6] == 'sean':
    
        _, audio = read(path+filenames[i])
        indices = np.zeros((2))
        for p, amp in enumerate(audio):
            if abs(amp) > noise_thresh:
                indices[0] = p
                break
        for q in reversed(range(len(audio))):
            if abs(audio[q]) > noise_thresh:
                indices[1] = q
                break    
            
        new_audio = audio[p-buffer:q+buffer]       
        write(path+filenames[i], fs, new_audio)
        

    