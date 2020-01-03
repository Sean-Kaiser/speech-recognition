import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import live_predictions as lp
import webrtcvad as vd



#Initialize PyAudio              
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000 #Sampling rate
CHUNK = 1000 #Datapoints to read at a time
pred_interval = 2 #Seconds to predict on
disp_interval = 10
frame_duration = 30 #ms to be used for VAD
samp_width = 2 #Used for VAD


#Create open stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)
audio_data = np.zeros((int(RATE*pred_interval)))
predictions = np.zeros(10)
vad = vd.Vad(1) #Voice activity detector with aggressiveness set


#Create Figure
t = np.linspace(0,9,10)
fig, ax = plt.subplots(figsize=(14,6))
x = np.linspace(0,9,10)
ax.set_ylim(-5,10)
line, = ax.step(x, np.zeros((len(predictions))))


#While continuously collecting data
while True:
    
    #Get audio data
    data = stream.read(CHUNK)
    current_data = np.frombuffer(data, np.int16)
        
    #Rotat data and fill
    audio_data = np.roll(audio_data, -CHUNK) #negative to roll opposite direction
    audio_data[-CHUNK:] = current_data
    
    #If voice activity then predict
    if vad.is_speech(data[-int(frame_duration/1000*RATE*samp_width):], RATE):
        
        #Predict on data
        prediction = lp.predict_2d(audio_data, RATE)
    
    else:       
        prediction = -1
        
    #Update Predictions
    predictions = np.roll(predictions, -1)
    predictions[-1] = prediction
    
    #Plot Prediction Outputs    
    line.set_ydata(predictions)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)    
    








