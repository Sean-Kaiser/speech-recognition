import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import live_predictions as lp
import webrtcvad as vd


"""Initialize"""
#Initialize PyAudio              
p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000 #Sampling rate
CHUNK = 2000 #Datapoints to read at a time

#Create open stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK) #Incoming audio stream

pred_interval = 1 #Seconds to predict on
plot_interval = 2 #Seconds to graph
frame_duration = 10 #ms to be used for VAD
samp_width = 2 #Used for VAD


audio_data = np.zeros((int(RATE*plot_interval))) #Length of audio data to predict on from digit classification
preds_per_plot = int(RATE*plot_interval/CHUNK)
predictions = np.zeros((preds_per_plot)) #Number of predictions to display

vad = vd.Vad(3) #Voice activity detector with aggressiveness set
datapoints_per_vad = int(samp_width * frame_duration / 1000 * RATE) #Number of byte datapoints needed to pass vad
vad_output = np.zeros((int(plot_interval*RATE*samp_width/datapoints_per_vad)))
num_vads = int(CHUNK*samp_width / datapoints_per_vad) #Number of Vad predictions per chunk
min_thresh = len(vad_output)/8 #Thresholds for length of speech
max_thresh = min_thresh*6
window = pred_interval*RATE/(datapoints_per_vad/samp_width)


"""Create Plots"""
#Create Figure
fig = plt.figure()

#First figure = audio plot
ax1 = fig.add_subplot(311)
x1 = np.linspace(0, RATE*plot_interval, RATE*plot_interval)
ax1.set_ylim(-500, 500)
ax1.set_xlim(0, int(RATE*plot_interval)) #make sure our x axis matched our chunk size
line1, = ax1.plot(x1, np.random.rand(RATE*plot_interval))

#Second figure = vad output
ax2 = fig.add_subplot(312)
x2 = np.linspace(datapoints_per_vad/samp_width, int(RATE*plot_interval), len(vad_output))
ax2.set_ylim(-1,2)
ax2.set_xlim(0, int(RATE*plot_interval)) #make sure our x axis matched our chunk size
line2, = ax2.plot(x2, np.zeros((len(vad_output))))

#Third figure = prediction output
ax3 = fig.add_subplot(313)
x3 = np.linspace(0, RATE*plot_interval, len(predictions))
ax3.set_ylim(-2,10)
ax3.set_xlim(0, int(RATE*plot_interval)) #make sure our x axis matched our chunk size
line3, = ax3.step(x3, np.zeros((len(predictions))))




"""Run Audio Processing"""
#While continuously collecting data
while True:
    
    #Get audio data
    data = stream.read(CHUNK) #Acquire data from stream
    current_data = np.frombuffer(data, np.int16) #Convert data to numpy array
        
    #Rotate data and fill
    audio_data = np.roll(audio_data, -CHUNK) #negative to roll opposite direction
    audio_data[-CHUNK:] = current_data #Update latest audio data with current data
    
    
    #VAD Processing
    vad_output = np.roll(vad_output, -num_vads)
    vad_index = 1
    for i in range(datapoints_per_vad,int(CHUNK*samp_width+datapoints_per_vad), datapoints_per_vad):
        if vad.is_speech(data[int(i-datapoints_per_vad):i], RATE):
            vad_output[-vad_index] = 1
        else:
            vad_output[-vad_index] = 0
        vad_index+=1
    
    
    #Finding audio segment to predict on
    ind = np.zeros((2))
    for q in range(len(vad_output)-1):       
        if vad_output[q] == 0 and vad_output[q+1] == 1:
            ind[0] = q
            break
        
    for p in range(1, len(vad_output)-1):
        if vad_output[-p] == 1:
            p = len(vad_output) - p
            ind[1] = p
            break
    
    #If it hasn't predicted already
    to_pred = 1
    for u in range(len(predictions)):
        if predictions[u] != -1:
            to_pred = 0
            break
    
    if to_pred:
        if ind[1]-ind[0] > min_thresh and ind[1]-ind[0] < max_thresh and ind[0] < len(vad_output) - window:
            pred = lp.predict_2d(audio_data, RATE, plot_interval)
            #pred = lp.predict_2d(audio_data[int(ind[0]*datapoints_per_vad/samp_width):int(ind[0]*datapoints_per_vad/samp_width + pred_interval*RATE)], RATE, pred_interval)
            fig.suptitle("You said "+str(pred))
            print(pred)
            
    else:
        pred = -1 #Set arbitrary value
        fig.suptitle("Not Speaking")        
    
           
    #Update Predictions
    predictions = np.roll(predictions, -1)
    predictions[-1] = pred
    
    #Plot Prediction Outputs   
    line1.set_ydata(audio_data)
    line2.set_ydata(vad_output)
    line3.set_ydata(predictions)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)    
    
    








