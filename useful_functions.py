import numpy as np
import contextlib
import wave
import pyaudio
import webrtcvad as vd
from random import randint

def acc(cm):
    """Compute accuracy from Confusion Matrix"""
    
    sm = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                sm += cm[i,j]
                
    return sm / np.sum(cm)


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate
    
def record_with_vad(duration, fs):
    """Records a snippet of data and returns numpy array and vad
    classification """

    #Set parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    samp_width = 2
    frame_duration = 10 #ms
    RATE = fs #Sampling rate
    CHUNK = int(duration*RATE) #Datapoints to read at a time

    vad = vd.Vad(3)
    
    #Get audio data
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK) #Incoming audio stream

    print('Speak')
    data = stream.read(CHUNK) #Acquire data from stream
    print('Stop')

    numpy_data = np.frombuffer(data, np.int16) #Convert data to numpy array    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    #Run VAD
    datapoints_per_vad = int(samp_width * frame_duration / 1000 * fs)
    t_vad = np.linspace(int(datapoints_per_vad/2), len(numpy_data), int(len(numpy_data)/(datapoints_per_vad/2)))
    vad_output = np.zeros((len(t_vad)))
    
    vad_index = 0
    for i in range(len(data)):
        if i > 0 and i % datapoints_per_vad == 0:
            if vad.is_speech(data[int(i-datapoints_per_vad):int(i)], fs):
                vad_output[vad_index] = 1
            vad_index += 1
        
    return numpy_data, vad_output, t_vad


def augment(audio, length):
    """Return audio in variable location of length"""
    
    aud_len = len(audio)
    
    if aud_len > length:
        aud_len = length
        diff = 0
    else:
        diff = length - aud_len
    
    rand_index = randint(0, diff) #Random offset for audio data
    
    if rand_index != 0:
        rand_index -= 1
    
    X = np.zeros((length))
    X[rand_index:int(rand_index+aud_len)] = audio[:aud_len]

    return X    





