import numpy as np
import speech_recognition as sr
import time


def find_words(recognizer, audio):
    
    """Takes a recognizer and microphone instance, returns list of words"""

    if not isinstance(r, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")
    
    try:       
        return recognizer.recognize_google(audio).split()
        
    except sr.UnknownValueError:
        print("Say something!!")
    


if __name__ == "__main__":
    
    """Run main program to identify if 'like' is in text"""
    
    #Create instances of recognizer and mic
    r = sr.Recognizer()
    mic = sr.Microphone()
    
    #Check Microphone instance of sr.mic
    if not isinstance(mic, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")
    
    #Collect Audio Data
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, timeout=10)
        
    #Find words in data and convert
    words = find_words(r, audio)

    if words:
        words = str([word.lower() for word in words])
    
        #Print
        if 'like' in words:
            print("DON'T SAY LIKE")
        else:
            print("Good job not saying like")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    