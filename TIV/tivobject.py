from TIVlib import TIV as tiv
from essentia.standard import MonoLoader, Windowing, Spectrum, SpectralPeaks, FrameGenerator, HPCP
import numpy as np
import json

import pickle

def file_to_hpcp(filename):
    audio = MonoLoader(filename=filename)()
    windowing = Windowing(type='blackmanharris62')
    spectrum = Spectrum()
    spectral_peaks = SpectralPeaks(orderBy='magnitude',
                                    magnitudeThreshold=0.001,
                                    maxPeaks=5,
                                    minFrequency=20,
                                    maxFrequency=8000)
    hpcp = HPCP(maxFrequency=8000)#,
                #normalized='unitSum')

    spec_group = []
    hpcp_group = []


    for frame in FrameGenerator(audio,frameSize=1024,hopSize=512):
        windowed = windowing(frame)
        fft = spectrum(windowed)
        frequencies, magnitudes = spectral_peaks(fft)
        final_hpcp = hpcp(frequencies, magnitudes)

        spec_group.append(fft)
        hpcp_group.append(final_hpcp)
    
    mean_hpcp = np.mean(np.array(hpcp_group).T, axis = 1)
    #Rotate the HPCP so that it starts in C
    mean_hpcp = np.roll(mean_hpcp,-3)
    return mean_hpcp  

metal = "TIVlib/audio_file.wav"
metal_hpcp = file_to_hpcp(metal)
metal_tiv = tiv.from_pcp(metal_hpcp)