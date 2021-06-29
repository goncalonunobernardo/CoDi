from TIV.TIVlib import TIV as tiv
from essentia.standard import MonoLoader, Windowing, Spectrum, SpectralPeaks, FrameGenerator, HPCP
import pickle

def load(filename):
    # Load data (deserialize)
    with open(filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        return unserialized_data

tiv_object = load('json_features/tivdata.pickle')
print(tiv_object[0]['TIV'].vector)