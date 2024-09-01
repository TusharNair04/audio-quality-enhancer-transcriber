import librosa
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def enhance_audio(input_path, output_path, lowcut=300.0, highcut=3000.0):
    y, sr = librosa.load(input_path, sr=None)
    y_filtered = librosa.effects.preemphasis(y)
    y_filtered = bandpass_filter(y_filtered, lowcut, highcut, sr)
    y_normalized = librosa.util.normalize(y_filtered)
    write(output_path, sr, np.int16(y_normalized * 32767))
