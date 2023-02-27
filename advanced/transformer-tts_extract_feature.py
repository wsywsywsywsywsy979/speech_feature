import librosa
import torch
from layers import TacotronSTFT
import numpy as np
stft = TacotronSTFT()
def get_mel(filename):
    wav, sr = librosa.load(filename, sr=22050)
    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0), wav