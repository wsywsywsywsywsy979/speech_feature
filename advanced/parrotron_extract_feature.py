import librosa
import numpy as np
import torch
def parse_audio(self, audio_path):
        signal, _ = librosa.load(audio_path, self.sample_rate)
        spec = librosa.feature.melspectrogram(y=signal, 
                                                sr=self.sample_rate,
                                                n_fft=1024, 
                                                hop_length=200, 
                                                win_length=800, 
                                                window='hann',
                                                n_mels=80, 
                                                fmin=125, 
                                                fmax=7600)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = torch.from_numpy(spec)
        spec = spec.unsqueeze(0)
        return spec

def parse_audio_tts(self, audio_path):
        '''
        #check sample rate
        with wave.open(audio_path, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
            print(frame_rate)
        '''
        signal, _ = librosa.load(audio_path, self.sample_rate)
        spec = librosa.stft(signal, n_fft=2048, hop_length=200, win_length=800, window='hann')
        spec = np.abs(spec)
        spec = torch.from_numpy(spec)
        spec = spec.unsqueeze(0)
        return spec