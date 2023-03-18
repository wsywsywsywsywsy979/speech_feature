"""
抽取spectrogram：预加重->分帧->加窗->FFT->幅值平方->对数功率->spectrugram
"""
from basic_operator import pre_emphasis, framing,add_window,my_fft,stft,log_pow,plot_time,plot_freq,plot_spectrogram

# import wavfile
from scipy.io import wavfile # 注意使用的包的来源不一样也会有很大的影响

from matplotlib import pyplot as plt

# 语谱图(spectrogram)：输入语音，预加重，分帧，加窗，FFT，幅值平方，对数功率 
path="wav/我爱南开.wav"
fs, data = wavfile.read(path)
step1   =   pre_emphasis(data) # 预加重
step2   =   framing(step1,fs) # 分帧
step3   =   add_window(step2,fs) # 加窗

# step4   =   my_fft(step3) 

step4=step3
step5   =   stft(step4) # FFT+幅值平方
spectrogram =   log_pow(step5) # 对数功率

# plot_time(spectrogram[1],fs,"spectrogram_time.png")
# plot_freq(spectrogram[1],fs,"spectrogram_freq.png")

plot_spectrogram(spectrogram.T, ylabel='Spectrogram',png_name="spectrogram.png")
