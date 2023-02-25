"""
抽取spectrogram：预加重->分帧->加窗->FFT->幅值平方->对数功率->spectrugram
"""
from basic_operator import pre_emphasis, framing,add_window,my_fft,stft,log_pow
# import wavfile
from scipy.io import wavfile # 注意使用的包的来源不一样也会有很大的影响
from matplotlib import pyplot as plt
# 语谱图(spectrogram)：输入语音，预加重，分帧，加窗，FFT，幅值平方，对数功率 
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
step1   =   pre_emphasis(data) # 预加重
step2   =   framing(step1,fs) # 分帧
step3   =   add_window(step2,fs) # 加窗
step4   =   my_fft(step3) # FFT
step5   =   stft(step4) # 幅值平方
spectrogram =   log_pow(step5) # 对数功率
plt.figure(figsize=(10, 5))
plt.plot(spectrogram)
plt.grid()  
plt.title("spectrogram")
plt.savefig('spectrogram.png')
plt.show()
