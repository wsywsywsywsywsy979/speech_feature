"""
抽取Fbank：输入语音->预加重->分帧->加窗->FFT->幅值平方->mel 滤波器->对数功率->Fbank
"""
from basic_operator import pre_emphasis,framing,add_window,my_fft,stft,mel_filter,log_pow,plot_time,plot_freq,plot_spectrogram
from matplotlib import pyplot as plt
from scipy.io import wavfile 
import numpy as np
# 提取fbank：输入语音，预加重，分帧，加窗，FFT，幅值平方，mel滤波器，对数功率
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
# plot_time(data, fs) 就是和cool edit中的一样的波形
# plot_freq(data, fs)  # 果然是低频能量更高一些
step1   =   pre_emphasis(data) # 预加重
# plot_time(step1,fs)
# plot_freq(step1,fs) # 预加重之后低频和高频的能量基本一致了，确实起到了平衡频谱的作用。
step2   =   framing(step1,fs) # 分帧
step3   =   add_window(step2,fs) # 加窗
# plot_time(step3[1],fs) # 注意是只取一帧来画
# plot_freq(step3[1],fs)
# step4   =   my_fft(step3) # FFT
step4=step3
step5   =   stft(step4) # 幅值平方
step6   =   mel_filter(step5, fs) # mel滤波
fbank   =   log_pow(step6) # 对数功率
plot_time(fbank[1],fs,"fbank_time.png")
plot_freq(fbank[1],fs,"fbank_freq.png")
plot_spectrogram(fbank.T, ylabel='Filter Banks',png_name="fbank.png")
# plt.figure(figsize=(10, 5))
# plt.plot(fbank)
# plt.grid()  
# plt.title("fbank") # 画图发现确实fbank比语谱图干净多了
# plt.savefig('fbank.png') # 注意，此处画出来的是频域图
# plt.show()