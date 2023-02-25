"""
MFCC： 输入语音->预加重->分帧->加窗->FFT->幅值平方->mel滤波器->对数功率->离散余弦变换->MFCC
"""
from basic_operator import pre_emphasis,framing,add_window,stft,mel_filter,log_pow,discrete_cosine_transform,plot_time,plot_freq,plot_spectrogram
from matplotlib import pyplot as plt
from scipy.io import wavfile 
# 提取mfcc：输入语音，预加重，分帧，加窗，FFT，幅值平方，mel滤波器，对数功率，离散余弦变换
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
step1   =   pre_emphasis(data) # 预加重
step2   =   framing(step1,fs) # 分帧
step3   =   add_window(step2,fs) # 加窗
# step4   =   my_fft(step3) # 
step4=step3
step5   =   stft(step4) # FFT+幅值平方
step6   =   mel_filter(step5, fs) # mel滤波
step7   =   log_pow(step6) # 对数功率
mfcc    =   discrete_cosine_transform(step7)
plot_time(mfcc[1],fs,"mfcc_time.png")
plot_freq(mfcc[1],fs,"mfcc_freq.png")
plot_spectrogram(mfcc.T, ylabel='Filter Banks',png_name="mfcc.png")
# plt.figure(figsize=(10, 5))
# plt.plot(mfcc)
# plt.grid()  
# plt.title("mfcc") # 相比与fbank更加规整了
# plt.savefig('mfcc.png')
# plt.show()