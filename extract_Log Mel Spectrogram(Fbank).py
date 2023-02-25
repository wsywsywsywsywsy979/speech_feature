"""
抽取Fbank：输入语音->预加重->分帧->加窗->FFT->幅值平方->mel 滤波器->对数功率->Fbank
"""
from basic_operator import pre_emphasis,framing,add_window,my_fft,stft,mel_filter,log_pow
from matplotlib import pyplot as plt
from scipy.io import wavfile 
# 提取fbank：输入语音，预加重，分帧，加窗，FFT，幅值平方，mel滤波器，对数功率
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
step1   =   pre_emphasis(data) # 预加重
step2   =   framing(step1,fs) # 分帧
step3   =   add_window(step2,fs) # 加窗
step4   =   my_fft(step3) # FFT
step5   =   stft(step4) # 幅值平方
step6   =   mel_filter(step5, fs) # mel滤波
fbank   =   log_pow(step6) # 对数功率
plt.figure(figsize=(10, 5))
plt.plot(fbank)
plt.grid()  
plt.title("fbank") # 画图发现确实fbank比语谱图干净多了
plt.savefig('fbank.png')
plt.show()