"""
PLP：输入语音->预加重->分帧->加窗->FFT->幅值平方->bark滤波器->等响度预加重->强度-响度转换->逆傅里叶变换->线性预测->PLP
"""
# 提取PLP：输入语音，预加重，分帧，加窗，FFT，幅值平方，bark滤波器，等响度预加重，强度-响度转换，逆傅里叶变换，线性预测
# print("使用spafe/features/test_bywsy.py进行测试")
from scipy.io import wavfile
from spafe.features.rplp import plp as PLP
from matplotlib import pyplot as plt
from basic_operator import plot_time,plot_freq,plot_spectrogram,pre_emphasis,framing,add_window,stft,bark_filter_banks,log_pow,equal_loudness
import numpy as np
import librosa
path="wav/我爱南开.wav"
fs, data = wavfile.read(path)

#(1)-----------------------------------------------------------------------
plp_feature=PLP(data)
# plot_time(plp_feature[1],fs,"plp_feature_time.png")
# plot_freq(plp_feature[1],fs,"plp_feature_freq.png")
plot_spectrogram(plp_feature.T, ylabel='Filter Banks',png_name="plp1.png")
#--------------------------------------------------------------------------

#(2)--------------------------------------------------------------------------------------------------
step1 = pre_emphasis(data) # 预加重
step2 = framing(step1,fs) # 分帧
step3 = add_window(step2,fs) # 加窗
step4 = step3
step5 = stft(step4)
filterbanks = bark_filter_banks()
step6 = np.dot(filterbanks, step5.T)
step7=equal_loudness(step6) # 等响度预加重
step8=step7**0.33 # 强度响度转换？
step9=np.fft.ifft(step8,30) # 逆傅里叶变换
plp=librosa.lpc(abs(step9), 15)# 线性预测：要求输入参数为浮点型，经过ifft得到的plp_data有复数，因此要取ab
plot_spectrogram(plp.T, 'Perceptual Linear Predictive', "plp2.png")
#----------------------------------------------------------------------------------------------------------

#(3)-------------------------------------------------------------------------------
h1=1.0/np.fft.fft(plp,1024)
spec_envelope_plp=10*np.log10(abs(h1[0:512]))
lpc=librosa.lpc(step3,15) # y是加窗后的信号
h2=1.0/np.fft.fft(lpc,1024)
spec_envelope_lpc=10*np.log10(abs(h2[0:512]))
plot_spectrogram(spec_envelope_lpc.T, 'Perceptual Linear Predictive', "plp3.png")
#------------------------------------------------------------------------------------
