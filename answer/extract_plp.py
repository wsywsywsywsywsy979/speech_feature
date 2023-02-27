"""
PLP：输入语音->预加重->分帧->加窗->FFT->幅值平方->bark滤波器->等响度预加重->强度-响度转换->逆傅里叶变换->线性预测->PLP
"""
# 提取PLP：输入语音，预加重，分帧，加窗，FFT，幅值平方，bark滤波器，等响度预加重，强度-响度转换，逆傅里叶变换，线性预测
# print("使用spafe/features/test_bywsy.py进行测试")
from scipy.io import wavfile
from spafe.features.rplp import plp
from matplotlib import pyplot as plt
from basic_operator import plot_time,plot_freq,plot_spectrogram
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
plp_feature=plp(data)
plot_time(plp_feature[1],fs,"plp_feature_time.png")
plot_freq(plp_feature[1],fs,"plp_feature_freq.png")
plot_spectrogram(plp_feature.T, ylabel='Filter Banks',png_name="plp_feature.png")
# plt.figure(figsize=(10, 5))
# plt.plot(plp_feature)
# plt.grid()  
# plt.title("plp") # 感觉plp画出来很奇怪:有两条直线，其余都均匀在-40波动
# plt.savefig('plp.png')
# plt.show()
