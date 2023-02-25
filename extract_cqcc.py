"""
cqcc：输入语音->预加重->分帧->加窗->CQT->幅值平方对数功率->均匀采样->离散余弦变换->CQCC
"""
# 提取 CQCC:输入语音，预加重，分帧，加窗，CQT，幅值平方对数功率，均匀采样，离散余弦变换
# print("使用spafe/features/test_bywsy.py进行测试")
from scipy.io import wavfile
from spafe.features.cqcc import cqcc
from matplotlib import pyplot as plt
path="D:/大四下/助教/实验一/我爱南开.wav"
fs, data = wavfile.read(path)
cqcc_feature=cqcc(data)
plt.figure(figsize=(10, 5))
plt.plot(cqcc_feature)
plt.grid()  
plt.title("cqcc") # 清晰可见的几条线
plt.savefig('cqcc.png')
plt.show()