"""
target：用于测试该仓库下的特征进行抽取
"""
from scipy.io import wavfile
from rplp import plp
from cqcc import cqcc
from matplotlib import pyplot as plt
if __name__=="__main__":
    path=r"要进行抽取的音频路径" 
    while True:
        c=int(input("输入1抽取plp；输入2抽取cqcc；输入其他退出程序："))
        fs, data = wavfile.read(path)
        if c==1:
            plp_feature=plp(data)
            plt.figure(figsize=(20, 5))
            plt.plot(plp_feature)
            plt.grid()  
            plt.title("plp") # 感觉plp画出来很奇怪:有两条直线，其余都均匀在-40波动
            # a=1
            plt.show()
        elif c==2:
            cqcc_feature=cqcc(data)
            plt.figure(figsize=(20, 5))
            plt.plot(cqcc_feature)
            plt.grid()  
            plt.title("cqcc") # 清晰可见的几条线
            plt.show()
        else:
            break
