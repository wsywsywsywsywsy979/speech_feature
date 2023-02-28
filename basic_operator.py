from scipy.fftpack import dct
import numpy as np
from matplotlib import pyplot as plt

def pre_emphasis(sig):
    """
    function:预加重
    para: sig:要进行处理的音频数据
    return 进行加强处理后的音频数据
    """
    preemphasis=0.97
    # sig=np.append(sig[0],sig[1:]-np.array([preemphasis])*sig[:-1])
    sig=np.append(sig[0],sig[1:]-preemphasis*sig[:-1])
    return sig

def framing(sig,fs,frame_len_s=0.025,frame_shift_s=0.01):
    """
    function：分帧
    para：
        frame_len_s:每一帧的长度,单位为s
        frame_shift_s:分帧的shift,单位为s
        fs：采样率，hz
        sig：要进行分帧的音频信号
    return：进行分帧后的数据，一个二维list，一个元素是一帧信号
    """
    sig_n=len(sig)
    frame_len_n=int(round(fs*frame_len_s))
    frame_shift_n=int(round(fs*frame_shift_s))
    num_frame=int(np.ceil(float(sig_n-frame_len_n)/frame_shift_n)+1) 
    pad_num=frame_shift_n*(num_frame-1)+frame_len_n-sig_n # 待补0的个数
    # 一种前后向拼接array的方法-------------
    pad_zero=np.zeros(int(pad_num)) 
    pad_sig=np.append(sig,pad_zero) 
    #-------------------------------------
    # 计算下标：
    frame_inner_index=np.arange(0,frame_len_n)
    # 分帧后每个帧的起始下标：
    frame_index=np.arange(0,num_frame)*frame_shift_n    
    # 在行方向上复制每个帧的内部下标：
    frame_inner_index_extend=np.tile(frame_inner_index,(num_frame,1))
    # 各帧起始下标扩展维度，便于后续相加：
    frame_index_extend=np.expand_dims(frame_index,1)
    # 分帧后各帧下标：
    each_frame_index=frame_inner_index_extend+frame_index_extend
    each_frame_index=each_frame_index.astype(np.int,copy=False)
    frame_sig=pad_sig[each_frame_index]
    return frame_sig

def add_window(frame_sig,fs,frame_len_s=0.025):
    """
    function：加窗
    para：
        frame_len_s：每一帧的长度,单位为s
        fs：采样率
        frame_sig:进行分帧后的数据
    return：加窗后的数据
    """
    window=np.hamming(int(round(frame_len_s*fs)))
    frame_sig*=window
    return frame_sig

def my_fft(frame_sig):
    """
    function：傅里叶变换
    para:frame_sig：进行加窗处理后的数据
    return：进行傅里叶变换后的数据
    """
    NFFT=512 # NFFT常为256或512
    mag_frames = np.absolute(np.fft.rfft(frame_sig, NFFT)) 
    return mag_frames   

def stft(frame_sig, nfft=512):    
    """
    function：短时傅里叶变换将帧信号变为帧功率（对应幅值频发）
    para：
        frame_sig: 分帧后的信号
        nfft: fft点数
    return: 返回分帧信号的功率谱
    """
    frame_spec = np.fft.rfft(frame_sig, nfft)
    """
    np.fft.fft vs np.fft.rfft
    fft 返回 nfft
    rfft 返回 nfft // 2 + 1，即rfft仅返回有效部分
    """
    # 幅度谱
    frame_mag = np.abs(frame_spec) # 语音信号频谱取模
    # 功率谱
    frame_pow = (frame_mag ** 2) * 1.0 / nfft # 取平方 
    return frame_pow

def mel_filter(frame_pow, fs, n_filter=80, nfft=512):    
    """
    function:mel 滤波器系数计算
    para:
        frame_pow: 分帧信号功率谱
        fs: 采样率 hz
        n_filter: 滤波器个数
        nfft: fft点数
    return: 分帧信号功率谱mel滤波后的值的对数值
    mel = 2595 * log10(1 + f/700) # 频率到mel值映射
    f = 700 * (10^(m/2595) - 1 # mel值到频率映射
    上述过程本质上是对频率f对数化
    """
    mel_min = 0 # 最低mel值
    mel_max = 2595 * np.log10(1 + fs / 2.0 / 700) # 最高mel值，最大信号频率为 fs/2
    mel_points = np.linspace(mel_min, mel_max, n_filter + 2) # n_filter个mel值均匀分布与最低与最高mel值之间
    hz_points = 700 * (10 ** (mel_points / 2595.0) - 1) # mel值对应回频率点，频率间隔指数化
    filter_edge = np.floor(hz_points * (nfft + 1) / fs) # 对应到fft的点数比例上
    # 求mel滤波器系数
    fbank = np.zeros((n_filter, int(nfft / 2 + 1)))
    for m in range(1, 1 + n_filter):
        f_left = int(filter_edge[m - 1]) # 左边界点
        f_center = int(filter_edge[m]) # 中心点
        f_right = int(filter_edge[m + 1]) # 右边界点
        for k in range(f_left, f_center):
            fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    # mel 滤波
    # [num_frame, nfft/2 + 1] * [nfft/2 + 1, n_filter] = [num_frame, n_filter]
    filter_banks = np.dot(frame_pow, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # np.finfo函数是根据括号中的类型来获得信息，获得符合这个类型的数型, eps是取非负的最小值
    return filter_banks

def log_pow(filter_banks):
    """
    function：功率取对数
    para：filter_banks:经过mel滤波器的数据
    return：取对数后的功率数据，即fbank
    """
    # 取对数 (功率取对数)
    filter_banks = 20 * np.log10(filter_banks) # dB
    return filter_banks

def discrete_cosine_transform(filter_banks):
    """
    function：离散余弦变换
    para：filter_banks:fbanks
    return: mfcc
    """
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)] # 保持在2-13
    """
    scipy.fftpack.dct:返回任意类型序列x的离散余弦变换。
    """
    return mfcc

def hz2bark(f):
    """ 
    function:Hz to bark频率 (Wang, Sekey & Gersho, 1992.) 
    para: f：要进行转换的频率
    return: 转换后的bark频率
    """
    return 6. * np.arcsinh(f / 600.)

def bark2hz(fb):
    """ 
    function:Bark频率 to Hz
    para:fb
    return: 转换后的赫兹频率
    """
    return 600. * np.sinh(fb / 6.)

def bark2fft(fb, fs=16000, nfft=512):
    """ 
    function:Bark频率 to FFT频点 
    para:   
        fb
        fs:采样率
        nfft
    return:
    """
    # bin = sample_rate/2 / nfft/2=sample_rate/nfft    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*nfft/ sample_rate    # hz_points对应第几个fft频点
    return (nfft + 1) * bark2hz(fb) / fs

def fft2bark(fft, fs=16000, nfft=512):
    """ 
    function：FFT频点 to Bark频率 
    para:
        fft
        fs：采样率
        nfft
    """
    return hz2bark((fft * fs) / (nfft + 1))

def Fm(fb, fc):
    """ 
    计算一个特定的中心频率的Bark filter
    para：
        fb: frequency in Bark.
        fc: center frequency in Bark.
    return: 相关的Bark filter 值/幅度
    """
    if fc - 2.5 <= fb <= fc - 0.5:
        return 10 ** (2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10 ** (-2.5 * (fb - fc - 0.5))
    else:
        return 0

def bark_filter_banks(nfilts=20, nfft=512, fs=16000, low_freq=0, high_freq=None, scale="constant"):
    """ 
    function：计算Bark-filterbanks,(B,F)
    para：
        nfilts: 滤波器组中滤波器的数量 (Default 20)
        nfft: FFT size.(Default is 512)
        fs: 采样率，(Default 16000 Hz)
        low_freq: MEL滤波器的最低带边。(Default 0 Hz)
        high_freq: MEL滤波器的最高带边。(Default samplerate/2)
        scale (str): 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
    return:一个大小为(nfilts, nfft/2 + 1)的numpy数组，包含滤波器组。
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # 按Bark scale 均匀间隔计算点数(点数以Bark为单位)
    low_bark = hz2bark(low_freq)
    high_bark = hz2bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4) # 作为序列生成器， numpy.linspace () 函数用于在线性空间中以均匀步长生成数字序列。

    bins = np.floor(bark2fft(bark_points))  # Bark Scale等分布对应的 FFT bin number
    # [  0.   2.   5.   7.  10.  13.  16.  20.  24.  28.  33.  38.  44.  51.
    #   59.  67.  77.  88. 101. 115. 132. 151. 172. 197. 224. 256.]
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    for i in range(0, nfilts):      # --> B
        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)
        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for j in range(int(bins[i]), int(bins[i + 4])):     # --> F
            fc = bark_points[i+2]   # 中心频率
            fb = fft2bark(j)        # Bark 频率
            fbank[i, j] = c * Fm(fb, fc)
    return np.abs(fbank)

def plot_time(sig, fs,png_name):
    """
    绘制时域图
    """
    time = np.arange(0, len(sig)) * (1.0 / fs)
    plt.figure(figsize=(10, 5))
    plt.plot(time, sig)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(png_name)
    plt.show()

def plot_freq(sig, sample_rate, png_name,nfft=512):
    """
    绘制频域图
    """
    xf = np.fft.rfft(sig, nfft) / nfft
    freqs = np.linspace(0, int(sample_rate/2), int(nfft/2 + 1))
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.savefig(png_name)
    plt.show()

def plot_spectrogram(spec, ylabel,png_name):
    """
    绘制二维数组
    """
    fig = plt.figure(figsize=(10, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Frame(s)')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(png_name)
    plt.show()
# 读取音频文件，可参考librosa，soundfile，wavefile包的使用