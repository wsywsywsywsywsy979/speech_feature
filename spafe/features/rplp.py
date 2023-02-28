"""

- Description : (Rasta) Perceptual linear prediction coefficents (RPLPs/PLPs) extraction algorithm implementation.
- Copyright (c) 2019-2022 Ayoub Malek.
  This source code is licensed under the terms of the BSD 3-Clause License.
  For a copy, see <https://github.com/SuperKogito/spafe/blob/master/LICENSE>.

"""
import numpy as np
#--------wsy add and fix------ 
from ..utils.filters import rasta_filter
from ..features.lpc import __lpc_helper, lpc2lpcc
from ..fbanks.bark_fbanks import bark_filter_banks
from ..utils.cepstral import normalize_ceps, lifter_ceps
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
#-------------------


def __rastaplp(
    sig,
    fs=16000,
    order=13,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    do_rasta=False,
    nfilts=24,
    nfft=512,
    low_freq=0,
    high_freq=None,
    scale="constant",
    lifter=None,
    normalize=None,
    fbanks=None,
    conversion_approach="Wang",
):
    """
    Compute Perceptual Linear Prediction coefficients with or without rasta filtering.

    Args:
        sig               (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01). # 一般=win_len/2
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        do_rasta           (bool) : apply Rasta filtering if True.
                                    (Default is False).
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specifid.
                                    (Default is 0).
        normalize           (int) : apply normalization if approach specifid.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of the PLP or RPLP coefficients.
        (Matrix of features, row = feature, col = frame).

    Tip:
        - :code:`scale` : can take the following options ["constant", "ascendant", "descendant"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    """
    high_freq = high_freq or fs / 2
    num_ceps = order

    # run checks
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    #  compute fbanks 
    if fbanks is None:
        bark_fbanks_mat, _ = bark_filter_banks(
            nfilts=nfilts, # 24
            nfft=nfft, # 512
            fs=fs, # 16000
            low_freq=low_freq, # 0
            high_freq=high_freq, # 8000.0
            scale=scale,
            conversion_approach=conversion_approach,
        )
        fbanks = bark_fbanks_mat

    # pre-emphasis
    if pre_emph: # 所以对于plp 预加重是可选的
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing # 分帧
    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # -> windowing # 加窗
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    # -> FFT -> |.| # fft
    ## Magnitude of the FFT 
    fourrier_transform = np.absolute(np.fft.fft(windows, nfft))
    fourrier_transform = fourrier_transform[:, : int(nfft / 2) + 1]

    ##  -> |.|^2 (Power Spectrum) # 幅值平方
    abs_fft_values = (1.0 / nfft) * np.square(fourrier_transform)

    # -> x filter bank = auditory spectrum # bark进行滤波
    auditory_spectrum = np.dot(a=abs_fft_values, b=fbanks.T)

    # rasta filtering
    if do_rasta:
        # put in log domain
        nl_aspectrum = np.log(auditory_spectrum)

        # next do rasta filtering
        ras_nl_aspectrum = rasta_filter(nl_aspectrum)

        # do inverse log
        auditory_spectrum = np.exp(ras_nl_aspectrum)

    # equal loudness pre_emphasis # 等响度预加重
    E = lambda w: ((w**2 + 56.8 * 10**6) * w**4) / (
        (w**2 + 6.3 * 10**6)
        * (w**2 + 0.38 * 10**9)
        * (w**6 + 9.58 * 10**26)
    )
    Y = [E(w) for w in auditory_spectrum]

    # intensity loudness compression 强度响度转换
    L = np.abs(Y) ** (1 / 3)

    # ifft 逆傅里叶变换
    inverse_fourrier_transform = np.absolute(np.fft.ifft(L, nfft))

    # compute lpcs and lpccs 线性预测（lpc)
    lpcs = np.zeros((L.shape[0], order))
    lpccs = np.zeros((L.shape[0], order))
    for i in range(L.shape[0]):

        a, e = __lpc_helper(inverse_fourrier_transform[i, :], order - 1)
        lpcs[i, :] = a
        lpcc_coeffs = lpc2lpcc(a, e, order)
        lpccs[i, :] = np.array(lpcc_coeffs)

    # liftering
    if lifter:
        lpccs = lifter_ceps(lpccs, lifter)

    # normalize
    if normalize:
        lpccs = normalize_ceps(lpccs, normalize)

    return lpccs


def plp(
    sig,
    fs=16000,
    order=13,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfilts=24,
    nfft=512,
    low_freq=0,
    high_freq=None,
    scale="constant",
    lifter=None,
    normalize=None,
    fbanks=None,
    conversion_approach="Wang",
):
    """
    Compute Perceptual linear prediction coefficents according to [Hermansky]_
    and [Ajibola]_.

    Args:
        sig               (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01).
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specifid.
                                    (Default is 0).
        normalize           (int) : apply normalization if approach specifid.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of PLP features (num_frames x order)

    Tip:
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.

    Note:
        .. figure:: ../_static/architectures/plps.png

           Architecture of perceptual linear prediction coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.rplp import plp
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute plps
            plps = plp(sig,
                       fs=fs,
                       pre_emph=0,
                       pre_emph_coeff=0.97,
                       win_len=0.030,
                       win_hop=0.015,
                       win_type="hamming",
                       nfilts=128,
                       nfft=1024,
                       low_freq=0,
                       high_freq=fs/2,
                       lifter=0.9,
                       normalize="mvn")

            # visualize features
            show_features(plps, "Perceptual linear predictions", "PLP Index", "Frame Index")
    """
    return __rastaplp(
        sig,
        fs=fs, # 16000
        order=order, # 13
        pre_emph=pre_emph, #0
        pre_emph_coeff=pre_emph_coeff, # 0.97 这是进行预加重的参数
        win_len=win_len, # 0.025 应该是窗口长度
        win_hop=win_hop, # 0.01 ?
        win_type=win_type,# 'hamming'
        do_rasta=False,
        nfilts=nfilts, # 24 ?
        nfft=nfft, # 512
        low_freq=low_freq, # 0
        high_freq=high_freq, # None
        scale="constant",
        lifter=lifter, # None
        normalize=normalize, # None
        fbanks=fbanks, # NOne
        conversion_approach="Wang",
    )


def rplp(
    sig,
    fs=16000,
    order=13,
    pre_emph=0,
    pre_emph_coeff=0.97,
    win_len=0.025,
    win_hop=0.01,
    win_type="hamming",
    nfilts=24,
    nfft=512,
    low_freq=0,
    high_freq=None,
    scale="constant",
    lifter=None,
    normalize=None,
    fbanks=None,
    conversion_approach="Wang",
):
    """
    Compute rasta Perceptual linear prediction coefficents according to [Hermansky]_
    and [Ajibola]_.

    Args:
        sig               (numpy.ndarray) : a mono audio signal (Nx1) from which to compute features.
        fs                  (int) : the sampling frequency of the signal we are working with.
                                    (Default is 16000).
        order               (int) : number of cepstra to return.
                                    (Default is 13).
        pre_emph            (int) : apply pre-emphasis if 1.
                                    (Default is 1).
        pre_emph_coeff    (float) : pre-emphasis filter coefﬁcient.
                                    (Default is 0.97).
        win_len           (float) : window length in sec.
                                    (Default is 0.025).
        win_hop           (float) : step between successive windows in sec.
                                    (Default is 0.01).
        win_type          (float) : window type to apply for the windowing.
                                    (Default is "hamming").
        nfilts              (int) : the number of filters in the filter bank.
                                    (Default is 40).
        nfft                (int) : number of FFT points.
                                    (Default is 512).
        low_freq            (int) : lowest band edge of mel filters (Hz).
                                    (Default is 0).
        high_freq           (int) : highest band edge of mel filters (Hz).
                                    (Default is samplerate/2).
        scale              (str)  : monotonicity behavior of the filter banks.
                                    (Default is "constant").
        lifter              (int) : apply liftering if specifid.
                                    (Default is 0).
        normalize           (int) : apply normalization if approach specifid.
                                    (Default is None).
        fbanks    (numpy.ndarray) : filter bank matrix.
                                    (Default is None).
        conversion_approach (str) : bark scale conversion approach.
                                    (Default is "Wang").

    Returns:
        (numpy.ndarray) : 2d array of rasta PLP features (num_frames x order)


    Tip:
        - :code:`normalize` : can take the following options ["mvn", "ms", "vn", "mn"].
        - :code:`conversion_approach` : can take the following options ["Tjomov","Schroeder", "Terhardt", "Zwicker", "Traunmueller", "Wang"].
          Note that the use of different options than the ddefault can lead to unexpected behavior/issues.
    Note:
        .. figure:: ../_static/architectures/rplps.png

           Architecture of rasta perceptual linear prediction coefﬁcients extraction algorithm.

    Examples:
        .. plot::

            from scipy.io.wavfile import read
            from spafe.features.rplp import rplp
            from spafe.utils.vis import show_features

            # read audio
            fpath = "../../../test.wav"
            fs, sig = read(fpath)

            # compute rplps
            rplps = rplp(sig,
                         fs=fs,
                         pre_emph=0,
                         pre_emph_coeff=0.97,
                         win_len=0.030,
                         win_hop=0.015,
                         win_type="hamming",
                         nfilts=128,
                         nfft=1024,
                         low_freq=0,
                         high_freq=fs/2,
                         lifter=0.9,
                         normalize="mvn")

            # visualize features
            show_features(rplps, "Rasta perceptual linear predictions", "PLP Index", "Frame Index")

    References:

        .. [Ajibola] : Ajibola Alim, S., & Khair Alang Rashid, N. (2018). Some
                       Commonly Used Speech Feature Extraction Algorithms.
                       From Natural to Artificial Intelligence - Algorithms and
                       Applications. doi:10.5772/intechopen.80419
    """
    return __rastaplp(
        sig,
        fs=fs,
        order=order,
        pre_emph=pre_emph,
        pre_emph_coeff=pre_emph_coeff,
        win_len=win_len,
        win_hop=win_hop,
        win_type=win_type,
        do_rasta=True,
        nfilts=nfilts,
        nfft=nfft,
        low_freq=low_freq,
        high_freq=high_freq,
        scale="constant",
        lifter=lifter,
        normalize=normalize,
        fbanks=fbanks,
        conversion_approach="Wang",
    )
