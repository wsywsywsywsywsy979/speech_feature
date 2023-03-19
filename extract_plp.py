from scipy.io.wavfile import read
from spafe.features.rplp import plp
from spafe.utils.vis import show_features

# read audio
fpath = "wav/我爱南开.wav"
fs, sig = read(fpath)

# compute plps
plps = plp(sig,
           fs=fs,
           pre_emph=0,
           pre_emph_coeff=0.97,
           nfilts=128,
           nfft=1024,
           low_freq=0,
           high_freq=fs/2,
           lifter=0.9,
           normalize="mvn")

# visualize features
show_features(plps, "Perceptual linear predictions", "PLP Index", "Frame Index")