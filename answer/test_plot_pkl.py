import pickle 
import torch
# f = open('D:/大四下/助教/实验一/left0_melspectrogram.pkl','rb')
f = open('D:/大四下/助教/实验一/left0_sequence.pkl','rb')
melspec =pickle.load(f)
f.close()
from basic_operator import plot_spectrogram
# plot_spectrogram(melspec[0].numpy(),ylabel='melspec',png_name="left0_melspectrogram.png")
plot_spectrogram(melspec[0].numpy(),ylabel='melspec',png_name="left0_sequence.png")