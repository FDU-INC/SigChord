import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from prelude import CNDarray

def main():
    waveform_info = scipy.io.loadmat("./5g/5g_mcs0_id0.mat")
    waveform: CNDarray = waveform_info["waveform"].flatten()
    sampling_rate = 15.36e6
    waveform = waveform.reshape((-1, 256))
    waveform_f = np.fft.fft(waveform, axis=-1)
    waveform_f = np.fft.fftshift(waveform_f, axes=-1)
    # for i in range(len(waveform_f)):
    #     sig_f = np.fft.fft(waveform[i][-1024:])
    #     qpsk_ids = 2 * (sig_f.real > 0) + (sig_f.imag > 0)
    #     print(qpsk_ids)
    #     plt.plot(sig_f.real)
    #     plt.plot(sig_f.imag)
    #     plt.show()
    plt.imshow(np.abs(waveform_f.T))
    plt.show()

if __name__ == "__main__":
    main()
