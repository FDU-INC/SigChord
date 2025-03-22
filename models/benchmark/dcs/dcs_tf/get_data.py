import h5py
import numpy as np

DEFAULT_OFFSETS = [2, 11, 38, 7, 22, 24, 0, 16, 9, 17, 10, 21, 23, 33, 4, 34, 37, 18, 30, 32, 8, 15, 5, 3, 27, 12, 6, 20, 29, 28, 1, 13, 35, 36, 19, 31, 26, 14, 25, 39]
DEFAULT_OFFSETS_SORTED_IDS = np.argsort(DEFAULT_OFFSETS)

n_avail_bands = 16
n_total_bands = 40
N_signals = 130

file = h5py.File("./data/multi_usrp_test.h5", "r")

waveforms = file["waveforms"][:N_signals]
waveforms = waveforms[:,:, ::2] + waveforms[:,:, 1::2] * 1j
waveforms = waveforms[:,:, DEFAULT_OFFSETS_SORTED_IDS]

def dftA(signals, offsets, nBands):
    """
    Transforms signals sampled by multicoset method into frequency domain

    #### returns

      The measurement matrix A with shape(nChannels, nBands)
    """
    # DTFT * Ts = DTF, so the coeffecient 1/LT will be eleminated
    # To get matrix Y, we need to use the coeffecients to time the fft results.
    A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, nBands, 1))
    A = np.matrix(np.exp(2j * np.pi * A_aux / nBands))

    # to compensate the amplitudes caused by numpy.fft.fft(signals)
    #A = signals.shape[1] * A
    return A

def dftY(signals, offsets, nBands):
    """
    Transforms signals sampled by multicoset method into frequency domain

    #### returns

      Multicoset sampling results in frequency domain, say, Y
    """
    # DTFT * Ts = DTF, so the coeffecient 1/LT will be eleminated
    # To get matrix Y, we need to use the coeffecients to time the fft results.
    Y_aux = np.matrix(offsets).T * np.matrix(np.arange(0, signals.shape[1], 1))
    YCoeff = np.exp(-2j * np.pi / (nBands * signals.shape[1]) * Y_aux)

    # amplitudes got by numpy.fft.fft are N times larger than actual amplitudes
    # in frequency domain. We don't shrink here considering float point
    # arithmetic errors
    Y = np.matrix(np.multiply(YCoeff, np.fft.fft(signals)))

    return Y

def awgn(pure, snr, random_seed=None):
    if snr is None:
        return pure

    # sig_p_sqrt = np.linalg.norm(pure) / np.sqrt(sz)
    sig_p_sqrt = np.sqrt(np.mean(np.abs(pure) ** 2))
    noise_scale = sig_p_sqrt / (10 ** (snr / 20)) / np.sqrt(2)

    rng = np.random
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)

    n = rng.normal(size=pure.shape, scale=1.0) + 1j * rng.normal(
        size=pure.shape,
        scale=1.0,
    )
    n = n * noise_scale

    return pure + n

def getA(nChannels):
    sample = waveforms[0][:, DEFAULT_OFFSETS[:nChannels]].T
    A = dftA(sample, DEFAULT_OFFSETS[:nChannels], nBands=40)
    return A[:,:n_avail_bands]


def get_single_signal(item_index,use_noise):
    signal = waveforms[item_index]
    if use_noise > 0:
        signal = awgn(signal, use_noise)
    signal = signal.reshape([-1])
    signal = np.fft.fft(signal)/(n_total_bands*2400)
    signal = signal[:n_avail_bands*2400]
    signal = signal.reshape([n_avail_bands,2400])
    signal = signal.T #[2400,16]
    scale = np.max(np.abs(signal))
    signal=signal/scale
    return signal,scale