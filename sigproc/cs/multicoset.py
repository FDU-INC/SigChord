"""
A brief implementation for multicoset sampling scheme
"""

import numpy as np
import torch

DEFAULT_N_BANDS = 40
# DEFAULT_OFFSETS = [16, 11, 1, 0, 27, 24, 37, 31]
DEFAULT_OFFSETS = [2, 11, 38, 7, 22, 24, 0, 16, 9, 17, 10, 21, 23, 33, 4, 34, 37, 18, 30, 32, 8, 15, 5, 3, 27, 12, 6, 20, 29, 28, 1, 13, 35, 36, 19, 31, 26, 14, 25, 39]
DEFAULT_OFFSETS_SORTED_IDS = np.argsort(DEFAULT_OFFSETS)
DEFAULT_BEST_OFFSETS = {
    3: [1, 4, 8],
    4: [2, 3, 6, 15],
    5: [2, 4, 5, 10, 20],
}


def sample(
    signal,
    time: float,
    bandwidth: float,
    n_bands: int,
    n_chs: int,
    offsets: list[int] | None = None,
    scheme="discrete",
):
    """
    #### parameters:

    - `signal`: input signals. When `scheme` is `continuous`, `signal` is a
      function of time, Reals to Complex. When `scheme` is 'discrete', `signal`
      is a numpy.ndarray containing discrete signals, whose sampling frequency
      is indicated by `bandwidth`
    - `time`: sampling time duration, only used when `scheme` is 'continuous'
    - `bandwidth`: bandwidth W of the input signal, only used when `scheme` is
      'contiuous', where `bandwidth` is viewed as the sampling frequency of the input
      signal.
    - `n_bands`: Number of bands L. The cognitive radio system divides the
      shared spectrum into L orthogonal channels, so the bandwidth of each
      channel is B = W/L. The sampling interval for each band is `nBands` times
      input discrete signal sampling interval
    - `n_chs`: number of parallel sampling channels
    - `offsets`: offsets of each sampling channel, integers
    - `scheme`: 'continuous' by default. When set as 'continuous', use
      `bandwidth` to indicate the sampling frequency of the input signal

    #### returns:

    1. A numpy.ndarray. Each row is the sub-nyquist sampled signal sequence
    with time interval L/W and time offset c_i/W, where {c_i, i = 1, 2, ...}
    are non-negative integers
    2. offsets
    """
    if n_chs > n_bands:
        raise ValueError("nBands should be greater than or equal to nChannels")

    if offsets is not None:
        if len(offsets) < n_chs:
            raise ValueError(
                "length of offsets should not be less than sampling channels"
            )
        elif np.max(offsets) >= n_bands or np.min(offsets) < 0:
            raise ValueError("offset value should be in [0, L-1]")

        offsets = list(offsets[:n_chs])
    else:
        # construct offsets randomly
        rng = np.random.RandomState(43)
        pool = np.arange(0, n_bands, 1)
        # np.random.shuffle(pool)
        rng.shuffle(pool)
        offsets = list(pool[0:n_chs])

    sampling_res = np.zeros((1))
    if scheme == "continuous":
        T = 1 / bandwidth
        Ts = T * n_bands
        sampling_res = np.zeros((n_chs, int(np.floor(time / Ts))), dtype=np.cdouble)
        for i in range(n_chs):
            for j in range(len(sampling_res[i])):
                sampling_res[i][j] = signal(j * Ts + offsets[i] * T)

    elif scheme == "discrete":
        col = int(np.floor(len(signal) / n_bands))
        sampling_res = np.zeros((n_chs, col), dtype=np.cdouble)
        for i in range(n_chs):
            sampling_res[i] = signal[offsets[i] :: n_bands][:col]
    else:
        raise NotImplementedError("scheme not supported")

    return sampling_res, offsets


def dft(signals: np.ndarray, offsets: list[int], nBands: int):
    """
    Transforms signals sampled by multicoset method into frequency domain

    #### returns

    1.Multicoset sampling results in frequency domain, say, Y
    2. The measurement matrix A with shape(nChannels, nBands)
    """
    # DTFT * Ts = DTF, so the coeffecient 1/LT will be eleminated
    # To get matrix Y, we need to use the coeffecients to multiply the fft
    # results.

    Y_aux = np.array(offsets).reshape(-1, 1) @ np.arange(signals.shape[-1]).reshape(1, -1)
    YCoeff = np.exp(-2j * np.pi / (nBands * signals.shape[-1]) * Y_aux)

    # amplitudes got by numpy.fft.fft are N times larger than actual amplitudes
    # in frequency domain. We don't shrink here considering float point
    # arithmetic errors
    Y = YCoeff * np.fft.fft(signals)
    # Y = np.matrix(np.multiply(YCoeff, np.fft.fft(signals)))

    A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, nBands, 1))
    A = np.matrix(np.exp(2j * np.pi * A_aux / nBands))

    # to compensate the amplitudes caused by numpy.fft.fft(signals)
    A = signals.shape[-1] * A
    return Y, A

def dft_torch(signals: torch.Tensor, offsets: list[int], nBands: int):
    """
    Transforms signals sampled by multicoset method into the frequency domain.

    #### Returns
    1. Multicoset sampling results in the frequency domain (Y).
    2. The measurement matrix A with shape (nChannels, nBands).
    """

    device = signals.device  # Ensure all operations run on the same device

    # Convert offsets to a PyTorch tensor
    offsets = torch.tensor(offsets, dtype=torch.float32, device=device).reshape(-1, 1)
    
    # Create frequency domain transformation matrix
    nSamples = signals.shape[-1]
    time_indices = torch.arange(nSamples, device=device).reshape(1, -1).float()
    YCoeff = torch.exp(-2j * torch.pi / (nBands * nSamples) * (offsets @ time_indices))

    # Compute FFT (PyTorch's FFT preserves scale)
    Y = YCoeff * torch.fft.fft(signals, dim=-1)

    # Construct measurement matrix A
    freq_indices = torch.arange(0, nBands, device=device).reshape(1, -1).float()
    A = torch.exp(2j * torch.pi * (offsets @ freq_indices) / nBands)

    # Scale A to compensate for FFT amplitude differences
    A = nSamples * A

    return Y, A.to(torch.cdouble)

def get_measurement_matrix(nBands: int, offsets: list[int]):
    A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, nBands, 1))
    A = np.matrix(np.exp(2j * np.pi * A_aux / nBands))
    return A


def get_correlation(A: np.matrix) -> float:
    corr = A.H * A

    for i in range(corr.shape[0]):
        corr[i, i] = 0

    return np.max(np.abs(corr)) / (np.linalg.norm(A) ** 2)


def find_best_offsets():
    def combine(n, k):
        result = []
        current = []
        backtrack(result, current, n, k, 0)
        return result

    def backtrack(result, current, n, k, start):
        if len(current) == k:
            result.append(current[:])
            return

        for i in range(start, n):
            current.append(i)
            backtrack(result, current, n, k, i + 1)
            current.pop()

    all_offsets = combine(40, 4)
    nBands = 40
    min_offsets = None
    min_corr = np.inf
    for offsets in all_offsets:
        A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, nBands, 1))
        A = np.matrix(np.exp(2j * np.pi * A_aux / nBands))
        corr = get_correlation(A)
        if corr < min_corr:
            min_corr = corr
            min_offsets = offsets

    print(min_offsets)
    print(min_corr)

if __name__ == "__main__":
    find_best_offsets()
