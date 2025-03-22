import numpy as np
import tqdm
import h5py

from sigproc.prelude import CNDarray
import sigproc.cs.multicoset as M

DATA_LEN = 500
FRAME_LEN = 96000
N_SIG = 6
N_BAND = 16
SUB_BANDWIDTH = 50e6
NYQ_RATE = 2e9

class SignalGenerator():
    def __init__(self):
        pass

    def one(self, bid: int) -> CNDarray:
        signal_f = np.zeros(FRAME_LEN, dtype=np.csingle)
        n_freq_bin_per_band = int(FRAME_LEN / (NYQ_RATE / SUB_BANDWIDTH))
        indices = range(bid * n_freq_bin_per_band, (bid + 1) * n_freq_bin_per_band)
        signal_f[indices] = np.random.randn(n_freq_bin_per_band) + 1j * np.random.randn(n_freq_bin_per_band)
        signal = np.fft.ifft(signal_f)
        scale = np.random.uniform(0.5, 1)
        return signal * scale

    def nyq2sub(self, signal):
        sub_samples, _ = M.sample(
            signal,
            0.0,
            0.0,
            n_bands=len(M.DEFAULT_OFFSETS),
            n_chs=len(M.DEFAULT_OFFSETS),
            offsets=M.DEFAULT_OFFSETS,
            scheme="discrete",
        )
        sub_samples = sub_samples.T
        res = np.zeros((sub_samples.shape[0], sub_samples.shape[1] * 2), dtype=np.float16)
        res[:, 0::2] = sub_samples.real
        res[:, 1::2] = sub_samples.imag
        return res


def main():
    np.random.seed(101)
    fcs = (np.arange(N_BAND) + 0.5) * SUB_BANDWIDTH
    siggen = SignalGenerator()

    sub_band_info = np.full((DATA_LEN, N_BAND), -1, dtype=np.int8)

    with h5py.File("./data/multi_random.h5", "w") as file:
        file.create_dataset("waveforms", (DATA_LEN, FRAME_LEN // len(M.DEFAULT_OFFSETS), 2 * len(M.DEFAULT_OFFSETS)), dtype=np.float16)
        file.create_dataset("bands", sub_band_info.shape, dtype=sub_band_info.dtype)

        for i in tqdm.tqdm(range(DATA_LEN)):
            multiband_signal = np.zeros((FRAME_LEN), dtype=np.csingle)
            bids = np.random.choice(N_BAND, N_SIG, replace=False)
            sub_band_info[i][bids] = 0
            for j in range(N_SIG):
                signal = siggen.one(bids[j])
                multiband_signal += signal


            file["waveforms"][i] = siggen.nyq2sub(multiband_signal)
        file["bands"][:] = sub_band_info
    # with h5py.File("./data/multi_random.h5", "w") as file:
    #     file.create_dataset("waveforms", signals.shape, dtype=signals.dtype, data=signals)
    #     file.create_dataset("bands", sub_band_info.shape, dtype=sub_band_info.dtype, data=sub_band_info)

if __name__ == "__main__":
    main()
