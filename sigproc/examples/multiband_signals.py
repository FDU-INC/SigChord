"""
use square root raised cosine pulse shaping to generate multiband signals
"""
import numpy as np
import h5py
import tqdm
import os
import sys

from ..filters import SqrtRaisedCosFilter
from ..cs import multicoset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    qpsk_symbols = np.exp(1j * np.pi * np.array([1, 3, 7, 5]) / 4)
    data_len = 1000
    n_symbs = 400
    n_sigs = 6
    n_bands = 16
    bw = 50e6
    n_bands_tot = 40
    # multicoset sampling channels
    n_chs = 40

    sps = int(round(bw * n_bands_tot / 20e6))
    filter = SqrtRaisedCosFilter(
        fsymb=20e6,
        rolloff=0.35,
        sps=sps,
    )

    symbols = np.random.randint(0, len(qpsk_symbols), (data_len, n_symbs))
    symbols = qpsk_symbols[symbols]
    r = np.random.uniform(0, 1, (data_len, n_sigs, n_symbs))
    phi = np.exp(1j * np.random.uniform(0, 2 * np.pi, (data_len, n_sigs, n_symbs)))
    symbols = r * phi

    sig_len = n_symbs * sps
    signals = np.zeros((data_len, int(sig_len / n_chs), 2 * n_chs), dtype=np.float32)
    occupied_bands = np.zeros((data_len, n_bands), dtype=np.uint8)
    for i in tqdm.tqdm(range(data_len)):
        bids = np.random.choice(n_bands, n_sigs, replace=False)
        occupied_bands[i][bids] = 1
        signal = np.zeros(sig_len, dtype=np.cdouble)
        for j, bid in enumerate(bids):
            fc = (bid + 0.5) * bw
            signal += filter.pulse_shape(symbols[i][j], fc)

        import matplotlib.pyplot as plt
        signal_f = np.fft.fft(signal)
        plt.plot(np.abs(signal_f))
        plt.show()

        # multicoset sampling
        sub_samples, _ = multicoset.sample(
            signal,
            0,
            0,
            n_bands_tot,
            n_chs,
            offsets=multicoset.DEFAULT_OFFSETS,
        )

        sub_samples = sub_samples.T
        signals[i, :, 0::2] = sub_samples.real
        signals[i, :, 1::2] = sub_samples.imag

    # then save signals and occupied_bands
    with h5py.File("./pkt_detect/multiband.h5", "w") as file:
        file.create_dataset("waveform", signals.shape, dtype=np.float32)
        file.create_dataset("label", occupied_bands.shape, dtype=np.uint8)
        file["waveform"][:] = signals
        file["label"][:] = occupied_bands


if __name__ == "__main__":
    main()
