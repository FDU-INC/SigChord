import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from ..wifi.mat_wave_struct import WaveStruct
from ..wifi import viterbi
from ..prelude import CNDarray


def indices_of_symbol(n: int):
    return np.arange(64) + 400 + n * 80 + 16

def autocorrelate(waveform: CNDarray, ws: int, n_st: int, n: int):
    rs = np.sum(waveform[n : n + ws] * waveform[n + n_st : n + n_st + ws].conj())
    d = np.sum(waveform[n : n + ws] * waveform[n : n + ws].conj())
    return rs / d

def find_start(waveform: CNDarray, fs: float) -> np.intp:
    n_st = int(round(0.8e-6 * fs))
    rs = [autocorrelate(waveform, 9 * n_st, n_st, n) for n in range(10 * n_st)]
    id = np.argmax(np.abs(rs))
    return id


def sig_raw_bits(waveform: CNDarray, bw: float, oversampling_rate: float, start_id: np.intp | int, field: str):
    sig_start = 0
    sig_end = 0
    fs = bw * oversampling_rate
    if field == "L-SIG":
        sig_start = int(round(16.8e-6 * fs))
        sig_end = int(round(20e-6 * fs))
    elif field == "HT-SIG1" or field == "VHT-SIG-A1":
        sig_start = int(round(20.8e-6 * fs))
        sig_end = int(round(24e-6 * fs))
    elif field == "HT-SIG2" or field == "VHT-SIG-A2":
        sig_start = int(round(24.8e-6 * fs))
        sig_end = int(round(28e-6 * fs))
    else:
        raise ValueError(f"field \"{field}\" not valid")

    sig_waveform = waveform[start_id + np.arange(sig_start, sig_end)]
    sig_f = np.fft.fft(sig_waveform)

    n_20mhz = int(round(bw) / 20e6)
    symbs_start = -32 * n_20mhz
    symbs_end = symbs_start + 64
    symbs = (
        sig_f[symbs_start:symbs_end]
        if symbs_end < 0
        else np.concatenate((sig_f[symbs_start:], sig_f[:symbs_end]))
    )

    symbs = np.roll(symbs, 32)
    # remove pilot
    symbs = np.concatenate(
        (
            symbs[-26:-21],
            symbs[-20:-7],
            symbs[-6:],
            symbs[1:7],
            symbs[8:21],
            symbs[22:27],
        )
    )

    if field == "L-SIG" or field == "VHT-SIG-A1":
        return np.array(symbs.real > 0, dtype=np.uint8)
    else:
        return np.array(symbs.imag > 0, dtype=np.uint8)

def interleave(raw_bits: np.ndarray, n_bpsc: int):
    # carried bits per symbol
    n_cbps = len(raw_bits)
    s = max(n_bpsc // 2, 1)
    k_ids = np.arange(n_cbps)
    i_ids = int(n_cbps / 16) * (k_ids % 16) + k_ids // 16
    j_ids = s * (i_ids // s) + (i_ids + n_cbps - 16 * i_ids // n_cbps) % s
    res = np.zeros_like(raw_bits, dtype=int)
    res[j_ids] = raw_bits
    return res


def deinterleave(raw_bits: np.ndarray, n_bpsc: int):
    """
    # parameters

    - `raw_bits`: interleaved bits
    - `n_bpsc`: number of bits per carrier. For example, n_bpsc = 2 for QPSK

    # return

    deinterleaved bits
    """
    # carried bits per symbol
    n_cbps = len(raw_bits)
    s = max(n_bpsc // 2, 1)
    j_ids = np.arange(n_cbps)
    i_ids = s * (j_ids // s) + (j_ids + 16 * j_ids // n_cbps) % s
    k_ids = 16 * i_ids - (n_cbps - 1) * (16 * i_ids // n_cbps)
    res = np.zeros_like(raw_bits, dtype=int)
    res[k_ids] = raw_bits
    return res


def parse_lsig_bits(bits: np.ndarray):
    parity = bits[:17].sum() % 2
    if parity != bits[17]:
        raise ValueError("signal parity check failed")

    rate = bits[:4]
    print("RATE:", rate)
    length_bits = np.concatenate((bits[5:17], np.zeros(4, dtype=np.uint8)))
    byte_array = np.packbits(length_bits[::-1])
    length = int.from_bytes(byte_array, "big")
    print("LENGTH:", length)

    print("If VT/VHT: ")
    # equation (21-24), page 3054
    # For HT-mixed formats, specify the transmission time as described in
    # Sections 19.3.9.3.5 and 10.27.4 of IEEE Std 802.11-2020.
    duration = (length + 3) / 3 * 4 + 20
    print(f"duration: {duration} us")

    print("If HE: ")
    duration = (length + 3 + 2) / 3 * 4 + 20
    print(f"duration: {duration} us")


def test_interleave_signal():
    wave_struct = WaveStruct("./wifi/normal.mat")
    raw_bits = wave_struct.sig_raw_bits(0, "L-SIG")
    raw_bits = np.tile(raw_bits, 1)
    print("received raw bits:")
    print(raw_bits)
    deinterleaved_bits = deinterleave(raw_bits, n_bpsc=1)
    print("deinterleaved bits:")
    print(deinterleaved_bits)
    reinterleaved_bits = interleave(deinterleaved_bits, n_bpsc=1)
    assert np.all(reinterleaved_bits == raw_bits)
    print("test interleave pass")
    print()


def test_decode_signal():
    file = "./sigproc/wifi/mcs3_501.mat"
    # file = "./sigproc/wifi/40ht_mcs2_1024.mat"
    # file = "./wifi/vht_default.mat"
    # file = "./wifi/40he_default.mat"
    print(file)
    wave_struct = WaveStruct(file)
    print(f"duration: {len(wave_struct.waveform) / wave_struct.fs * 1e6} us")
    raw_bits = wave_struct.sig_raw_bits(0, "L-SIG")
    print(raw_bits)
    deinterleaved_bits = deinterleave(raw_bits, n_bpsc=1)
    decoded_bits = viterbi.decode(deinterleaved_bits)
    print(decoded_bits)
    parse_lsig_bits(decoded_bits)

def test_decode_nonHT(waveform: CNDarray):
    fs = 20e6
    start_id = find_start(waveform, fs)
    raw_bits = sig_raw_bits(waveform, fs, 1, start_id, "L-SIG")
    deinterleaved_bits = deinterleave(raw_bits, n_bpsc=1)
    decoded_bits = viterbi.decode(deinterleaved_bits)
    print(decoded_bits)
    parse_lsig_bits(decoded_bits)

def test_htsig():
    file = "./wifi/vht_default.mat"
    # file = "./wifi/40ht_mcs2_1024.mat"
    print(file)
    wave_struct = WaveStruct(file)
    print(f"duration: {len(wave_struct.waveform) / wave_struct.fs * 1e6} us")
    raw_bits1 = wave_struct.sig_raw_bits(0, "VHT-SIG-A1")
    raw_bits2 = wave_struct.sig_raw_bits(0, "VHT-SIG-A2")
    deinterleaved_bits1 = deinterleave(raw_bits1, n_bpsc=1)
    deinterleaved_bits2 = deinterleave(raw_bits2, n_bpsc=1)
    deinterleaved_bits = np.concatenate((deinterleaved_bits1, deinterleaved_bits2))
    decoded_bits = viterbi.decode(deinterleaved_bits)
    print(decoded_bits[:24])
    print(decoded_bits[24:])
    # parse_signal_bits(decoded_bits)


def main():
    # wave_struct = WaveStruct("./sigproc/wifi/wifi-non-HT.mat")
    # waveform = awgn(wave_struct.waveform * 1.0, None)
    waveform = scipy.io.loadmat("./sigproc/wifi/wifi-non-HT.mat")["res"].T[0].flatten()
    # start_id = wave_struct.find_start()
    # print(f"start id: {start_id}")
    start_id = 0
    # matlab is good
    assert start_id == 0

    waveform = waveform[start_id:]
    signal_start = 320 + 16
    signal_end = 400
    print(f"signal start: {signal_start}")
    print(f"signal end: {signal_end}")
    signal = waveform[signal_start : signal_end]
    signal_f = np.fft.fft(signal)
    signal_f = np.concatenate((signal_f[-26:], signal_f[1:26]))
    plt.plot(signal_f.real)
    plt.plot(signal_f.imag)
    plt.show()
    plt.plot(signal_f.real, signal_f.imag, ".")
    plt.show()

    for n in range(5):
        symbol = waveform[indices_of_symbol(n)] * 1.0
        symbol = np.fft.fft(symbol)
        symbol = np.concatenate(
            (
                symbol[-26:-21],
                symbol[-20:-7],
                symbol[-6:],
                symbol[1:7],
                symbol[8:21],
                symbol[22:26],
            )
        )
        plt.plot(symbol.real, symbol.imag, "b.")
    plt.show()


if __name__ == "__main__":
    # test_htsig()
    # test_interleave_signal()
    test_decode_signal()
    # main()
    # waveform = scipy.io.loadmat("./sigproc/wifi/wifi-non-HT.mat")["res"].T[0].flatten()
    # test_decode_nonHT(waveform)
