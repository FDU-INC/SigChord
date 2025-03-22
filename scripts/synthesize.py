"""
Combine DVB-S2, non-HT Wi-Fi, HT Wi-Fi, to generate multiband signals.

Signals are segmented into frames. A frame has a duration of 48 microseconds.
Nyquist sampling (2 GSPS) of a frame contains 96,000 samples. Multi-coset
sampling (50 MSPS) of a frame contains samples of shape (2400, n_chs).

For each frame:

- every sub-band may contain noise, head, or payload
- signal on each sub-band is independent.
"""
import h5py
import numpy as np
import tqdm
import matlab.engine
import argparse

from sigproc.prelude import CNDarray
from sigproc.cs import multicoset as M

NYQ_RATE = 2e9
SUB_NYQ_RATE = 50e6

DVB_HEAD_LEN = int(4.5e-6 * NYQ_RATE)
NON_HT_HEAD_LEN = int(20e-6 * NYQ_RATE)
HT_HEAD_LEN = int(32e-6 * NYQ_RATE)

HEAD_LENS = [
    DVB_HEAD_LEN,
    NON_HT_HEAD_LEN,
    HT_HEAD_LEN,
]

GUARD_LENS = [
    DVB_HEAD_LEN // 5,
    NON_HT_HEAD_LEN // 5,
    # HT is special. Frame only without HT-STF should be regarded as non-HT
    HT_HEAD_LEN // 5,
]

FRAME_LEN = 96_000
# DATA_LEN = 50
N_SIG = 6
N_BAND = 16
SUB_BANDWIDTH = 50e6
MAX_BIT_LEN = 48

def link_start():
    print("loading matlab")
    eng = matlab.engine.start_matlab()
    eng.addpath("./scripts/matlab_ffi")
    assert eng is not None
    eng.run('./scripts/matlab_ffi/setupDvbs2.m', nargout=0)
    eng.run('./scripts/matlab_ffi/setupNonHT.m', nargout=0)
    eng.run('./scripts/matlab_ffi/setupHT.m', nargout=0)
    eng.run('./scripts/matlab_ffi/setupChannels.m', nargout=0)
    print("matlab loaded")
    return eng

class SignalGenerator():
    def __init__(self, batch: int, near_far: bool = False):
        self.batch = batch
        self.eng = link_start()
        self.dvbs2_cnt = batch - 1
        self.nonHT_cnt = batch - 1
        self.HT_cnt = batch - 1
        self.near_far = near_far

    def one(self, type_id):
        res = {}
        _one = {0: self.one_dvbs2, 1: self.one_NonHT, 2: self.one_HT}
        sig_info = _one[type_id]()
        waveform: CNDarray = np.array(sig_info["waveform"]).flatten()
        if self.near_far:
            waveform *= np.random.uniform(0.5, 1)
        length = int(sig_info["length"])
        waveform = waveform[:length]
        head_len = HEAD_LENS[type_id]
        bits = np.array(sig_info["bits"], dtype=np.int8).flatten()
        res["bits"] = bits
        res["type"] = type_id + 1

        signal = np.zeros(FRAME_LEN, dtype=np.csingle)
        if length < FRAME_LEN:
            # randomly put the waveform in the frame
            # we do not consider header split
            # it must contain the header
            #     |--H--|P|
            # |------F-----|
            offset = np.random.randint(FRAME_LEN - head_len)
            signal[offset:offset + length] = waveform[:FRAME_LEN - offset]
            res["signal"] = signal
            res["offset"] = offset
            return res

        # for convenient cutting
        waveform = np.concatenate((waveform, waveform))
        if np.random.randint(3) < 1:
            # payload, GL is doubled
            # |G|----F----|
            # |--H--|---P---|--H--|---P---|
            # offset_min = 2 * GL
            #
            #         |----F----|G|
            # |--H--|---P---|--H--|---P---|
            # offset_max + FL + 2 * GL = L + HL
            guard_len = GUARD_LENS[type_id]
            offset = np.random.randint(guard_len, length + head_len - FRAME_LEN - guard_len)
            signal[:2 * length - offset] = waveform[offset: offset + FRAME_LEN]

            if type_id == 2 and offset + FRAME_LEN >= length + NON_HT_HEAD_LEN:
                # -1 is for turning HT to non-HT
                # +1 is for turning file id to signal type id
                res["type"] = type_id - 1 + 1
                res["offset"] = length - offset
            else:
                res["type"] = 0
                res["offset"] = -1
                res["bits"] = np.full_like(bits, -1)
        else:
            # head
            #           |----F----|
            # |--H--|---P---|--H--|---P---|
            # offset_min + FL = L + HL
            #
            #               |----F----|
            # |--H--|---P---|--H--|---P---|
            # offset_max = L
            offset = np.random.randint(length + head_len - FRAME_LEN, length)
            signal = waveform[offset: offset + FRAME_LEN]
            res["offset"] = length - offset

        res["signal"] = signal
        return res

    def one_dvbs2(self):
        MODCOD_LEN = 5
        self.dvbs2_cnt = (self.dvbs2_cnt + 1) % self.batch
        if self.dvbs2_cnt == 0:
            self.dvbs2_sig_infos = self.eng.oneDvbs2(self.batch)
            for k in self.dvbs2_sig_infos.keys():
                self.dvbs2_sig_infos[k] = np.array(self.dvbs2_sig_infos[k])

        sig_info = {}
        sig_info["waveform"] = self.dvbs2_sig_infos["waveforms"][self.dvbs2_cnt]
        sig_info["fecFrame"] = self.dvbs2_sig_infos["fecFrames"][self.dvbs2_cnt]
        sig_info["hasPilots"] = self.dvbs2_sig_infos["hasPilots"][self.dvbs2_cnt]
        sig_info["length"] = self.dvbs2_sig_infos["lengths"][self.dvbs2_cnt]
        sig_info["modcod"] = self.dvbs2_sig_infos["modcods"][self.dvbs2_cnt]

        modcod = sig_info.pop("modcod")
        bin_str = np.binary_repr(int(modcod), width=MODCOD_LEN)
        modcod_bits = list(bin_str)
        fec_frame = int(sig_info.pop("fecFrame"))
        has_pilots = int(sig_info.pop("hasPilots"))
        bits = modcod_bits + [fec_frame, has_pilots]
        bits = np.array(bits, dtype=np.uint8)
        sig_info["bits"] = bits

        return sig_info

    def one_NonHT(self):
        self.nonHT_cnt = (self.nonHT_cnt + 1) % self.batch
        if self.nonHT_cnt == 0:
            self.nonHT_sig_infos = self.eng.oneNonHT(self.batch)
            for k in self.nonHT_sig_infos.keys():
                self.nonHT_sig_infos[k] = np.array(self.nonHT_sig_infos[k])
        sig_info = {}
        sig_info["waveform"] = self.nonHT_sig_infos["waveforms"][self.nonHT_cnt]
        sig_info["length"] = int(self.nonHT_sig_infos["lengths"][self.nonHT_cnt])
        sig_info["bits"] = self.nonHT_sig_infos["bits"][self.nonHT_cnt]
        return sig_info

    def one_HT(self):
        self.HT_cnt = (self.HT_cnt + 1) % self.batch
        if self.HT_cnt == 0:
            self.HT_sig_infos = self.eng.oneHT(self.batch)
            for k in self.HT_sig_infos.keys():
                self.HT_sig_infos[k] = np.array(self.HT_sig_infos[k])
        sig_info = {}
        sig_info["waveform"] = self.HT_sig_infos["waveforms"][self.HT_cnt]
        sig_info["length"] = int(self.HT_sig_infos["lengths"][self.HT_cnt])
        sig_info["bits"] = self.HT_sig_infos["bits"][self.HT_cnt]
        return sig_info

    def channel(self, signal):
        signal = signal.reshape(-1, 1)
        signal = self.eng.funcChannel(signal, self.eng.workspace['channels'])
        signal = np.array(signal).flatten()
        signal /= np.sqrt((np.abs(signal) ** 2).mean())
        return signal

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
    # np.random.seed(38)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--out", type=str, default="data/synthetic.h5")
    argparser.add_argument("--data_len", type=int)
    args = argparser.parse_args()
    assert args.data_len is not None
    DATA_LEN = args.data_len

    siggen = SignalGenerator(4, near_far=True)
    fcs = (np.arange(N_BAND) + 0.5) * SUB_BANDWIDTH
    carriers = [np.exp(2j * np.pi * fc / NYQ_RATE * np.arange(FRAME_LEN)) for fc in fcs]
    carriers = np.array(carriers)

    sub_band_info = np.full((DATA_LEN, N_BAND), -1, dtype=np.int8)
    all_bits = np.full((DATA_LEN, N_BAND, MAX_BIT_LEN), -1, dtype=np.int8)
    # signals = np.zeros((DATA_LEN, FRAME_LEN), dtype=np.csingle)
    signals = np.zeros((DATA_LEN, FRAME_LEN // len(M.DEFAULT_OFFSETS), 2 * len(M.DEFAULT_OFFSETS)), dtype=np.float16)
    offsets = np.full((DATA_LEN, N_BAND), -1, dtype=int)
    for i in tqdm.tqdm(range(DATA_LEN)):
        bids = np.random.choice(N_BAND, N_SIG, replace=False)
        sig_types = np.random.choice(len(HEAD_LENS), N_SIG, replace=True)
        multiband_signal = np.zeros(FRAME_LEN, dtype=np.csingle)
        for j in range(N_SIG):
            sig_info = siggen.one(sig_types[j])
            multiband_signal += sig_info["signal"] * carriers[bids[j]]
            sub_band_info[i][bids[j]] = sig_info["type"]
            bits = sig_info["bits"]
            all_bits[i, bids[j], :len(bits)] = bits
            offsets[i, bids[j]] = sig_info["offset"]

        # apply channel
        multiband_signal = siggen.channel(multiband_signal)
        # apply multicoset sampling
        multiband_signal = siggen.nyq2sub(multiband_signal)
        signals[i] = multiband_signal

    with h5py.File(args.out, "w") as file:
        file.create_dataset("waveforms", signals.shape, dtype=signals.dtype, data=signals)
        file.create_dataset("bits", all_bits.shape, dtype=all_bits.dtype, data=all_bits)
        file.create_dataset("bands", sub_band_info.shape, dtype=sub_band_info.dtype, data=sub_band_info)
        file.create_dataset("offsets", offsets.shape, dtype=offsets.dtype, data=offsets)

if __name__ == "__main__":
    main()
