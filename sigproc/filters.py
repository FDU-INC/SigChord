# filter and pulse shape
import numpy as np
from abc import abstractmethod, ABC

from .prelude import CNDarray

class Filter(ABC):
    @abstractmethod
    def __call__(self, signal: CNDarray, fc: float) -> CNDarray:
        raise NotImplementedError

    @abstractmethod
    def pulse_shape(self, symbols: CNDarray, fc: float) -> CNDarray:
        raise NotImplementedError


class SincFilter(Filter):
    def __init__(self, fsymb: float, sps: int, window_sz_in_symbols: int = 10) -> None:
        super().__init__()
        self.window_sz = window_sz_in_symbols * sps
        self.fsymb = fsymb
        self.sps = sps
        self.fsamp = sps * fsymb

        parity = self.window_sz % 2
        pulse_len = self.window_sz + 1 - parity
        self.pulse_len = pulse_len

        # len(self.__pulse) is 2 * half + 1
        self.pulse = np.sinc((np.arange(0, pulse_len) - int(pulse_len / 2)) / sps)
        self.pulse = np.array(self.pulse, dtype=np.cdouble)
        self.pulse /= np.linalg.norm(self.pulse, 2)

    def __call__(self, signal: CNDarray, fc: float) -> CNDarray:
        # the peek of the first symbol is at len(pulse) / 2, and the pulse
        # representing the first symbol is centered at the peek with length of sps
        # start = int((self.pulse_len - self.sps) / 2)
        # start = 0
        # signal = signal[start : (start + len(signal))]
        # shift from fc to baseband
        sig_shifted = np.multiply(
            signal, np.exp(-2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )
        sig_shifted = np.convolve(sig_shifted, self.pulse, "full")
        # here start is different from pulse shaping
        start = int((self.pulse_len) / 2)
        sig_shifted = sig_shifted[start : (start + len(signal))]

        return sig_shifted

    def pulse_shape(self, symbols: CNDarray, fc: float) -> CNDarray:
        # upsample
        signal = np.zeros(len(symbols) * self.sps, dtype=np.cdouble)
        signal[:: self.sps] = symbols
        # convolved signal has length of int(pulse_len / 2) * 2 + len(symbols) * sps
        signal = np.convolve(signal, self.pulse, "full")

        # the peek of the first symbol is at len(pulse) / 2, and the pulse
        # representing the first symbol is centered at the peek with length of sps
        start = int((self.pulse_len - self.sps) / 2)
        signal = signal[start : (start + len(symbols) * self.sps)]
        # shift to fc
        sig_shifted = np.multiply(
            signal, np.exp(2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )

        return sig_shifted


class SqrtRaisedCos:
    def __init__(self, fsymb: float, rolloff: float):
        self.f = fsymb
        self.alpha = rolloff

    def __call__(self, t) -> float:
        return self.time(t)

    def time(self, t) -> float:
        res = 0
        if t == 0:
            # res = (1 + self.alpha * (4 / np.pi - 1)) * self.f
            res = 1 - self.alpha + 4 * self.alpha / np.pi
        elif t == 1 / (4 * self.alpha * self.f) or t == -1 / (4 * self.alpha * self.f):
            res = (
                self.alpha
                # * self.f
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * self.alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * self.alpha))
                )
            )
        else:
            term1 = np.sin(np.pi * self.f * t * (1 - self.alpha))
            term2 = (
                4
                * self.alpha
                * self.f
                * t
                * np.cos(np.pi * self.f * t * (1 + self.alpha))
            )
            term3 = np.pi * t * self.f * (1 - (4 * self.alpha * t * self.f) ** 2)

            res = (term1 + term2) / term3

        return res

    def fre(self, f: float) -> float:
        unsqrt = 0.0
        f_cmp = np.abs(f) * 2 / self.f
        if f_cmp <= (1 - self.alpha):
            unsqrt = 1 / self.f
        elif f_cmp > (1 - self.alpha) and f_cmp <= (1 - self.alpha):
            unsqrt = (
                1
                / (2 * self.f)
                * (1 - np.sin(np.pi * (f - self.f / 2) / (self.alpha * self.f)))
            )

        return np.sqrt(unsqrt)


class SqrtRaisedCosFilter(Filter):
    def __init__(
        self, fsymb: float, rolloff: float, sps: int, window_sz_in_symbols: int = 10
    ):
        super().__init__()
        self.sqrt_raised_cos = SqrtRaisedCos(fsymb, rolloff)
        self.window_sz = window_sz_in_symbols * sps
        self.fsymb = fsymb
        self.sps = sps
        fsamp = fsymb * sps
        self.fsamp = fsamp
        self.rolloff = rolloff

        parity = self.window_sz % 2
        pulse_len = self.window_sz + 1 - parity
        self.pulse_len = pulse_len

        # len(self.__pulse) is 2 * half + 1
        half = int(pulse_len / 2)
        self.pulse = np.array(
            [self.sqrt_raised_cos((i - half) / fsamp) for i in range(pulse_len)]
        )
        self.pulse /= np.linalg.norm(self.pulse, 2)

    def __call__(self, signal: CNDarray, fc: float) -> CNDarray:
        sig_shifted = np.multiply(
            signal, np.exp(-2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )
        sig_shifted = np.convolve(sig_shifted, self.pulse, "full")
        # here start is different from pulse shaping
        start = int((self.pulse_len) / 2)
        sig_shifted = sig_shifted[start : (start + len(signal))]
        return sig_shifted

    def filter_matlab(self, input: np.ndarray) -> np.ndarray:
        signal = np.convolve(input, self.pulse, "full")
        return signal[: len(input)]

    def pulse_shape(self, symbols: CNDarray, fc: float) -> CNDarray:
        signal = np.zeros(len(symbols) * self.sps, dtype=np.cdouble)
        signal[:: self.sps] = symbols
        signal = np.convolve(signal, self.pulse, "full")

        start = int((self.pulse_len - self.sps) / 2)
        signal = signal[start : (start + len(symbols) * self.sps)]

        sig_shifted = np.multiply(
            signal, np.exp(2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )
        return sig_shifted


class OFDMPseudoFilter(Filter):
    """
    Different from Sinc or Raised Cosine, this does not involve convolution.
    Filtering, pulse shaping are done by IFFT, FFT.
    """

    def __init__(self, fsymb: float, sps: int) -> None:
        super().__init__()
        self.fsymb = fsymb
        self.sps = sps
        self.fsamp = fsymb * sps

    def __call__(self, signal: CNDarray, fc: float) -> CNDarray:
        sig_shifted = np.multiply(
            signal, np.exp(-2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )

        sig_f = np.fft.fft(sig_shifted) / np.sqrt(len(signal))
        n_carriers = int(len(signal) / self.sps)
        sig_f = np.roll(sig_f, n_carriers // 2)
        return sig_f

    def pulse_shape(self, symbols: CNDarray, fc: float) -> CNDarray:
        """
        cyc prefix is not implemented
        """
        symbs_up = np.zeros((len(symbols) * self.sps), dtype=np.cdouble)
        symbs_up[: len(symbols)] = symbols
        symbs_up = np.roll(symbs_up, -int(len(symbols) / 2))
        sig = np.fft.ifft(symbs_up) * np.sqrt(len(symbs_up))
        sig_shifted = np.multiply(
            sig, np.exp(2j * np.pi * fc / self.fsamp * np.arange(len(sig)))
        )
        return sig_shifted


class SqrtRaisedCosFilterMat(SqrtRaisedCosFilter):
    """
    behaves the same as comm.RaisedCosineReceiveFilter in MATLAB
    """
    def __init__(
        self, fsymb: float, rolloff: float, sps: int, window_sz_in_symbols: int = 10
    ):
        super().__init__(fsymb, rolloff, sps, window_sz_in_symbols)
        self.pulse = self.pulse / np.linalg.norm(self.pulse, 2)
        self.gain = 1.0
        self.reset()

    def reset(self):
        self.buffer = np.zeros(self.pulse_len, dtype=np.cdouble)

    def set_gain(self, gain: float):
        self.gain = gain

    def __call__(self, x: np.ndarray[int, np.dtype[np.cdouble]] | np.cdouble):
        x = np.atleast_1d(np.array(x, dtype=np.cdouble))
        assert len(x.shape) == 1

        unfiltered = np.concatenate((self.buffer, x), axis=-1)
        filtered = self.gain * np.convolve(unfiltered, self.pulse, "full")
        buf_len = self.pulse_len
        res = filtered[buf_len: buf_len + len(x)]
        self.buffer = unfiltered[-buf_len:]

        return res
