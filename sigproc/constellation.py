import numpy as np


LIST_QPSK = np.exp(1j * np.pi * np.array([1, 3, 7, 5]) / 4)

LIST_8PSK = np.exp(1j * np.pi / 8) * np.exp(1j * np.pi * np.array([1,0,4,5,2,7,3,6]) / 4)

_LONG = 0.707107
_SHORT = 0.235702
_16QAM_INPHASE = np.array([
    _SHORT,_LONG,_SHORT,_LONG,
    _SHORT,_SHORT,_LONG,_LONG,
    -_SHORT,-_SHORT,-_LONG,-_LONG,
    -_SHORT,-_LONG,-_SHORT,-_LONG,
])
_16QAM_QUADRATURE = np.array([
    _SHORT,_SHORT,_LONG,_LONG,
    -_SHORT,-_LONG,-_SHORT,-_LONG,
    _SHORT,_LONG,_SHORT,_LONG,
    -_SHORT,-_SHORT,-_LONG,-_LONG,
])
LIST_16QAM = _16QAM_INPHASE + 1j * _16QAM_QUADRATURE

# https://en.wikipedia.org/wiki/File:Circular_8QAM.svg
_INNER_8QAM = 1 / (1 + np.sqrt(3))
LIST_8QAM = np.array([
    (1 + 1j) * _INNER_8QAM, -1, 1j, (-1 + 1j) * _INNER_8QAM,
    1, (-1 -1j) * _INNER_8QAM, (1 - 1j) * _INNER_8QAM, -1j,
])

MOD_LIST_DICT = {
    "QPSK": LIST_QPSK,
    "8PSK": LIST_8PSK,
    "8QAM": LIST_8QAM,
    "16QAM": LIST_16QAM,
}

MOD_ID = {
    "QPSK": 0,
    "8PSK": 1,
    "8QAM": 2,
    "16QAM": 3,
}

MOD_NAME = {
    0: "QPSK",
    1: "8PSK",
    2: "8QAM",
    3: "16QAM",
}

MOD_ORDER = {
    "QPSK": 2,
    "8PSK": 3,
    "8QAM": 3,
    "16QAM": 4,
}
