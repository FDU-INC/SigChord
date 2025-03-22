"""
generate dvbs2 signal of PLS part
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from .. import filters

G = np.array(
    [
        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ]
).T

SCRAMBLE_SEQ = np.array( [ 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, ])

def get_pls_bits_random():
    pls_code = np.random.randint(0, 2, 7).reshape(7, 1)
    y = G @ pls_code[:6] % 2
    y = np.vstack((y, (y + pls_code[-1]) % 2)).flatten()
    y = (y + SCRAMBLE_SEQ) % 2
    return y

def bits_to_symbols(bits: np.ndarray):
    symbols = np.zeros(len(bits), dtype=np.cdouble)
    symbols[::2] = (1 + 1j) / np.sqrt(2) * (1 - 2 * bits[::2])
    symbols[1::2] = (-1 + 1j) / np.sqrt(2) * (1 - 2 * bits[1::2])
    return symbols

def get_pulse_conv_matrix(pulse: np.ndarray, signal_len: int, uprate: int):
    pulse_len = len(pulse)
    out_len = pulse_len + uprate * (signal_len - 1)
    pulse_conv = np.zeros((out_len, signal_len), dtype=np.cdouble)
    for i in range(signal_len):
        pulse_conv[i * uprate: i * uprate + pulse_len, i] = pulse
    return pulse_conv

def test_depulse_shaping():
    sps = 8

    bits = get_pls_bits_random()
    symbols = bits_to_symbols(bits)
    filter = filters.SqrtRaisedCosFilter(20e6, 0.5, sps, window_sz_in_symbols=10)
    pulse_conv_matrix = get_pulse_conv_matrix(filter.pulse, len(symbols), sps)

    pulse_len = len(filter.pulse)
    pulse_D = np.zeros(len(filter.pulse))
    pulse_D[pulse_len//2::pulse_len] = 1
    pulse_D = get_pulse_conv_matrix(pulse_D, len(symbols), sps)

    pulse_conv_matrix1 = get_pulse_conv_matrix(filter.pulse, pulse_conv_matrix.shape[0], 1)
    pulse_conv_matrix1 = pulse_conv_matrix1[pulse_len//2:-(pulse_len//2), :]

    print(pulse_conv_matrix.shape, pulse_D.shape, pulse_conv_matrix1.shape)
    # plt.imshow((pulse_conv_matrix.T @ pulse_D).real)
    plt.imshow(((pulse_conv_matrix1 @ pulse_conv_matrix).T @ pulse_D).real)
    plt.show()
    signal = filter.pulse_shape(symbols, 0.0)

    dft = np.fft.fft(np.eye(len(signal))) / np.sqrt(len(signal))

def test_rank():
    n_monte_carlo = 1000
    sps = 2

    filter = filters.SqrtRaisedCosFilter(25e6, 0.35, sps, window_sz_in_symbols=10)

    ranks = []
    ranks_bits = []
    for _ in range(n_monte_carlo):
        X = np.zeros((40, 64 * sps), dtype=np.cdouble)
        X_bits = np.zeros((40, 64), dtype=np.cdouble)

        bids = np.random.choice(40, 12, replace=False)
        for bid in bids:
            bits = get_pls_bits_random()
            X_bits[bid] = 1 - 2 * bits
            symbols = bits_to_symbols(bits)
            signal = filter.pulse_shape(symbols, 0.0)
            signal_f = np.fft.fft(signal)
            X[bid] = signal_f

        ranks.append(np.linalg.matrix_rank(X))
        ranks_bits.append(np.linalg.matrix_rank(X_bits))

    # plot mean of ranks and std as the error bar, like hist
    plt.plot(ranks, fillstyle='none', marker='o', label='signal')
    plt.plot(ranks_bits, fillstyle='none', marker='x', label='bits')
    plt.legend()
    plt.show()
    plt.errorbar("signal", np.mean(ranks), yerr=np.std(ranks), fmt='o')
    plt.errorbar("bits", np.mean(ranks_bits), yerr=np.std(ranks_bits), fmt='x')
    plt.show()

def test_random_rank(n_monte_carlo):
    sps = 2

    filter = filters.SqrtRaisedCosFilter(25e6, 0.35, sps, window_sz_in_symbols=10)

    ranks = []
    ranks_random = []
    for _ in tqdm.tqdm(range(n_monte_carlo)):
        X = np.zeros((40, 64 * sps), dtype=np.cdouble)
        X_rand = np.zeros((40, 64 * sps), dtype=np.cdouble)

        bids = np.random.choice(40, 12, replace=False)
        for bid in bids:
            bits = get_pls_bits_random()
            symbols = bits_to_symbols(bits)
            signal = filter.pulse_shape(symbols, 0.0)
            signal_f = np.fft.fft(signal)
            X[bid] = signal_f

            rand_bits = np.random.randint(0, 2, 64)
            symbols = bits_to_symbols(rand_bits)
            signal = filter.pulse_shape(symbols, 0.0)
            signal_f = np.fft.fft(signal)
            X_rand[bid] = signal_f

        ranks.append(np.linalg.matrix_rank(X @ X.T.conj()))
        ranks_random.append(np.linalg.matrix_rank(X_rand))

    return ranks, ranks_random

def plot_ranks(ranks, ranks_random, n_monte_carlo):
    ranks = np.array(ranks)
    ranks_random = np.array(ranks_random)
    bins = np.arange(min(ranks.min(), ranks_random.min()), max(ranks.max(), ranks_random.max()) + 2)

# Define the width of the bars
    bar_width = 0.4

# Shift the positions for the second dataset to place bars side by side
    # bin_centers = bins[:-1] + 0.5
    # ranks_pos = bin_centers - bar_width / 2
    # ranks_random_pos = bin_centers + bar_width / 2
    ranks_pos = bins[:-1] - bar_width / 2
    ranks_random_pos = bins[:-1] + bar_width / 2

# Calculate the histograms (frequency counts)
    ranks_counts, _ = np.histogram(ranks, bins=bins)
    ranks_random_counts, _ = np.histogram(ranks_random, bins=bins)

# Plot histograms side by side
    plt.bar(ranks_pos, ranks_counts, width=bar_width, alpha=0.7, align="center", label='DVB PLS', color='skyblue')
    plt.bar(ranks_random_pos, ranks_random_counts, width=bar_width, alpha=0.7, align="center", label='random signal', color='salmon')

# Labels and title
    plt.xlabel('Rank', fontsize=24)
    plt.ylabel('Occurrences', fontsize=24)
    plt.title('Rank distribution', fontsize=24)

# Annotate bar values
    for x, y in zip(ranks_pos, ranks_counts):
        plt.text(x, min(y, 0.95 * n_monte_carlo), str(y), ha='center', va='bottom', color='blue', fontsize=20)

    for x, y in zip(ranks_random_pos, ranks_random_counts):
        plt.text(x, min(y, 0.95 * n_monte_carlo), str(y), ha='center', va='bottom', color='darkred', fontsize=20)

# Show legend and plot
    plt.xticks(np.arange(min(ranks), max(ranks) + 1), fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.show()

if  __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    n_monte_carlo = 10000
    ranks = np.load("ranks.npy")
    ranks_random = np.load("ranks_random.npy")
    # ranks, ranks_random = test_random_rank(n_monte_carlo)
    plot_ranks(ranks, ranks_random, n_monte_carlo)
    np.save("ranks.npy", ranks)
    np.save("ranks_random.npy", ranks_random)
