import numpy as np
import torch

CNDarray = np.ndarray[int, np.dtype[np.cdouble]]


def load_matlab(path: str, ratio=1.0):
    def rep(s):
        return complex(s.decode().replace("i", "j"))

    s = np.loadtxt(
        path,
        dtype=np.cdouble,
        converters={0: rep},
    )

    sz = int(np.floor(ratio * len(s)))

    return np.array(s[0:sz])


def add_noise(rawS: np.ndarray, noise=0.0, random_seed=1):
    if noise == 0.0:
        # print("No noise signals added, SNR=inf")
        return rawS, np.inf

    # print("Adding noise")
    sz = len(rawS)
    sPsqrt = np.linalg.norm(rawS)

    # n = np.random.normal(size=(sz), scale=np.sqrt(noise/2)) + \
    #         1j * np.random.normal(size=(sz), scale=np.sqrt(noise/2))

    rng = np.random.RandomState(random_seed)
    n = rng.normal(size=(sz), scale=np.sqrt(noise / 2)) + 1j * rng.normal(
        size=(sz), scale=np.sqrt(noise / 2)
    )

    nPsqrt = np.linalg.norm(n)
    snr = 20 * np.log(sPsqrt / nPsqrt)
    # print("SNR:", snr)

    return rawS + n, snr


def awgn(pure: np.ndarray, snr: float | None, random_seed=None) -> CNDarray:
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

def awgn_torch(pure: torch.Tensor, snr: float | None, random_seed=None) -> torch.Tensor:
    if snr is None:
        return pure

    # Compute signal power (root mean square)
    sig_p_sqrt = torch.sqrt(torch.mean(torch.abs(pure) ** 2))

    # Compute noise scaling factor
    noise_scale = sig_p_sqrt / (10 ** (snr / 20)) / torch.sqrt(torch.tensor(2.0, device=pure.device))

    # Set random seed for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Generate complex Gaussian noise
    n = torch.randn_like(pure, dtype=torch.float32) + 1j * torch.randn_like(pure, dtype=torch.float32)
    
    # Scale noise
    n = n * noise_scale

    return pure + n


def awgn_scale(pure: np.ndarray, snr: float | None) -> CNDarray:
    if snr is None:
        return pure

    # sig_p_sqrt = np.linalg.norm(pure) / np.sqrt(sz)
    sig_p_sqrt = np.sqrt(np.mean(np.abs(pure) ** 2))
    noise_scale = sig_p_sqrt / (10 ** (snr / 20)) / np.sqrt(2)

    return noise_scale
