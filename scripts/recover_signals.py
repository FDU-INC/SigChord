import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import h5py
import os
import torch.utils.benchmark as benchmark

from models.spectrum_sensor import Net as SpectrumSensor
from lse_solver import LSESolver
import sigproc.cs.multicoset as M
import utils

class Reconstructor(nn.Module):
    N_SAMPLE = 2400
    SAMPLING_RATIO = 40
    def __init__(self, spectrum_sensor: SpectrumSensor) -> None:
        super().__init__()
        self.spectrum_sensor = spectrum_sensor
        self.spectrum_sensor.eval()
        self.n_coset = spectrum_sensor.n_coset
        self.total_band = self.SAMPLING_RATIO

        offsets = M.DEFAULT_OFFSETS[:self.n_coset]
        if M.DEFAULT_BEST_OFFSETS.get(self.n_coset) is not None:
            offsets = M.DEFAULT_BEST_OFFSETS[self.n_coset]
        self.lse_solver = LSESolver(self.n_coset, offsets)

    def recover(self, x: Tensor, threshold: float=None) -> Tensor:
        assert x.shape[-1] == self.spectrum_sensor.n_coset * 2, f"input shape {x.shape} does not match n_coset {self.spectrum_sensor.n_coset}"
        support = self.spectrum_sensor(x)

        if threshold is None:
            support = (support > 0).long()
        else:
            support = (torch.sigmoid(support) > threshold).long()

        Xs = self.lse_solver.recover_with_support(x, support)

        # So far we have already recovered all the narrowband signals that can
        # be forwarded to next stages. To calculate the MSE, we need to
        # reorganize these narrowband signals into high-speed multiband signals.
        X = (
            torch.zeros((x.shape[0], self.total_band, x.shape[1]), dtype=torch.complex64)
        )
        # Allocate X is really slow......
        sup = torch.argwhere(support)
        X[sup[:, 0], sup[:, 1]] = Xs.to(X.device)
        X = torch.flatten(X, start_dim=-2)
        Xt = torch.fft.ifft(X, dim=-1)
        return Xt

    def timeit(self, data: DataLoader, runs: int, threshold: float | None = None) -> None:
        for x, _ in data:
            x = x.cuda().float()
            assert x.shape[-1] == self.spectrum_sensor.n_coset * 2, f"input shape {x.shape} does not match n_coset {self.spectrum_sensor.n_coset}"
            support = self.spectrum_sensor(x)

            if threshold is None:
                support = (support > 0).long()
            else:
                support = (torch.sigmoid(support) > threshold).long()

            timer = benchmark.Timer(
                stmt="self.lse_solver.recover_with_support(x, support)",
                globals={"self": self, "x": x, "support": support},
            )
            results = [timer.timeit(1).mean for _ in range(runs)]
            print(f"{runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")

            return

def test():
    n_coset = 6
    spectrum_sensor = SpectrumSensor(
        n_coset=n_coset,
        n_band=16,
        d_model=512,
        n_layer=1,
        n_fold=16
    )

    reconstructor = Reconstructor(spectrum_sensor)

    x = torch.randn(3, 2400, 2 * n_coset)
    x_rec = reconstructor.recover(x)
    print(x_rec.shape)

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--batch_size", type=int, default=1024, help="default 1024")
    parser.add_argument("--cosets", type=int, help="number of sampling cosets")
    parser.add_argument("--test", type=str, help="test dataset")
    parser.add_argument("--params", type=str, help="path of spectrum sensor parameters")
    args = parser.parse_args()
    print("args:")
    print(args)
    return args

def load_data(args: argparse.Namespace):
    f_test = h5py.File(str(args.test), "r")
    test_len = 5000

    n_coset = args.cosets
    # align with __init__ in Reconstructor
    indices = range(2*args.cosets)
    if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
        indices_c = M.DEFAULT_OFFSETS_SORTED_IDS[M.DEFAULT_BEST_OFFSETS[n_coset]]
        indices_c = 2 * np.array(indices_c)
        indices = np.zeros(2 * len(indices_c), dtype=np.uint8)
        indices[::2] = indices_c
        indices[1::2] = indices_c + 1


    data_test = f_test["waveforms"][-test_len:, :, indices]
    data_test = torch.from_numpy(data_test)
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    print("data_test shape", data_test.shape)

    original_signal = f_test["waveforms"][-test_len:, :, :]
    original_signal = original_signal[..., 0::2] + 1j * original_signal[..., 1::2]
    original_signal = original_signal[..., M.DEFAULT_OFFSETS_SORTED_IDS]
    original_signal = torch.from_numpy(original_signal).flatten(-2, -1)

    test_ds = TensorDataset(data_test, original_signal)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return test_dl

def mse(x: Tensor, x_gt: Tensor) -> Tensor:
    x /= torch.sqrt(torch.mean(torch.abs(x)**2, dim=-1, keepdim=True))
    x_gt /= torch.sqrt(torch.mean(torch.abs(x_gt)**2, dim=-1, keepdim=True))
    np.save("x.npy", x.cpu().numpy())
    np.save("x_gt.npy", x_gt.cpu().numpy())
    return torch.mean(torch.abs(x - x_gt)**2)

def main():
    args = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(22)
    print("===> loading model")
    spectrum_sensor = SpectrumSensor(
        n_coset=args.cosets,
        n_band=16,
        d_model=128,
        n_layer=2,
        n_fold=16
    ).cuda()
    spectrum_sensor.load_state_dict(torch.load(args.params))
    reconstructor = Reconstructor(spectrum_sensor).cuda()

    print("===> loading data")
    data = load_data(args)
    print("===> data loaded")

    total = 0
    mses_sum = 0

    with torch.no_grad():
        reconstructor.eval()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        for x, x_gt in data:
            n = x.shape[0]
            x = x.cuda().float()
            x = utils.add_noise(x, 10)
            x_rec = reconstructor.recover(x)
            err = mse(x_rec, x_gt)
            print(err)
            mses_sum += err.item() * n
            total += n

        print(mses_sum / total)
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"usage: {(total_mem - free_mem) / 1024 / 1024} MB", flush=True)

        reconstructor.timeit(data, 20)

if __name__ == "__main__":
    main()