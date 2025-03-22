import h5py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.utils.benchmark as benchmark
import argparse

import sigproc.cs.multicoset as M

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spectrum_sensor", help="spectrum_sensor or wrt")
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--batch_size", type=int, default=1024, help="default 1024")
    parser.add_argument("--layers", type=int, default=2, help="default 2")
    parser.add_argument("--d_model", type=int, default=128, help="default 128")
    parser.add_argument("--cosets", type=int, help="number of sampling channels")
    parser.add_argument("--test", type=str, help="test dataset")
    parser.add_argument("--params", type=str, help="path of saved model parameters")
    args = parser.parse_args()
    print("args:")
    print(args)
    return args

def load_data(args: argparse.Namespace):
    f_test = h5py.File(str(args.test), "r")
    test_len = 5000

    indices = range(2*args.cosets)
    if M.DEFAULT_BEST_OFFSETS.get(args.cosets) is not None:
        indices_c = M.DEFAULT_OFFSETS_SORTED_IDS[M.DEFAULT_BEST_OFFSETS[args.cosets]]
        indices_c = 2 * np.array(indices_c)
        indices = np.zeros(2 * len(indices_c), dtype=np.uint8)
        indices[::2] = indices_c
        indices[1::2] = indices_c + 1

    data_test = f_test["waveforms"][-test_len:, :, indices]
    print("data_test shape", data_test.shape)
    data_test = torch.from_numpy(data_test).half()
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    label_test = f_test["bands"][-test_len:]
    label_test = torch.from_numpy(label_test).long()

    test_ds = TensorDataset(data_test, label_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return test_dl

def main():
    print(f"pid: {os.getpid()}", flush=True)
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(22)

    print("===> loading data")
    test_dl = load_data(args)

    print("===> loading model")
    if args.model == "spectrum_sensor":
        import models.spectrum_sensor as md
        model = md.Net(
            n_band=16,
            n_coset=args.cosets,
            d_model=args.d_model,
            n_layer=args.layers,
            n_fold=16,
        ).to(device=device)
    elif args.model == "wrt":
        import models.benchmark.wrt as md
        model = md.Net(
            spectra_size = (2400, 2 * args.cosets),
            patch_size = (16, 2 * args.cosets),
            num_bands = 16,
            dim = args.d_model,
            depth = args.layers + 1,
            heads = 4,
            mlp_dim = 4 * args.d_model,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device=device)
    else:
        raise ValueError(f"model {args.model} not supported, choose from 'spectrum_sensor' or 'wrt'")

    print(model.name())
    if args.params is not None:
        model_state_dict = torch.load(args.params)
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except RuntimeError:
            pass

    loss_fn = md.Loss()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    md.test(model, loss_fn, test_dl, snr=10)

    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Memory usage: {(total_mem - free_mem) / 1024 / 1024} MB", flush=True)

    md.timeit(model, test_dl, runs=20)

if __name__ == "__main__":
    main()