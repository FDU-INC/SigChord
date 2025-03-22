import h5py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse

import sigproc.cs.multicoset as M
import utils

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pkt_cls", help="pkt_cls or tprime")
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

    n_coset = args.cosets
    indices = range(2*n_coset)
    if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
        indices_c = M.DEFAULT_OFFSETS_SORTED_IDS[M.DEFAULT_BEST_OFFSETS[n_coset]]
        indices_c = 2 * np.array(indices_c)
        indices = np.zeros(2 * len(indices_c), dtype=np.uint8)
        indices[::2] = indices_c
        indices[1::2] = indices_c + 1

    data_test = f_test["waveforms"][-test_len:, :, indices]
    print("data_test shape", data_test.shape)
    data_test = torch.from_numpy(data_test).half()
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    bands_test = f_test["bands"][-test_len:]
    bands_test = torch.from_numpy(bands_test).long()

    test_ds = TensorDataset(data_test, bands_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return test_dl


def main():
    print(f"pid: {os.getpid()}", flush=True)
    args = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(23)

    print("===> loading model")
    if args.model == "pkt_cls":
        import models.pkt_classifier as md
        model = md.Net(
            n_coset=args.cosets,
            n_class=4,
            d_model=args.d_model,
            n_layer=args.layers,
        ).cuda()
    elif args.model == "tprime":
        import models.benchmark.tprime as md
        model = md.Net(
            n_coset=args.cosets,
            d_model=64 // 1,
            seq_len=75,
            nhead=8,
            nlayers=args.layers,
            n_class=4,
        ).cuda()
    else:
        raise ValueError(f"unknown model name: {args.model}, should be pkt_cls or tprime")
    print(f"model name: {model.name()}", flush=True)

    print("===> loading data")

    if args.params is not None:
        assert isinstance(args.params, str)
        model_state_dict = torch.load(args.params)
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except RuntimeError:
            pass

    test_dl = load_data(args)
    loss_fn = md.Loss()

    model = torch.compile(model)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    md.test(model, loss_fn, test_dl, snr=10)

    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Memory usage: {(total_mem - free_mem) / 1024 / 1024} MB", flush=True)

    md.timeit(model, test_dl, runs=20)


if __name__ == "__main__":
    main()
