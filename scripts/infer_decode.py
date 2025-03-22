import h5py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse
import copy

import models.pkt_classifier as pkt
import sigproc.cs.multicoset as M

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="dvbs2 or nonHT or HT")
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--batch_size", type=int, default=1024, help="default 1024")
    parser.add_argument("--layers", type=int, default=3, help="default 3")
    parser.add_argument("--d_model", type=int, default=384, help="default 384")
    parser.add_argument("--cosets", type=int, help="number of sampling cosets")
    parser.add_argument("--test", type=str, help="test dataset prename")
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
    bits_test = f_test["bits"][-test_len:]
    bits_test = torch.from_numpy(bits_test).long()

    test_ds = TensorDataset(data_test, bands_test, bits_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return test_dl


def main():
    print(f"pid: {os.getpid()}", flush=True)
    args = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(22)

    print("===> loading model")
    # please make sure the pkt_cls params match the model
    pkt_classifier_path = "./params/pkt_cls_L2_D128_C8_F32_epoch_300.pth"
    print(f"loading pkt_classifier from {pkt_classifier_path}", flush=True)
    pkt_classifier = pkt.Net(
        n_coset=8,
        d_model=128,
        n_layer=2,
        n_class=4,
    )
    pkt_classifier.load_state_dict(torch.load(pkt_classifier_path))

    if args.model == "dvbs2":
        import models.dvbs2_decoder as md
    elif args.model == "nonHT":
        import models.nonHT_decoder as md
    elif args.model == "HT":
        import models.HT_decoder as md
    else:
        raise ValueError(f"unknown model name: {args.model}, should be dvbs2 or nonHT or HT")

    model = md.Net(
        n_band=16,
        d_model=args.d_model,
        n_layer=args.layers,
        n_coset=args.cosets,
        n_fold=32,
        pkt_classifier=copy.deepcopy(pkt_classifier.cuda()),
    ).cuda()
    print(model.name())

    print("===> loading data", flush=True)
    test_dl = load_data(args)

    loss_fn = md.Loss()

    if args.params is not None:
        assert isinstance(args.params, str)
        model_state_dict = torch.load(args.params)
        try:
            filtered_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("pkt_class")}
            model.load_state_dict(filtered_state_dict, strict=False)
            # prevent overwriting pkt_classifier.
            model.pkt_classifier = pkt_classifier.requires_grad_(False)
        except RuntimeError as e:
            print(e)
            pass

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    md.test(model, loss_fn, test_dl, snr=10, verbose=True)

    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Memory usage: {(total_mem - free_mem) / 1024 / 1024} MB", flush=True)

    md.timeit(model, test_dl, runs=20)

if __name__ == "__main__":
    main()
