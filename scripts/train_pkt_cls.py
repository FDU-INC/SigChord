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
    parser.add_argument("--model", default="pkt_cls", type=str, help="pkt_cls or tprime")
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--epoch", type=int, default=300, help="default 300")
    parser.add_argument("--batch_size", type=int, default=128, help="default 128")
    parser.add_argument("--layers", type=int, default=2, help="default 2")
    parser.add_argument("--d_model", type=int, default=128, help="default 128")
    parser.add_argument("--cosets", type=int, help="number of sampling cosets")
    parser.add_argument("--train", type=str, help="train dataset prename")
    parser.add_argument("--test", type=str, help="test dataset prename")
    parser.add_argument("--params", type=str, help="path of saved model parameters")
    args = parser.parse_args()
    print("args:")
    print(args)
    return args


def save_checkpoint(model, epoch, remove_old=False, step: int = None):
    model_folder = "params/"
    model_out_path = model_folder + f"{model.name()}_epoch_{epoch}.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if remove_old:
        for f in os.listdir(model_folder):
            model_old_path = f"{model.name()}_epoch_{epoch - step}.pth"
            if (epoch - step) % 100 == 0:
                # ignore the hundredth epoch
                continue
            if f == model_old_path:
                os.remove(os.path.join(model_folder, f))
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def load_data(args: argparse.Namespace):
    f_train = h5py.File(str(args.train), "r")
    f_test = h5py.File(str(args.test), "r")
    train_len = 300000 - 5000
    test_len = 5000

    n_coset = args.cosets
    indices = range(2*n_coset)
    if 2 * n_coset == f_train["waveforms"].shape[-1]:
        indices = range(2 * n_coset)
    if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
        indices_c = M.DEFAULT_OFFSETS_SORTED_IDS[M.DEFAULT_BEST_OFFSETS[n_coset]]
        indices_c = 2 * np.array(indices_c)
        indices = np.zeros(2 * len(indices_c), dtype=np.uint8)
        indices[::2] = indices_c
        indices[1::2] = indices_c + 1

    data_train = f_train["waveforms"][:train_len, :, indices]
    print("data_train shape", data_train.shape)
    data_train = torch.from_numpy(data_train).half()
    data_train = (data_train / torch.std(data_train, [1, 2], keepdim=True))
    bands_train = f_train["bands"][:train_len]
    bands_train = torch.from_numpy(bands_train).long()

    data_test = f_test["waveforms"][-test_len:, :, indices]
    print("data_test shape", data_test.shape)
    data_test = torch.from_numpy(data_test).half()
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    bands_test = f_test["bands"][-test_len:]
    bands_test = torch.from_numpy(bands_test).long()

    train_ds = TensorDataset(data_train, bands_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)

    test_ds = TensorDataset(data_test, bands_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return train_dl, test_dl


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
    print(f"model name: {model.name()}", flush=True)

    print("===> loading data")
    train_dl, test_dl = load_data(args)

    loss_fn = md.Loss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = utils.ScheduledOptim(optimizer, args.d_model, 1000)

    start_epoch = 1
    if args.params is not None:
        assert isinstance(args.params, str)
        model_state_dict = torch.load(args.params)
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except RuntimeError:
            pass
        start_epoch = int(args.params.split("_")[-1].split(".")[0]) + 1
        scheduler.n_current_steps = len(train_dl) * start_epoch
        print("fine-tuning model")

    print("===> train process", flush=True)
    for i in range(start_epoch, args.epoch + start_epoch):
        print(f".............training epoch {i}", flush=True)
        md.train(model, loss_fn, train_dl, optimizer, scheduler)
        md.test(model, loss_fn, test_dl, snr=10)
        if i % 20 == 0:
            save_checkpoint(model, i, remove_old=True, step=20)
        print()


if __name__ == "__main__":
    main()
