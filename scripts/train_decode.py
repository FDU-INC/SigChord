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
import utils

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="dvbs2 or nonHT or HT")
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--epoch", type=int, default=600, help="default 600")
    parser.add_argument("--batch_size", type=int, default=128, help="default 128")
    parser.add_argument("--layers", type=int, default=3, help="default 3")
    parser.add_argument("--d_model", type=int, default=384, help="default 384")
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
            if (epoch - step) % 300 == 0:
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
    bits_train = f_train["bits"][:train_len]
    bits_train = torch.from_numpy(bits_train).long()

    data_test = f_test["waveforms"][-test_len:, :, indices]
    print("data_test shape", data_test.shape)
    data_test = torch.from_numpy(data_test).half()
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    bands_test = f_test["bands"][-test_len:]
    bands_test = torch.from_numpy(bands_test).long()
    bits_test = f_test["bits"][-test_len:]
    bits_test = torch.from_numpy(bits_test).long()

    train_ds = TensorDataset(data_train, bands_train, bits_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)

    test_ds = TensorDataset(data_test, bands_test, bits_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return train_dl, test_dl


def main():
    print(f"pid: {os.getpid()}", flush=True)
    args = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(22)

    print("===> loading model")
    # please make sure the pkt_cls params match the model
    pkt_classifier_path = "./params/pkt_cls_L2_D128_C6_F32_epoch_300.pth"
    print(f"loading pkt_classifier from {pkt_classifier_path}", flush=True)
    pkt_classifier = pkt.Net(
        n_coset=6,
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
    train_dl, test_dl = load_data(args)

    loss_fn = md.Loss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = utils.ScheduledOptim(optimizer, args.d_model, 10000, decay=-0.5, init_lr=0.03)

    start_epoch = 1
    if args.params is not None:
        assert isinstance(args.params, str)
        model_state_dict = torch.load(args.params)
        try:
            filtered_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("pkt_class")}
            model.load_state_dict(filtered_state_dict, strict=False)
            # prevent overwriting pkt_classifier.
            model.pkt_classifier = pkt_classifier.requires_grad_(False)
            start_epoch = int(args.params.split("_")[-1].split(".")[0]) + 1
            scheduler.n_current_steps = (start_epoch) * len(train_dl)
        except RuntimeError as e:
            print(e)
            pass
        print("fine-tuning model")

    print("===> train process", flush=True)
    for i in range(start_epoch, args.epoch + start_epoch):
        print(f".............training epoch {i}", flush=True)
        md.train(model, loss_fn, train_dl, optimizer, scheduler)
        md.test(model, loss_fn, test_dl, snr=10, verbose=True)
        if i % 20 == 0:
            save_checkpoint(model, i, remove_old=(i != start_epoch + 20), step=20)
        print()


if __name__ == "__main__":
    main()
