import h5py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import argparse

import sigproc.cs.multicoset as M

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spectrum_sensor", help="spectrum_sensor or wrt")
    parser.add_argument("--gpu", type=int, default=0, help="dafault 0")
    parser.add_argument("--epoch", type=int, default=100, help="default 100")
    parser.add_argument("--batch_size", type=int, default=512, help="default 512")
    parser.add_argument("--layers", type=int, default=2, help="default 2")
    parser.add_argument("--d_model", type=int, default=128, help="default 128")
    parser.add_argument("--cosets", type=int, help="number of sampling channels")
    parser.add_argument("--train", type=str, help="train dataset")
    parser.add_argument("--test", type=str, help="test dataset")
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

    indices = range(2*args.cosets)
    if M.DEFAULT_BEST_OFFSETS.get(args.cosets) is not None:
        indices_c = M.DEFAULT_OFFSETS_SORTED_IDS[M.DEFAULT_BEST_OFFSETS[args.cosets]]
        indices_c = 2 * np.array(indices_c)
        indices = np.zeros(2 * len(indices_c), dtype=np.uint8)
        indices[::2] = indices_c
        indices[1::2] = indices_c + 1

    data_train = f_train["waveforms"][:train_len, :, indices]
    print("data_train shape", data_train.shape)
    data_train = torch.from_numpy(data_train).half()
    data_train = (data_train / torch.std(data_train, [1, 2], keepdim=True))
    label_train = f_train["bands"][:train_len]
    label_train = torch.from_numpy(label_train).long()

    data_test = f_test["waveforms"][-test_len:, :, indices]
    print("data_test shape", data_test.shape)
    data_test = torch.from_numpy(data_test).half()
    data_test = (data_test / torch.std(data_test, [1, 2], keepdim=True))
    label_test = f_test["bands"][-test_len:]
    label_test = torch.from_numpy(label_test).long()

    train_ds = TensorDataset(data_train, label_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)

    test_ds = TensorDataset(data_test, label_test)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, prefetch_factor=4)
    return train_dl, test_dl

def main():
    print(f"pid: {os.getpid()}", flush=True)
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    torch.random.manual_seed(22)

    print("===> loading data")
    train_dl, test_dl = load_data(args)

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
            depth = args.layers,
            heads = 4,
            mlp_dim = 4 * args.d_model,
            dropout = 0.1,
            emb_dropout = 0.1
        ).to(device=device)
    else:
        raise ValueError(f"model {args.model} not supported, choose from 'spectrum_sensor' or 'wrt'")

    print(model.name())

    loss_fn = md.Loss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=False, min_lr=1e-6
    )

    if args.params is not None:
        assert isinstance(args.params, str)
        model_state_dict = torch.load(args.params)
        try:
            model.load_state_dict(model_state_dict, strict=False)
            optimizer.param_groups[0]["lr"] = 1e-6
        except RuntimeError:
            pass
        print("fine-tuning model")

    print("===> train process", flush=True)
    for i in range(1, args.epoch + 1):
        print(f".............training epoch {i}", flush=True)
        md.train(model, loss_fn, train_dl, optimizer, scheduler)
        md.test(model, loss_fn, test_dl, snr=10)

        if i % 10 == 0:
            save_checkpoint(model, i, remove_old=True, step=10)

        print()

if __name__ == "__main__":
    main()
