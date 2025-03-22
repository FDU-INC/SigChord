import math
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.utils.benchmark as benchmark

import sigproc.cs.multicoset as M
from lse_solver import LSESolver
import utils

class TprimeBase(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        seq_len: int = 64,
        nhead: int = 8,
        nlayers: int = 2,
        dropout: float = 0.1,
        classes: int = 4,
        use_pos: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.norm = nn.LayerNorm(d_model)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define the encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 16 * d_model, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model * seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

        self._name = f"tprime_L{nlayers}_D{d_model}"
        self.n_class = classes

    def name(self) -> str:
        return self._name

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, 96000, 2]
        Returns:
            output classifier label
        """
        B = src.size(0)
        src = src.reshape(B, -1, self.d_model)

        # src = src * math.sqrt(self.d_model)
        src = self.norm(src.float())
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.logSoftmax(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class Net(nn.Module):
    SAMPLING_RATIO = 40
    def __init__(
        self,
        n_coset: int,
        d_model: int = 512,
        seq_len: int = 64,
        nhead: int = 8,
        nlayers: int = 2,
        dropout: float = 0.1,
        n_class: int = 4,
        use_pos: bool = False,
    ):
        super().__init__()
        self.n_coset = n_coset
        self.n_class = n_class

        offsets = M.DEFAULT_OFFSETS[:n_coset]
        if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
            offsets = M.DEFAULT_BEST_OFFSETS[n_coset]
        self.lse_solver = LSESolver(n_coset, offsets)

        # input to tprime_base is (n_sigs_in_batch, 2400, 2)
        self.tprime_base = TprimeBase(
            d_model,
            seq_len,
            nhead,
            nlayers,
            dropout,
            n_class,
            use_pos,
        )
        self._name = f"tprime_L{nlayers}_C{n_coset}_D{d_model}"

    def name(self) -> str:
        return self._name

    def forward(self, x: Tensor, bands: Tensor, timeit_runs: int = 0) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, 2400, 2 * n_coset]
        Returns:
            output classifier label
        """
        x = x[:, ::1, :].float()
        support = (bands.cuda() > -1).long()
        x = torch.fft.ifft(self.lse_solver.recover_with_support(x, support))
        x = torch.stack([x.real, x.imag], dim=-1)
        x /= torch.std(x, [1, 2], keepdim=True)

        if timeit_runs > 0:
            timer = benchmark.Timer(
                stmt="self.tprime_base(x)",
                globals={"self": self, "x": x},
            )
            results = [timer.timeit(1).mean for _ in range(timeit_runs)]
            print(f"{timeit_runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")
        x = self.tprime_base(x)
        return x

class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, cls_pred: Tensor, cls_gt: Tensor) -> Tensor:
        cls_loss = self.nll_loss(cls_pred, cls_gt)
        return cls_loss

    def acc(self, cls_pred: Tensor, cls_gt: Tensor):
        cls_pred = cls_pred.argmax(dim=-1)
        shot = (cls_gt == cls_pred).sum().item()
        total = cls_gt.size(0)

        return total, shot

H = 6.626e-34
def train(model: Net, loss_fn: Loss, train_dl: DataLoader, optimizer: Optimizer, scheduler):
    model.train()
    train_loss = 0
    total = 0
    shot = 0
    print("learning rate:", optimizer.param_groups[0]["lr"])
    for n, (x, bands) in enumerate(train_dl):
        x = x.cuda().float()
        x = utils.add_noise(x)
        x /= torch.std(x, [1, 2], keepdim=True)
        bands = bands.cuda()

        band_ids = torch.argwhere(bands > -1)
        cls_gt = bands[band_ids[:, 0], band_ids[:, 1]]

        cls_pred = model(x, bands)

        loss = loss_fn(cls_pred, cls_gt)
        train_loss = train_loss * n / (n + 1) + loss.item() / (n + 1)

        total_, shot_ = loss_fn.acc(cls_pred, cls_gt)
        total += total_
        shot += shot_

        scheduler.zero_grad()
        loss.backward()
        scheduler.step()

    print(
        ".......train loss {}, acc {}".format(train_loss, round(shot / (total + H), 4)), flush=True
    )

def test(model: Net, loss_fn: Loss, test_dl: DataLoader, snr: float | None = 10):
    model.eval()
    test_loss = 0
    total = 0
    shot = 0
    conf_matrix = np.zeros((model.n_class, model.n_class))
    with torch.no_grad():
        for n, (x, bands) in enumerate(test_dl):
            x = x.cuda().float()
            x = utils.add_noise(x, snr)
            bands = bands.cuda()

            band_ids = torch.argwhere(bands > -1)
            cls_gt = bands[band_ids[:, 0], band_ids[:, 1]]

            cls_pred = model(x, bands)

            loss = loss_fn(cls_pred, cls_gt)
            test_loss = test_loss * n / (n + 1) + loss.item() / (n + 1)
            conf_m = confusion_matrix(cls_gt.cpu().numpy(), cls_pred.argmax(dim=-1).cpu().numpy(), labels=range(model.n_class))
            conf_matrix += conf_m

            total_, shot_ = loss_fn.acc(cls_pred, cls_gt)
            total += total_
            shot += shot_

    conf_matrix = np.array(conf_matrix, dtype=np.uint64)
    print(conf_matrix / (np.sum(conf_matrix, axis=-1, keepdims=True) + H))
    np.save(f"./results/conf_m_tprime_C{model.n_coset}.npy", conf_matrix)
    print(
        ".......test loss {}, acc {}".format(test_loss, round(shot / (total + H), 4)), flush=True
    )

def timeit(model: Net, test_dl: DataLoader, runs: int):
    with torch.no_grad():
        for (x, bands) in test_dl:
            x = x.cuda().float()
            bands = bands.cuda()
            model(x, bands, timeit_runs=runs)
            return

def params_test():
    n_coset = 8
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super(Wrapper, self).__init__()
            self.model = Net(
                n_coset=n_coset,
                d_model=64,
                seq_len=75,
                nhead=8,
                nlayers=3,
                n_class=4,
            )

        def forward(self, x: Tensor):
            bands = torch.randint(-1, 4, (x.size(0), 16)).to(x.device)
            return self.model(x, bands)

    model = Wrapper()
    import torchinfo
    torchinfo.summary(model, input_size=(3, 2400, 2*n_coset))


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    params_test()