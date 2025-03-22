"""
This model classifies headers of non-HT, HT Wi-Fi and DVB-S2.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.utils.benchmark as benchmark

import sigproc.cs.multicoset as M
import utils

class Classifier(nn.Module):
    MAX_TOKEN_LEN = 2400
    def __init__(self, seq_len: int, d_model: int, n_layer: int, n_fold: int, n_class: int):
        super().__init__()
        assert seq_len % n_fold == 0
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_fold = n_fold
        self.n_layer = n_layer

        self.slot_embedding = nn.Sequential(
            nn.Linear(2 * n_fold, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.position_embedding = nn.Embedding(seq_len // n_fold + 1, d_model)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=3 * d_model,
                activation="gelu",
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=n_layer,
        )

        self.decoders = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=3 * d_model,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=1,
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_class),
        )

    def forward(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = x.view(B, -1, 2 * self.n_fold)
        x = self.slot_embedding(x) + self.position_embedding.weight[:x.size(1)]
        x = self.encoders(x)
        x = self.decoders(self.query_token.repeat(B, 1, 1), x)
        x = self.fc(x.flatten(-2, -1))
        return x

class Net(nn.Module):
    """
    suppose the support is known (predict by the spectrum sensor)
    recover signals as in scripts.recovery_signals
    then classify the recovered signals each
    """
    N_SAMPLE = 2400
    SAMPLING_RATIO = 40
    def __init__(
        self,
        n_coset: int,
        n_class: int,
        d_model: int,
        n_layer: int,
    ):
        super().__init__()
        self.n_coset = n_coset
        self.offsets = M.DEFAULT_OFFSETS[:n_coset]
        self.n_class = n_class
        self.d_model = d_model

        # untrainable parameters for recovery
        if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
            self.offsets = M.DEFAULT_BEST_OFFSETS[n_coset]

        indices = M.DEFAULT_OFFSETS_SORTED_IDS[self.offsets]
        orders = np.argsort(np.argsort(indices))
        self.x_access_order = orders

        Y_coeff, A = self.get_rec_mats(self.offsets)
        self.A = nn.Parameter(A, requires_grad=False)
        self.Y_coeff = nn.Parameter(Y_coeff, requires_grad=False)

        self.classifer_base = Classifier(
            seq_len=self.N_SAMPLE,
            d_model=d_model,
            n_layer=n_layer,
            n_fold=32,
            n_class=n_class,
        )
        self._name = f"pkt_cls_L{n_layer}_D{d_model}_C{n_coset}_F32"

    def name(self) -> str:
        return self._name

    def get_rec_mats(self, offsets: list[int]) -> torch.Tensor:
        """
        prepare the matrices needed for recovery
        """

        A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, self.SAMPLING_RATIO, 1))
        A = np.matrix(np.exp(2j * np.pi * A_aux / self.SAMPLING_RATIO))

        Y_aux = np.array(offsets).reshape(-1, 1) @ np.arange(self.N_SAMPLE).reshape(1, -1)
        Y_coeff = np.exp(-2j * np.pi / (self.SAMPLING_RATIO * self.N_SAMPLE) * Y_aux)
        return torch.tensor(Y_coeff, dtype=torch.complex64), torch.tensor(A, dtype=torch.complex64)

    def recover_with_support(self, x: Tensor, support: Tensor) -> Tensor:
        """
        `support`, one-hot tensor of size (batch_size, n_band)
        """
        x = x[..., 0::2] + 1j * x[..., 1::2]
        x = x.transpose(-2, -1)
        # y = self.Y_coeff * torch.fft.fft(x, dim=-1)
        # ugly hack, see the comments in __init__
        y = self.Y_coeff * torch.fft.fft(x[:, self.x_access_order, :], dim=-1)

        n_occupied_each = torch.sum(support, dim=-1)
        xs_list = [None] * n_occupied_each.shape[0]
        unique_n_occupied = torch.unique(n_occupied_each)

        for n_occupied in unique_n_occupied:
            mask = n_occupied_each == n_occupied
            support_batch = support[mask]
            sup = torch.nonzero(support_batch)
            As = self.A[:, sup[:, 1]].view(self.n_coset, support_batch.shape[0], n_occupied)
            As = As.transpose(0, 1)
            xs = torch.linalg.lstsq(As, y[mask])[0]

            # keep the order
            cnt = 0
            for i in range(mask.shape[0]):
                if mask[i]:
                    xs_list[i] = xs[cnt]
                    cnt += 1

        Xs = torch.cat(xs_list, dim=0)

        return Xs

    def forward(self, x: Tensor, bands: Tensor, timeit_runs: int = 0) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, 2400, 2 * n_coset]
        Returns:
            output classifier label
        """
        x = x[:, ::1, :].float()
        support = (bands.cuda() > -1).long()
        x = torch.fft.ifft(self.recover_with_support(x, support))

        x = torch.stack([x.real, x.imag], dim=-1)
        x /= torch.std(x, [1, 2], keepdim=True)

        if timeit_runs > 0:
            timer = benchmark.Timer(
                stmt="self.classifer_base(x)",
                globals={"self": self, "x": x},
            )
            results = [timer.timeit(1).mean for _ in range(timeit_runs)]
            print(f"{timeit_runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")
        res = self.classifer_base(x)
        return res


class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, cls_pred: Tensor, cls_gt: Tensor) -> Tensor:
        cls_loss = self.cross_entropy_loss(cls_pred, cls_gt)
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
    np.save(f"./results/conf_m_pkt_cls_lse_C{model.n_coset}.npy", conf_matrix)
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
    n_coset = 4
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super(Wrapper, self).__init__()
            self.model = Net(
                n_coset=n_coset,
                n_class=4,
                d_model=128,
                n_layer=2,
            )

        def forward(self, x: Tensor):
            bands = torch.randint(-1, 4, (x.size(0), 16)).to(x.device)
            return self.model(x, bands)

    model = Wrapper()
    import torchinfo
    torchinfo.summary(model, input_size=(3, 2400, 2*n_coset))


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    params_test()