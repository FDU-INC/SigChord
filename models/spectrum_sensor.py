"""
Model structure, training and validation code for spectrum sensing.
"""
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.utils.benchmark as benchmark
import numpy as np

import utils

class Net(nn.Module):
    """
    input size (batch_size, 2400, 2 * n_coset)
    """

    MAX_TOKEN_LEN = 1024

    def __init__(
            self,
            n_coset: int,
            n_band: int,
            d_model: int,
            n_layer: int,
            n_fold: int
        ) -> None:
        super(Net, self).__init__()
        self.n_band = n_band
        self.n_coset = n_coset
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_fold = n_fold
        assert 2400 % n_fold == 0

        self.slot_embedding = nn.Sequential(
            nn.Linear(2 * n_fold * n_coset, 2 * d_model),
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

        self.sos_embedding = nn.Embedding(1, d_model)
        # one extra token for SOS
        self.position_embedding = nn.Embedding(self.MAX_TOKEN_LEN + 1, d_model)
        self.embedding_ln = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.1)

        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=4 * d_model,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=n_layer,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(d_model, n_band),
        )

        self._name = f"ss_L{self.n_layer}_D{self.d_model}_C{self.n_coset}_F{self.n_fold}"

    def input_embedding(self, x: Tensor) -> Tensor:
        """
        build embedding from multi-coset sub-Nyquist sampling inputs

        # parameters:
        - `x` of size (batch_size, 2400, 2 * n_cosets)
        - `band_ids` of size (n_sigs_in_batch, 2), band_ids[i][0] and
        band_ids[i][1] are the signal id in the batch and the occupied band id,
        respectively. In general, it is torch.argwhere(bands), where bands is
        the onehot band occupation of shape (batch_size, n_bands)
        """
        B = x.size(0)
        n_seq = x.size(1) // self.n_fold
        x = x.reshape(B, n_seq, -1)
        x = self.slot_embedding(x)

        position_ids = torch.arange(
            x.size(1), dtype=torch.long, device=x.device
        )
        x = x + self.position_embedding(position_ids)

        sos_tokens = torch.zeros(
            (x.size(0), 1), dtype=torch.long, device=x.device,
        )
        sos_embedding = self.sos_embedding(sos_tokens)
        x = torch.concat((sos_embedding, x), dim=1)
        x = self.embedding_ln(x)
        x = self.embedding_dropout(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        # parameters:

        - `x`: of size (batch_size, 2560, 2 * n_cosets)
        - `band_ids` of size (n_sigs_in_batch, 2), band_ids[i][0] and
        band_ids[i][1] are the signal id in the batch and the occupied band id,
        respectively. In general, it is torch.argwhere(bands), where bands is
        the onehot band occupation of shape (batch_size, n_bands)

        # return:

        - Tensor of size (n_sigs_in_batch, n_class)
        """
        # x = self.input_embedding(x.float())
        x = self.input_embedding(x)
        x = self.encoders(x, src_key_padding_mask=None)
        x = self.classifier(x[:, 0, :])
        return x

    def name(self) -> str:
        return self._name

class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
        # self.bce_loss = torchvision.ops.sigmoid_focal_loss
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        # return self.bce_loss(pred, gt.float(), alpha=0.25).sum(dim=-1).mean()
        return self.bce_loss(torch.sigmoid(pred), gt.float()).sum(dim=-1).mean()

    def acc(self, pred: Tensor, gt: Tensor, threshold: float = 0.5):
        pred = (torch.sigmoid(pred) > threshold).long()
        shot = torch.all(gt == pred, dim=-1).sum().item()
        total = gt.size(0)

        return total, shot

    def stat(self, pred: Tensor, gt: Tensor, threshold: float = 0.5):
        pred = (torch.sigmoid(pred) > threshold).long()
        fp = torch.sum((pred == 1) & (gt == 0)).item()
        fn = torch.sum((pred == 0) & (gt == 1)).item()
        positives = (gt == 1).sum().item()
        negatives = (gt == 0).sum().item()

        return fp, fn, positives, negatives

H = 6.626e-34
def train(model: Net, loss_fn: Loss, train_dl: DataLoader, optimizer: Optimizer, scheduler):
    model.train()
    train_loss = 0
    train_shot = 0
    train_total = 0
    print("learning rate:", optimizer.param_groups[0]["lr"])
    for n, (x, gt) in enumerate(train_dl):
        x = x.cuda().float()
        x = utils.add_noise(x)
        gt = gt.cuda()
        gt = (gt > -1).long()

        pred = model(x)
        loss = loss_fn(pred, gt)
        train_loss = train_loss * n / (n + 1) + loss.item() / (n + 1)
        stats = loss_fn.acc(pred, gt)

        total, shot = stats
        train_shot += shot
        train_total += total

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step(train_loss)

    train_acc = round(train_shot / (train_total + H), 4)
    print(
        ".......train loss {}, acc {}".format(train_loss, train_acc), flush=True
    )

def test(model: Net, loss_fn: Loss, test_dl: DataLoader, snr: float | None = 10):
    model = model.eval()
    test_loss = 0
    test_shot = 0
    test_total = 0
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    with torch.no_grad():
        for n, (x, gt) in enumerate(test_dl):
            x = x.cuda().float()
            x = utils.add_noise(x, snr)
            gt = gt.cuda()
            gt = (gt > -1).long()
            pred = model(x)
            loss = loss_fn(pred, gt)
            test_loss = test_loss * n / (n + 1) + loss.item() / (n + 1)

            stats = loss_fn.acc(pred, gt)
            total, shot = stats
            test_shot += shot
            test_total += total

            fp, fn, pos, neg = loss_fn.stat(pred, gt)
            false_positive += fp
            false_negative += fn
            positive += pos
            negative += neg

    test_acc = round(test_shot / (test_total + H), 4)
    print(
        ".......test loss {}, acc {}".format(test_loss, test_acc), flush=True
    )
    print("false alarm:", false_positive / (negative + H))
    print("miss detection:", false_negative / (positive + H))

def timeit(model: Net, test_dl: DataLoader, runs: int):
    with torch.no_grad():
        for (x, _) in test_dl:
            x = x.cuda().float()
            timer = benchmark.Timer(
                stmt="model(x)",
                globals={"model": model, "x": x},
            )
            results = [timer.timeit(1).mean for _ in range(runs)]
            print(f"{runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")
            return

def params_test():
    n_coset = 8
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super(Wrapper, self).__init__()
            self.model = Net(
                n_band=16,
                d_model=128,
                n_layer=2,
                n_coset=n_coset,
                n_fold=16,
            )

        def forward(self, x: Tensor):
            return self.model(x)

    model = Wrapper()
    import torchinfo
    torchinfo.summary(model, input_size=(3, 2400, 2*n_coset))
    model = model.cuda()
    model.eval()
    input = torch.randn(1024, 2400, 2 * n_coset).cuda()
    breakpoint()
    import torch.utils.benchmark as benchmark
    with torch.no_grad():
        timer = benchmark.Timer(
            stmt="model(input)",
            globals={"model": model, "input": input},
        )
        print(timer.timeit(10))

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params_test()