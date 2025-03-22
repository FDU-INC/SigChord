"""
fuse features output from pkt classifier encoders and samples
followed by a query decoder
"""

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import torch.utils.benchmark as benchmark
import numpy as np

from models.pkt_classifier import Net as PktClassifier
from lse_solver import LSESolver
import sigproc.cs.multicoset as M
import utils

TYPE_ID = 3
N_BIT = 48

class SlotEmbedder(nn.Module):
    def __init__(self, in_feature: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, 2 * d_model),
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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Net(nn.Module):
    """
    input size (batch_size, 2400, 2 * n_coset)
    """

    N_BIT = N_BIT
    MAX_TOKEN_LEN = 1500
    SAMPLING_RATIO = 40
    N_SAMPLE = 2400

    def __init__(
        self,
        n_band: int,
        d_model: int,
        n_layer: int,
        n_coset: int,
        n_fold: int,
        pkt_classifier: PktClassifier,
    ) -> None:
        super(Net, self).__init__()
        self.n_band = n_band
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_coset = n_coset
        self.n_fold = n_fold
        assert 2400 % n_fold == 0
        assert n_coset > 0

        self.pkt_classifier = pkt_classifier.requires_grad_(False)

        offsets = M.DEFAULT_OFFSETS[:n_coset]
        if M.DEFAULT_BEST_OFFSETS.get(n_coset) is not None:
            offsets = M.DEFAULT_BEST_OFFSETS[n_coset]
        self.lse_solver = LSESolver(n_coset, offsets)

        self.pkt_cls_feat_converter = nn.Sequential(
            nn.Linear(pkt_classifier.d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pkt_cls_rate = nn.Parameter(torch.Tensor([1.0]))

        self.single_slot_embedding = SlotEmbedder(2 * n_fold, d_model)
        self.position_embedding = nn.Embedding(self.MAX_TOKEN_LEN, d_model)
        self.embedding_ln = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.1)

        # 2 additional tokens for known 0s and 1s
        self.query_embedding = nn.Embedding(self.N_BIT + 2, d_model)
        self.__init_query_tokens()
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=n_layer,
        )

        self.classifier = nn.Linear(d_model, 2)

        self._name = f"HT_L{n_layer}_D{d_model}_C{n_coset}_F{n_fold}"

    def __pkt_cls_embedding(self, x: Tensor) -> Tensor:
        pkt_cls_base = self.pkt_classifier.classifer_base
        n_fold = pkt_cls_base.n_fold
        B = x.size(0)
        x = x.view(B, -1, 2 * n_fold)
        x = (
            pkt_cls_base.slot_embedding(x)
            + pkt_cls_base.position_embedding.weight[: x.size(1)]
        )
        x = pkt_cls_base.encoders(x)
        x = self.pkt_cls_feat_converter(x)
        return x

    def __pre_input_embedding(self, x: Tensor, bands: Tensor) -> Tensor:
        # recover signals
        support = (bands > -1).to(x.device).long()
        x = torch.fft.ifft(self.lse_solver.recover_with_support(x, support))

        # signals on active sub-bands are recovered, but we only need headers
        sup_id = torch.argwhere(support.flatten()).flatten()
        target_from_sup_id = torch.argwhere(
            bands.flatten()[sup_id] == TYPE_ID
        ).flatten()
        # x = x[target_from_sup_id][:1378]
        x = x[target_from_sup_id]

        x = torch.stack([x.real, x.imag], dim=-1)
        x /= torch.std(x, [1, 2], keepdim=True)

        # embedding from the packet classifier
        pkt_cls_feat = self.__pkt_cls_embedding(x)
        return x, pkt_cls_feat

    def __init_query_tokens(self) -> Tensor:
        query = torch.arange(self.N_BIT, dtype=torch.long)
        # reservation bit
        # query[4] = BIT_0
        # padding bits
        # query[18:] = BIT_0
        self.query_tokens = query.unsqueeze(0)

    def decode(self, x: Tensor, pkt_cls_feat: Tensor) -> Tensor:
        # single band signal embedding
        x = self.single_slot_embedding(x.view(x.size(0), -1, 2 * self.n_fold))

        # add position embedding
        x = x + self.position_embedding.weight[: x.size(1)]
        x[:, : pkt_cls_feat.size(1)] += self.pkt_cls_rate * pkt_cls_feat

        x = self.embedding_ln(x)
        x = self.embedding_dropout(x)
        query = self.query_tokens.expand(x.size(0), -1).to(x.device)
        query = self.query_embedding(query)
        x = self.decoder(tgt=query, memory=x)

        x = self.classifier(x[:, : self.N_BIT, :])
        return x

    def forward(self, x: Tensor, bands: Tensor, timeit_runs: int = 0) -> Tensor:
        """
        # parameters:

        - `x`: of size (batch_size, 2400, 2 * n_cosets)
        - `bands` of size (batch_size, n_band), bands[i][j] represents the
        signal type of the j-th sub-band. -1 means the sub-band is not occupied.
        0 means the sub-band does not contain any header.

        # return:

        - Tensor of size (n_sigs_in_batch, N_BITS, 2)
        """
        x = x.float()
        x, pkt_cls_feat = self.__pre_input_embedding(x, bands)
        if timeit_runs > 0:
            timer = benchmark.Timer(
                stmt="self.decode(x, pkt_cls_feat)",
                globals={"self": self, "x": x, "pkt_cls_feat": pkt_cls_feat},
            )
            results = [timer.timeit(1).mean for _ in range(timeit_runs)]
            print(f"{timeit_runs} runs, batch size {x.size(0)}, mean {np.mean(results) * 1000} ms, std {np.std(results) * 1000} ms")
        return self.decode(x, pkt_cls_feat)

    def name(self) -> str:
        return self._name


class Loss(nn.Module):
    N_BIT = N_BIT
    N_LEN_BIT = 12

    N_SIG_LEN_BITS = 12
    N_PKT_LEN_BITS = 16

    def __init__(self) -> None:
        super(Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

        self.valid_bits_ids = np.concatenate([
            np.arange(5, 17), # L-SIG length field
            np.arange(24, 48), # all HT-SIG1 fields
        ])
        self.valid_bits_ids = list(self.valid_bits_ids)

        self.orders_2 = torch.arange(self.N_PKT_LEN_BITS)
        self.length_weights = torch.pow(1.2, self.orders_2).float()
        self.orders_2 = torch.pow(2, self.orders_2).long()

        self.sig_len_diff_stat = torch.Tensor()
        self.pkt_len_diff_stat = torch.Tensor()

    def forward(self, bit_pred: Tensor, bit_gt: Tensor) -> Tensor:
        assert bit_pred.size(0) == bit_gt.size(0)

        # (n_sigs_in_batch * N_BITS)
        # bit_pred = bit_pred[:, self.valid_bits_ids, :].flatten(start_dim=0, end_dim=1)
        bit_pred = bit_pred[:, :, :].flatten(start_dim=0, end_dim=1)

        # bit_gt = bit_gt[..., self.valid_bits_ids].flatten(start_dim=0, end_dim=1)
        bit_gt = bit_gt[..., :N_BIT].flatten(start_dim=0, end_dim=1)

        bit_loss = self.cross_entropy_loss(bit_pred, bit_gt)
        bit_loss = bit_loss.view(-1, self.N_BIT)
        return bit_loss.mean()

    def acc(self, bit_pred: Tensor, bit_gt: Tensor):
        bit_gt = bit_gt[:, self.valid_bits_ids]
        bit_pred = bit_pred.argmax(dim=-1)[..., self.valid_bits_ids]
        shot = bit_gt == bit_pred
        shot = shot.sum().item()
        total = bit_gt.size(0) * bit_gt.size(1)

        return total, shot

    def stat(self, pred: Tensor, gt: Tensor, verbose=False):
        pred = pred.argmax(dim=-1)

        # sig length bits
        orders = self.orders_2.unsqueeze(0).expand(gt.size(0), -1).to(gt.device)
        sig_len_pred = (pred[:, 5:17] * orders[:, :12]).sum(dim=-1)
        sig_len_gt = (gt[:, 5:17] * orders[:, :12]).sum(dim=-1)

        sig_len_diff_sum = (sig_len_pred - sig_len_gt).abs().sum().item()
        sig_len_gt_sum = sig_len_gt.sum().item()

        # packet length bits
        pkt_len_pred = (pred[:, 32:] * orders).sum(dim=-1)
        pkt_len_gt = (gt[:, 32:] * orders).sum(dim=-1)

        pkt_len_diff_sum = (pkt_len_pred - pkt_len_gt).abs().sum().item()
        pkt_len_gt_sum = pkt_len_gt.sum().item()

        # modulation and coding scheme
        mcs_pred = pred[:, 24:31]
        mcs_gt = gt[:, 24:31]
        mcs_shot = (mcs_pred == mcs_gt).all(dim=-1).sum().item()
        mcs_total = gt.size(0)
        if verbose:
            each_bit = (pred == gt[:, :self.N_BIT]).sum(dim=0)
            self.sig_len_diff_stat = torch.cat([self.sig_len_diff_stat, (sig_len_pred - sig_len_gt).cpu()])
            self.pkt_len_diff_stat = torch.cat([self.pkt_len_diff_stat, (pkt_len_pred - pkt_len_gt).cpu()])
            return sig_len_diff_sum, sig_len_gt_sum, pkt_len_diff_sum, pkt_len_gt_sum, mcs_shot, mcs_total, each_bit
        else:
            return sig_len_diff_sum, sig_len_gt_sum, pkt_len_diff_sum, pkt_len_gt_sum, mcs_shot, mcs_total

    def clear_stat(self):
        self.sig_len_diff_stat = torch.Tensor()
        self.pkt_len_diff_stat = torch.Tensor()

H = 6.626e-34
def train(
    model: Net, loss_fn: Loss, train_dl: DataLoader, optimizer: Optimizer, scheduler
):
    model.train()
    train_loss = 0
    train_shot = 0
    train_total = 0
    print("learning rate:", optimizer.param_groups[0]["lr"])
    for n, (x, bands, bit_gt) in enumerate(train_dl):
        x = x.cuda().float()
        bands = bands.cuda()
        x = utils.add_noise(x)
        band_ids = torch.argwhere(bands == TYPE_ID)
        if len(band_ids) == 0:
            continue
        bit_gt = bit_gt.cuda()[band_ids[:, 0], band_ids[:, 1]]

        bit_pred = model(x, bands)
        loss = loss_fn(bit_pred, bit_gt)
        train_loss = train_loss * n / (n + 1) + loss.item() / (n + 1)
        stats = loss_fn.acc(bit_pred, bit_gt)

        total, shot = stats
        train_shot += shot
        train_total += total

        scheduler.zero_grad()
        loss.backward()
        scheduler.step()

    train_acc = round(train_shot / (train_total + H), 4)
    print(".......train loss {}, acc {}".format(train_loss, train_acc), flush=True)


def test(model: Net, loss_fn: Loss, test_dl: DataLoader, snr: float | None = 10, verbose=False):
    model.eval()
    test_loss = 0
    test_bit_shot = 0
    test_bit_total = 0
    sig_len_diff = 0
    sig_len_gt = 0
    pkt_len_diff = 0
    pkt_len_gt = 0
    mcs_shot = 0
    sig_total = 0
    each_bit_total = torch.zeros(N_BIT)
    with torch.no_grad():
        for n, (x, bands, bit_gt) in enumerate(test_dl):
            x = x.cuda().float()
            x = utils.add_noise(x, snr)
            bands = bands.cuda()
            band_ids = torch.argwhere(bands == TYPE_ID)
            if len(band_ids) == 0:
                continue
            bit_gt = bit_gt.cuda()[band_ids[:, 0], band_ids[:, 1]]

            bit_pred = model(x, bands)
            loss = loss_fn(bit_pred, bit_gt)
            test_loss = test_loss * n / (n + 1) + loss.item() / (n + 1)
            stats = loss_fn.acc(bit_pred, bit_gt)

            total, shot = stats
            test_bit_shot += shot
            test_bit_total += total

            if verbose:
                slen_d, slen_tot, plen_d, plen_tot, mcs, n_sig, each_bit = loss_fn.stat(bit_pred, bit_gt, verbose)
                each_bit_total += each_bit.to(each_bit_total.device)
            else:
                slen_d, slen_tot, plen_d, plen_tot, mcs, n_sig = loss_fn.stat(bit_pred, bit_gt, verbose)

            sig_len_diff += slen_d
            sig_len_gt += slen_tot
            pkt_len_diff += plen_d
            pkt_len_gt += plen_tot
            mcs_shot += mcs
            sig_total += n_sig

    h = 6.626e-34
    print("MCS acc:", mcs_shot / (sig_total + h))
    print("signal length diff:", sig_len_diff / (sig_len_gt + h))
    print("packet length diff:", pkt_len_diff / (pkt_len_gt + h))
    if verbose:
        print("each bit stat:", each_bit_total / (sig_total + h))
        print("mean bit accuracy: ", torch.mean(each_bit_total / (sig_total + h)))
        print("95th percentile packet length diff:", torch.quantile(torch.abs(loss_fn.pkt_len_diff_stat), 0.95))

    test_acc = round(test_bit_shot / (test_bit_total + H), 4)
    print(
        ".......test loss {}, acc {}".format(test_loss, test_acc), flush=True
    )

def timeit(model: Net, test_dl: DataLoader, runs=20):
    with torch.no_grad():
        for (x, bands, _) in test_dl:
            x = x.cuda().float()
            bands = bands.cuda()
            model(x, bands, timeit_runs=runs)
            return

def params_test():
    n_coset = 8
    pkt_classifier = PktClassifier(
        n_class=4,
        n_layer=2,
        d_model=128,
        n_coset=n_coset,
    )

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super(Wrapper, self).__init__()
            self.model = Net(
                n_band=16,
                d_model=384,
                n_layer=3,
                n_coset=n_coset,
                n_fold=32,
                pkt_classifier=pkt_classifier,
            )

        def forward(self, x: Tensor):
            bands = torch.randint(-1, 4, (x.size(0), self.model.n_band)).to(x.device)
            return self.model(x, bands)

    model = Wrapper()
    import torchinfo

    torchinfo.summary(model, input_size=(3, 2400, 2 * n_coset))


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params_test()