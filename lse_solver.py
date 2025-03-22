import torch
from torch import (nn, Tensor)
import numpy as np

import sigproc.cs.multicoset as M

class LSESolver(nn.Module):
    N_SAMPLE = 2400
    SAMPLING_RATIO = 40
    def __init__(self, n_coset: int, offsets: list[int] | None) -> None:
        super().__init__()
        self.n_coset = n_coset
        if offsets is None:
            self.offsets = M.DEFAULT_OFFSETS[:self.n_coset]
            if M.DEFAULT_BEST_OFFSETS.get(self.n_coset) is not None:
                self.offsets = M.DEFAULT_BEST_OFFSETS[self.n_coset]
        else:
            self.offsets = offsets
        self.Y_coeff, self.A = self.__get_rec_mats(self.offsets)

        # ugly hack
        # h5py access does not keep the index order
        # say, if we have [1, 2, 3, 4, 5, 6], then access it with [0, 4, 1, 5]
        # we get [1 2 5 6] instead of [1 5 2 6]
        # spectrum sensor is currently trained with the ascending order
        # but the recovery should use the real order
        # so we need to reorder the input of recovery
        # The recovery performance drop near Landau rate is due to missing this step
        indices = M.DEFAULT_OFFSETS_SORTED_IDS[self.offsets]
        orders = np.argsort(np.argsort(indices))
        self.x_access_order = orders

        Y_coeff, A = self.__get_rec_mats(self.offsets)
        self.A = nn.Parameter(A, requires_grad=False)
        self.Y_coeff = nn.Parameter(Y_coeff, requires_grad=False)

    def __get_rec_mats(self, offsets: list[int]) -> torch.Tensor:
        """
        prepare the matrices needed for recovery
        """

        A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, self.SAMPLING_RATIO, 1))
        A = np.matrix(np.exp(2j * np.pi * A_aux / self.SAMPLING_RATIO))

        n_sample = self.N_SAMPLE
        Y_aux = np.array(offsets).reshape(-1, 1) @ np.arange(n_sample).reshape(1, -1)
        Y_coeff = np.exp(-2j * np.pi / (self.SAMPLING_RATIO * n_sample) * Y_aux)
        Y_coeff = torch.tensor(Y_coeff, dtype=torch.complex64)
        A = torch.tensor(A, dtype=torch.complex64)
        return Y_coeff, A

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