import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        n_warmup_steps: int,
        decay: float = -0.5,
        init_lr=None,
    ):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.decay = decay
        assert self.decay < 0
        if init_lr is None:
            self.init_lr = np.power(256, self.decay)
        else:
            self.init_lr = init_lr

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        if self.n_warmup_steps == 0:
            return np.power(self.n_current_steps, self.decay)
        return np.min(
            [
                np.power(self.n_current_steps, self.decay),
                np.power(self.n_warmup_steps, self.decay - 1.0) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: 1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle
        self.is_warming_up = True

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps and self.is_warming_up:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            self.is_warming_up = False
            self.warmup_steps = 1
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle > self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch > self.first_cycle_steps:
                self.is_warming_up = False
                self.warmup_steps = 1
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def add_noise(x: torch.Tensor, snr: float = None) -> torch.Tensor:
    x = x[..., 0::2] + 1j * x[..., 1::2]

    if snr is None:
        snr = np.random.uniform(0, 16)
    sig_p_sqrt = torch.sqrt(torch.mean(torch.abs(x) ** 2))
    noise_scale = sig_p_sqrt / (10 ** (snr / 20)) / np.sqrt(2)
    n = torch.normal(0, 1, size=x.size(), device=x.device) + 1j * torch.normal(
        0, 1, size=x.size(), device=x.device
    )
    n = n * noise_scale.to(x.device)
    x += n

    x = torch.stack((x.real, x.imag), dim=-1).flatten(-2, -1)
    return x


def main():
    import torch.optim

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, 200, 1, warmup_steps=5, max_lr=0.001, min_lr=1e-12
    )
    rates = []
    for i in range(1200):
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    import matplotlib.pyplot as plt

    plt.plot(rates)
    plt.show()


if __name__ == "__main__":
    main()
