import math

import torch


class ConsinAnnealingWarmupRestartsWithDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: float = 1.0,
        eta_min: float = 0.0001,
        T_warmup: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        """

        Args:
            optimizer: the base optimizer
            T_0: the base number of iterations for a cycle
            T_mult: the scaling factor for how much a cycle lasts
            eta_min: the minimum learning rate
            T_warmup: the number of linear warmup iterations
            gamma: the exponential decay factor for the maximum learning rate
            last_epoch: the last epoch

        """

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.gamma = gamma

        # iterations in current cycle
        self.T_cur = 0
        self._last_lr = 0
        self.N_cycle = 0
        self._C_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs

        elif self.T_cur < self.T_warmup:
            return [(base_lr - self.eta_min) * self.T_cur / self.T_warmup + self.eta_min for base_lr in self.base_lrs]

        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_warmup) / (self.T_i - self.T_warmup)))
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.N_cycle = self.N_cycle + 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    # reset current epoch and compute current cycle
                    self.T_cur = epoch % self.T_0
                    self.N_cycle = epoch // self.T_0
                else:
                    # compute through how many cycles we have exponentiated the cycle size
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.N_cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1) - self.T_warmup
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self.base_lrs = [group["initial_lr"] * (self.gamma**self.N_cycle) for group in self.optimizer.param_groups]
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
