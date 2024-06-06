# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class MomSPS(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            c: float = 1.0,
            gamma_max: float = 10.0,
            beta: float = 0.9,
            ):
        """
        Implements the MomSPS optimizer.

        Arguments:
            params (iterable):
                Iterable of parameters to optimize or dicts defining parameter groups.
            c (float):
                The normalizing constant of the step-size.
            gamma_max (float):
                The upper bound of the step-size.
            beta (float):
                The momentum coefficient.
        """

        defaults = {"c": c, "gamma_max": gamma_max, "beta": beta}
        super().__init__(params, defaults)

        self.c = c
        self.gamma_max = gamma_max
        self.beta = beta
        self.number_steps = 0
        self.old_x = 0

    def step(self, loss=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        loss : torch.tensor
            The loss tensor. Use this when the backward step has already been performed. By default None.
        
        Returns
        -------
        (Stochastic) Loss function value.
        """
        
        self.number_steps += 1
        c = self.c
        gamma_max = self.gamma_max
        beta = self.beta
        grad_norm_sq = self.compute_grad_sq()
        momsps = (1-beta) * min(loss/(c*grad_norm_sq), gamma_max)

        for group in self.param_groups:
            for x in group['params']:
                # new_x = x^{t+1}
                # x = x^t
                # old_x = x^{t-1}
                grad = x.grad.data.detach()

                if self.number_steps == 1:
                    self.old_x = x.detach().clone()
                    new_x = x - momsps * grad
                else:
                    old_x = self.old_x
                    self.old_x = x.detach().clone()
                    new_x = x - momsps * grad + beta * (x-old_x)

                with torch.no_grad():
                        x.copy_(new_x)
        
        return loss
    
    @torch.no_grad()   
    def compute_grad_sq(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad.data.detach()
                grad_norm += torch.sum(torch.mul(g, g))
        
        return grad_norm
