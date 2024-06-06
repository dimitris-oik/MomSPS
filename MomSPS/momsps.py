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
        self.state["p"] = 0

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
        grad_norm = self.compute_grad_terms()
        momsps = (1-beta) * min(loss/(c*grad_norm**2), gamma_max)

        for group in self.param_groups:
            for p in group['params']:
                # new_p = x^{t+1}
                # p = x^t
                # old_p = x^{t-1}
                grad = p.grad.data.detach()
                state = self.state[p]

                if self.number_steps == 1:
                    state["p"] = p.detach().clone()
                    new_p = p - momsps * grad
                else:
                    old_p = state["p"]
                    state["p"] = p.detach().clone()
                    new_p = p - momsps * grad + beta * (p-old_p)

                with torch.no_grad():
                        p.copy_(new_p)
        
        return loss
    
    @torch.no_grad()   
    def compute_grad_terms(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad.data.detach()
                grad_norm += torch.sum(torch.mul(g, g))
          
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm
