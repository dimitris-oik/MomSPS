# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class MomSPS_smooth(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            batch_size: int = 256,
            init_ss: float = 1.0,
            c: float = 1.0,
            beta: float = 0.9,
            gamma: float = 2.0,
            ):
        """
        Implements the smoothed version of the MomSPS optimizer.

        Arguments:
            params (iterable):
                Iterable of parameters to optimize or dicts defining parameter groups.
            batch_size (int):
                The batch size of the iteration.
            init_ss (float):
                The initial step-size, i.e. \gamma_0.
            c (float):
                The normalizing constant of the step-size.
            beta (float):
                The momentum coefficient.
            gamma (float):
                The smoothing coefficient.
        """
        
        defaults = {"batch_size": batch_size, "init_ss": init_ss, "c": c, "beta": beta, "gamma": gamma}
        super().__init__(params, defaults)

        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.coeff = gamma**(1./batch_size)

        self.number_steps = 0
        self.old_x = 0
        self.olds_ss = init_ss

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
        beta = self.beta
        coeff = self.coeff
        grad_norm_sq = self.compute_grad_sq()
        momsps = (1-beta) * (loss/(c*grad_norm_sq))
        momsps = min(momsps.item(), coeff * self.old_ss)

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
                
                self.old_ss = momsps
        
        return loss
    
    @torch.no_grad()   
    def compute_grad_sq(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad.data.detach()
                grad_norm += torch.sum(torch.mul(g, g))
        
        return grad_norm
