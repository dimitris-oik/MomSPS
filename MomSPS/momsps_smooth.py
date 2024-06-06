# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class MomSPS_smooth(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            n_batches_per_epoch: int = 256,
            init_step_size: float = 1.0,
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

        defaults = {"n_batches_per_epoch": n_batches_per_epoch, "init_step_size": init_step_size, "c": c, "beta": beta, "gamma": gamma}
        super().__init__(params, defaults)

        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.coeff = gamma**(1./n_batches_per_epoch)

        self.number_steps = 0
        self.state["p"] = 0
        self.state["momsps"] = init_step_size

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
        grad_norm = self.compute_grad_terms()
        momsps = (1-beta) * (loss/(c*grad_norm**2))
        momsps = min(momsps.item(), coeff * self.state["momsps"])

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
                
                self.state["momsps"] = momsps
        
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
