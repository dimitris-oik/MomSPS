# MomSPS

**Adaptive Polyak step sizes for Stochastic Heavy Ball — no learning-rate tuning required.**

[![PyPI](https://img.shields.io/pypi/v/momsps)](https://pypi.org/project/momsps/)
[![arXiv](https://img.shields.io/badge/arXiv-2406.04142-b31b1b)](https://arxiv.org/abs/2406.04142)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`MomSPS` is the official PyTorch implementation of the optimizer proposed in:

> **Stochastic Polyak Step-sizes and Momentum: Convergence Guarantees and Practical Performance**
> Dimitris Oikonomou, Nicolas Loizou. *ICLR 2025.*

The package provides two `torch.optim.Optimizer` subclasses — `MomSPS` and `MomSPS_smooth` — that adapt the Stochastic Heavy Ball (SHB) step size at every iteration using a Polyak-style rule derived from the Iterate Moving Average viewpoint. The result is a momentum optimizer that **converges in non-interpolated regimes**, matches the deterministic Heavy Ball rate under interpolation, and **eliminates learning-rate tuning** in deep-learning training.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The algorithm](#the-algorithm)
- [API reference](#api-reference)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

---

## Installation

From PyPI:

```bash
pip install momsps
```

Or from source:

```bash
git clone https://github.com/dimitris-oik/MomSPS.git
cd MomSPS
pip install -e .
```

Requirements: `torch`, `numpy`, Python 3.8+.

---

## Quick start

`MomSPS` needs access to the **(mini-batch) loss value** inside `.step()` because the step size is computed from it. The only change to a standard PyTorch training loop is passing `loss=` to `.step()`:

```python
import torch
from momsps import MomSPS

model     = MyModel()
optimizer = MomSPS(model.parameters())      # defaults: c=1.0, gamma_max=10.0, beta=0.9
criterion = torch.nn.CrossEntropyLoss()

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(loss=loss)               # <-- the only change vs. SGD/Adam
```

For the smoothed variant (recommended in DNN training, matches the experiments in §4 of the paper):

```python
from momsps import MomSPS_smooth

optimizer = MomSPS_smooth(
    model.parameters(),
    n_batches_per_epoch=len(loader),
)
```

---

## The algorithm

Given a mini-batch loss $f_{S_t}$ and momentum coefficient $\beta$, **MomSPS<sub>max</sub>** sets

$$\gamma_t = (1 - \beta) \cdot \min\left\\{ \frac{f_{S_t}(x^t) - \ell^*_{S_t}}{c \\, \\|\nabla f_{S_t}(x^t)\\|^2},\ \gamma_b \right\\}$$

and applies the SHB update

$$x^{t+1} = x^t - \gamma_t \\, \nabla f_{S_t}(x^t) + \beta (x^t - x^{t-1}).$$

The smoothed variant **MomSPS_smooth** replaces the constant cap $\gamma_b$ by an iteration-dependent cap $\gamma_b^t = \tau^{b/n} \gamma_{t-1}$ with $\tau \approx 2$, which empirically yields more stable schedules on deep networks.

Key theoretical properties of MomSPS<sub>max</sub> (full statements: Theorems 3.2, Corollary 3.3, Corollary 3.4 of the paper):

| Property | MomSPS<sub>max</sub> |
|---|---|
| Convex, smooth, non-interpolated | $O(1/T)$ to a neighborhood |
| Deterministic Heavy Ball ($S_t = [n]$) | $O(1/T)$ to the exact minimizer |
| Convex, smooth, interpolated | $O(1/T)$ to the exact minimizer |

---

## API reference

### `MomSPS(params, c=1.0, gamma_max=10.0, beta=0.9)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters to optimize (as in any `torch.optim.Optimizer`). |
| `c` | float | `1.0` | Normalizing constant in the Polyak step size. |
| `gamma_max` | float | `10.0` | Upper bound $\gamma_b$ on the step size. |
| `beta` | float | `0.9` | Heavy-ball momentum coefficient. |

### `MomSPS_smooth(params, n_batches_per_epoch=256, init_step_size=1.0, c=1.0, beta=0.9, gamma=2.0)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters to optimize. |
| `n_batches_per_epoch` | int | `256` | Number of mini-batches per epoch, used to set the smoothing rate $\tau^{b/n}$. |
| `init_step_size` | float | `1.0` | Initial value of $\gamma_b$ (i.e. $\gamma_0$). |
| `c` | float | `1.0` | Normalizing constant in the Polyak step size. |
| `beta` | float | `0.9` | Heavy-ball momentum coefficient. |
| `gamma` | float | `2.0` | Smoothing base $\tau$ in $\gamma_b^t = \tau^{b/n}\gamma_{t-1}$. |

Both classes expose the standard `torch.optim.Optimizer` interface. The only non-standard requirement is that `.step()` is called as `optimizer.step(loss=loss)`.

---

## Citation

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{oikonomou2025stochastic,
  title = {Stochastic Polyak Step-sizes and Momentum: Convergence Guarantees and Practical Performance},
  author = {Oikonomou, Dimitris and Loizou, Nicolas},
  booktitle = {ICLR},
  year = {2025},
}
```

---

## License

Released under the [MIT License](LICENSE).
