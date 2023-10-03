from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["step"] += 1
                alpha = 1.0 - beta1
                m1 = (m * beta1 ) + (grad*alpha)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                assert(torch.allclose(m, m1))
                v1 = (v*beta2) + ((1.0 - beta2)*(grad*grad))
                # v = v1
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                assert(torch.allclose(v1, v))
                # Update first and second moments of the gradients
                m_hat = m / (1 - beta1 ** state["step"])
                v_hat = v / (1 - beta2 ** state["step"])

                if correct_bias:
                    step_size = lr / (v_hat.sqrt().add_(eps))
                else:
                    step_size = lr

                # Update parameters
                # Please note: you should update p.data (not p), to avoid an error about a leaf Variable being used in an in-place operation

                p.data.addcmul_(m_hat, step_size, value=-1)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

        return loss
