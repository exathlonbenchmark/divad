from typing import Optional, Union, Iterable
from argparse import Namespace

import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD, RMSprop, Adadelta, Adam, NAdam, AdamW

# known bug fixed in later version (https://github.com/huggingface/accelerate/issues/1019)
# from torch.optim.lr_scheduler import LRScheduler

from utils.guarding import check_is_not_none


# pytorch classes corresponding to string parameters (i.e. "parameter classes")
PC = {
    # class and keyword arguments to overwrite
    "opt": {
        "nag": (SGD, {"momentum": 0.9, "nesterov": True}),
        "adam": (Adam, dict()),
        "nadam": (NAdam, dict()),
        "adamw": (AdamW, {"betas": (0.9, 0.999)}),
        "rmsprop": (RMSprop, dict()),
        "adadelta": (Adadelta, dict()),
    }
}
PC = Namespace(**PC)


def get_and_set_device(model: Optional[nn.Module] = None) -> torch.device:
    """Returns the device according to CUDA available, and sets `device` for model if provided."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if model is not None:
            model.cuda()
    else:
        device = torch.device("cpu")
    return device


def get_optimizer(
    params: Union[Iterable, dict],
    optimizer: str,
    learning_rate: float,
    adamw_weight_decay: Optional[float] = None,
) -> Optimizer:
    """Returns the optimizer object for the provided string and parameters.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        optimizer: optimizer string (must be a key of `PC.opt`).
        learning_rate: learning rate used by the optimization algorithm.
        adamw_weight_decay: weight decay used for the AdamW optimizer if relevant.

    Returns:
        The pytorch optimizer object.
    """
    opt_class, opt_kwargs = PC.opt[optimizer]
    if optimizer == "adamw":
        check_is_not_none(
            adamw_weight_decay, "Weight decay should be provided for AdamW optimizer."
        )
        opt_kwargs = dict(opt_kwargs, **{"weight_decay": adamw_weight_decay})
    return opt_class(params, lr=learning_rate, **opt_kwargs)


class EarlyStopper:
    """Pytorch early stopper."""

    def __init__(self, patience: int = 1, min_delta: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float("inf")

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > self.min_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Checkpointer:
    """Pytorch checkpointer.

    Args:
        model: pytorch model whose states to checkpoint.
        optimizer: corresponding pytorch optimizer.
        optimizer: corresponding pytorch scheduler.
        full_output_path: Full output path, including the model name and extension.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler=None,
        full_output_path: str = "model.pt",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.min_loss = float("inf")
        self.full_output_path = full_output_path

    def checkpoint(self, epoch, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            saved = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            }
            if self.scheduler is not None:
                saved["scheduler"] = self.scheduler.state_dict()
            torch.save(saved, self.full_output_path)
