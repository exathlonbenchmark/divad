import logging
from typing import Union, List, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset

from numpy.typing import NDArray

from utils.guarding import check_value_in_choices
from data.helpers import get_sliding_windows
from detection.detectors.helpers.torch_helpers import get_and_set_device
from detection.detectors.helpers.torch_tranad_helpers import TranAD


class TorchTranad:
    """`TranAD` wrapper with adapted training and inference steps.

    Args:
        window_size: window size.
        n_features: number of input features.
        dim_feedforward: dimension of feed-forward layers.
        last_activation: activation function of the last FC layer (either "sigmoid" or "linear").
    """

    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 12,
        dim_feedforward: int = 64,
        last_activation: str = "sigmoid",
    ):
        check_value_in_choices(
            last_activation, "last_activation", ["sigmoid", "linear"]
        )
        self.model = TranAD(
            n_window=window_size,
            feats=n_features,
            dim_feedforward=dim_feedforward,
            last_activation=last_activation,
        )
        self.criterion = nn.MSELoss(reduction="none")
        self.device = get_and_set_device(self.model)
        logging.info(f"Device: {self.device}")

    def get_batch_loss(self, batch: Tensor, epoch: int) -> Tensor:
        """Returns the average loss for the provided `batch`.

        Args:
            batch: the current batch.
            epoch: the current epoch number (starting from zero).

        Returns:
            The average loss for the batch.
        """
        n = epoch + 1
        # shape `(batch_size, window_size, n_features)`
        batch = batch.float().to(self.device)

        local_bs = batch.shape[0]
        # shape `(window_size, batch_size, n_features)`
        window = batch.permute(1, 0, 2)
        # shape `(1, batch_size, n_features)`
        elem = window[-1, :, :].view(1, local_bs, self.model.n_feats)

        window = window.float().to(self.device)
        elem = elem.float().to(self.device)
        z = self.model(window, elem)
        # shape `(1, batch_size, n_features)`
        l1 = (1 / n) * self.criterion(z[0], elem) + (1 - 1 / n) * self.criterion(
            z[1], elem
        )
        loss = torch.mean(l1)
        return loss

    def get_window_scores(self, loader: DataLoader) -> NDArray[np.float32]:
        """Returns the anomaly scores for the windows in `loader` (which should contain only one batch).

        Args:
            loader: data loader providing the batch of windows to return anomaly scores for.

        Returns:
            The window anomaly scores.
        """
        self.model.eval()
        if len(loader) != 1:
            raise ValueError(
                "The data loader should only contain one batch at test time."
            )
        l1s = []
        # batch_size = tot_windows in sequence in inference mode: only one batch.
        for d in loader:
            local_bs = d.shape[0]
            # shape `(window_size, batch_size, n_features)`
            window = d.permute(1, 0, 2)
            # shape `(1, batch_size, n_features)`
            elem = window[-1, :, :].view(1, local_bs, self.model.n_feats)
            window = window.float().to(self.device)
            elem = elem.float().to(self.device)
            z = self.model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            # shape `(1, batch_size, n_features)`
            l1 = self.criterion(z, elem)[0]
            l1 = l1.data.cpu()
            l1s.append(l1)
        l1s = torch.cat(l1s)
        # shape `(batch_size, n_features)`
        l1s = l1s.numpy()
        # # shape `(batch_size,)`
        window_scores = np.mean(l1s, axis=1)
        return window_scores


def get_tranad_loader(
    X: NDArray[np.float32], info: dict, batch_size: int
) -> DataLoader:
    """Returns the data loader to use for training the TranAD model.

    Since anomaly ranges could have been removed, windows having the same `"period_id"` might have
    been cut into multiple contiguous subsequences.

    Args:
        X: windows of shape `(n_windows, window_size, n_features)`.
        info: corresponding windows information, with keys `"sample_id"` and `"period_id"`.
        batch_size: batch size.

    Returns:
        The data loader, with every batch being contiguous windows from a same sequence.
    """
    # sort samples by their unique ID (IDs are themselves sorted chronologically and by sequence)
    sorted_id_ids = np.argsort(info["sample_id"])
    X = X[sorted_id_ids]
    sample_ids = info["sample_id"][sorted_id_ids]
    seq_ids = info["period_id"][sorted_id_ids]
    seq_to_batched_X = dict()
    for seq_id in np.unique(seq_ids):
        seq_mask = seq_ids == seq_id
        seq_X = X[seq_mask]
        seq_sample_ids = sample_ids[seq_mask]
        seq_id_ranges = get_contiguous_ids_ranges(seq_sample_ids)
        for i, seq_id_range in enumerate(seq_id_ranges):
            final_seq_id = f"{seq_id}__{i}"
            final_seq_mask = np.isin(seq_sample_ids, seq_id_range)
            final_seq_X = seq_X[final_seq_mask]
            seq_to_batched_X[final_seq_id] = get_sliding_windows(
                final_seq_X,
                window_size=batch_size,
                window_step=batch_size,
                include_remainder=False,
                dtype=np.float32,
                ranges_only=False,
            )
            logging.info(
                f"{final_seq_id}: {final_seq_X.shape[0]} points to batches of shape "
                f"{seq_to_batched_X[final_seq_id].shape}."
            )
    batched_X = ConcatDataset(list(seq_to_batched_X.values()))
    loader = DataLoader(batched_X, batch_size=None, shuffle=True)
    return loader


def get_contiguous_ids_ranges(
    ids: NDArray[np.int32], start_end_only: bool = False
) -> Union[List[NDArray[np.int32]], List[Tuple]]:
    """Returns contiguous sequences of IDs (ID step of 1) in the provided *sorted* `ids`.

    Example:

    ```
    ids = (4, 27, 28, 29, 56, 57, 68)  # assumed already sorted
    id_ranges = get_contiguous_ids_ranges(ids, False)
    print(id_ranges)  # [np.array([4]), np.array([27, 28, 29]), np.array([56, 57]), np.array([68])]
    id_ranges_ids = get_contiguous_ids_ranges(ids, True)
    print(id_ranges_ids)  # [(0, 1), (1, 4), (4, 6), (6, 7)] - included start, excluded end.
    ```

    Args:
        ids: indices of shape `(n_ids,)`. E.g., `(4, 27, 28, 29, 56, 57, 68)`.
        start_end_only: only return the start and end indices of contiguous ranges in `ids`.

    Returns:
        The contiguous ID ranges if start_end_only is `False`, else the start and end indices of contiguous
         ID ranges in `ids`.
    """
    end_ids = np.concatenate([np.where(np.diff(ids) != 1)[0] + 1, [len(ids)]])
    start_ids = np.concatenate([[0], end_ids[:-1]])
    id_ranges = [(s, e) for s, e in zip(start_ids, end_ids)]
    if start_end_only:
        return id_ranges
    returned_ids = []
    for s, e in id_ranges:
        returned_ids.append(ids[s:e])
    return returned_ids
