import logging
from typing import Optional, List

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from deepod.models.time_series.dcdetector import DCdetectorModel, my_kl_loss

from detection.detectors.helpers.torch_helpers import get_and_set_device


class DeepODDcDetector:
    """`DCdetectorModel` wrapper with adapted training and inference steps.

    This model uses a dual attention encoder to process time series data, which has been split into
    multiple scales of patches. It learns representations at both patch and sub-patch scales and
    aggregates them to form a comprehensive feature set. The model can output attention weights for
    interpretability if required.

    Args:
        window_size: window size.
        n_features: number of input features.
        n_encoder_layers: number of encoder layers.
        n_attention_heads: number of attention heads.
        d_model: encoding dimensionality of the model.
        patch_sizes: sizes of patches to split the input data into (defaults to `[1, 5]`).
        dropout: dropout rate.
    """

    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 12,
        n_encoder_layers: int = 2,
        n_attention_heads: int = 1,
        d_model: int = 256,
        patch_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        if patch_sizes is None:
            patch_sizes = [1, 5]
        self.model = DCdetectorModel(
            win_size=window_size,
            enc_in=n_features,
            c_out=n_features,
            n_heads=n_attention_heads,
            d_model=d_model,
            e_layers=n_encoder_layers,
            patch_size=patch_sizes,
            channel=n_features,
            d_ff=-1,  # never used.
            dropout=dropout,
            activation="gelu",  # never used.
            output_attention=True,  # no output if False
        )
        self.device = get_and_set_device(self.model)
        logging.info(f"Device: {self.device}")

    def get_batch_losses(self, batch: Tensor) -> (Tensor, Tensor, Tensor):
        """Returns the average loss, series loss and prior loss for the provided `batch`."""
        batch = batch.float().to(self.device)
        series, prior = self.model(batch)
        series_loss = 0.0
        prior_loss = 0.0
        # TODO: `series_loss` and `prior_loss` are always the same, since
        #  detaching does not alter values.
        for u in range(len(prior)):
            series_loss += torch.mean(
                my_kl_loss(
                    series[u],
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.model.win_size
                        )
                    ).detach(),
                )
            ) + torch.mean(
                my_kl_loss(
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.model.win_size
                        )
                    ).detach(),
                    series[u],
                )
            )
            prior_loss += torch.mean(
                my_kl_loss(
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.model.win_size
                        )
                    ),
                    series[u].detach(),
                )
            ) + torch.mean(
                my_kl_loss(
                    series[u].detach(),
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                            1, 1, 1, self.model.win_size
                        )
                    ),
                )
            )
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        # TODO: not sure why not adding instead of subtracting, plus will always be zero this way.
        loss = prior_loss - series_loss
        return loss, series_loss, prior_loss

    def get_window_scores(
        self, loader: DataLoader, temperature: float = 50.0
    ) -> NDArray[np.float32]:
        """Returns the anomaly scores for the windows in `loader`.

        The method uses a temperature parameter to scale the losses and applies a softmax to
        obtain a probability distribution over the anomaly scores, which can be used to rank the inputs
        by their likelihood of being anomalies.

        Args:
            loader: data loader providing the windows to return anomaly scores for.
            temperature: temperature parameter to scale the losses.

        Returns:
            The window anomaly scores.
        """
        self.model.eval()
        attens_energy = []
        for batch in loader:
            batch = batch.float().to(self.device)
            series, prior = self.model(batch)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.model.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.model.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.model.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.model.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        # shape `(n_windows, n_features)`
        window_scores = np.concatenate(attens_energy, axis=0, dtype=np.float32)
        # # shape `(n_windows,)`
        return np.mean(window_scores, axis=1)
