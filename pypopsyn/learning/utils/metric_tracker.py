"""
    Metric tracker.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)
"""

from typing import Optional

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class MetricTracker:
    """
    Metric tracker.

    This class is responsible for keeping track of metrics throughout
    training and testing. It is able to update the values, add new
    metrics to be tracked, reset them all or extract higher-level info.
    such as averaging.
    """

    def __init__(
        self,
        keys: Optional[set] = None,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Metric tracker initialization.

        Args:
            keys (Optional[set]): A set of metric keys/identifiers to initialize the tracking.
            writer (Optional[SummaryWriter]): A TensorBoard writer to output metric info to.
        """

        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"]
        )
        self.reset()

    def reset(self) -> None:
        """
        Resets all tracked metrics values to zero.
        """

        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1) -> None:
        """
        Metric update.

        Updates a given metric adding the provided value and a specified count.
        If the metric is not yet tracked, it creates a new track for it.

        Args:
            key (str): Identifier/key for the metric in the dictionary.
            value (float): Value to add to the metric entry.
            n (int): Count value.
        """

        if key not in self._data.index:
            self._data = pd.concat(
                [
                    self._data,
                    pd.DataFrame(
                        index=[key], columns=["total", "counts", "average"]
                    ),
                ]
            )
            self._data.total[key] = 0
            self._data.counts[key] = 0
            self._data.average[key] = 0

        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = (
            self._data.total[key] / self._data.counts[key]
        )

    def avg(self, key: str) -> float:
        """
        Metric average.

        Args:
            key (str): Key of the metric.

        Returns:
            (float): The average of that metric.

        """

        return self._data.average[key]

    def result(self) -> dict:
        """
        Results dictionary.

        Returns:
            (dict): The averages of all tracked metrics in a dictionary.
        """

        return dict(self._data.average)
