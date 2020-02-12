""" Metric Tracker.

    Authors:

        Alberto Garcia Garcia (garciagarcia@ice.csic.es)

    Copyright (c) MAGNESIA (ICE-CSIC)

"""

import pandas as pd


class MetricTracker:
    """ Metric Tracker.

        This class is responsible for keeping track of metrics throughout
        training and testing. It is able to update the values, add new
        metrics to be tracked, reset them all or extract higher-level info.
        such as averaging.

    """

    def __init__(self, *keys, writer=None) -> None:
        """
        Metric tracker initialization.

        Args:
            keys: A set of metric keys/identifiers to initialize the tracking.
            writer: A TensorBoard writer to output metric info to.

        Returns:
            Nothing.

        """

        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"]
        )
        self.reset()

    def reset(self) -> None:
        """ Resets all tracked metrics values to zero.

        Args:
            None.

        Returns:
            Nothing.

        """

        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1) -> None:
        """ Metric update.

        Updates a given metric adding the provided value and a specified count.
        If the metric is not yet tracked, it creates a new track for it.

        Args:
            key: Identifier/key for the metric in the dictionary.
            value: Value to add to the metric entry.
            n: Count value.

        Returns:
            Nothing.

        """

        if self.writer is not None:
            self.writer.add_scalar(key, value)

        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = (
            self._data.total[key] / self._data.counts[key]
        )

    def avg(self, key) -> float:
        """
        Metric average.

        Args:
            key: Key of the metric.

        Returns:
            The average of that metric.

        """

        return self._data.average[key]

    def result(self) -> dict:
        """
        Results dictionary.

        Args:
            None.

        Returns:
            The averages of all tracked metrics in a dictionary.

        """

        return dict(self._data.average)
