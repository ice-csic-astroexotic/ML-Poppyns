"""
    Sampler from a sbi posterior distribution.

    Display help message to run the code:

    python posterior_sampler.py --help

    Displays all the relevant arguments that can be used.

    Authors:

        Celsa Pardo Araujo (pardo@ice.csic.es)
"""

import signal
from types import FrameType
from typing import Optional, Tuple

import torch
from sbi.inference.posteriors.direct_posterior import DirectPosterior


def handler(signum: int, frame: FrameType) -> None:
    """
    Signal handler that raises a TimeoutError when a SIGALRM signal is received. This function is needed in the
    sample_with_timeout function to raise an error if the sampling exceeds the maximum time.

    Args:
        signum (int): The signal number.
        frame (Any): The current stack frame.

    Raises:
        (TimeoutError): Indicates that the operation timed out.
    """
    raise TimeoutError("The operation timed out")


def posterior_sampler(
    posterior: DirectPosterior,
    simulation_output: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
     Perform sampling from the posterior distribution. This function is designed to be used with a signal handler
     to enforce a timeout during sampling, see below.

     Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples (int): The number of samples to draw from the posterior distribution.

    Returns:
        (torch.Tensor): The samples drawn from the posterior distribution.
    """
    return posterior.set_default_x(simulation_output).sample(
        (n_samples,), show_progress_bars=False
    )


def sample_with_timeout(
    posterior: DirectPosterior,
    simulation_output: torch.Tensor,
    n_samples: int,
    timeout: Optional[int] = 180,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    Perform sampling from the posterior distribution with a specified timeout.

    Args:
        posterior (DirectPosterior): Posterior distribution.
        simulation_output (torch.Tensor): Simulation output matrix.
        n_samples (int): The number of samples to draw from the posterior distribution.
        timeout (int, optional): The maximum time in seconds to allow for n_samples to be drawn. Defaults to 180 seconds.

    Returns:
        (Tuple[Optional[torch.Tensor], bool]): A tuple containing the result of the sampling (or None if it times out)
            and a boolean indicating whether the sampling was successful.
    """
    # Set the signal handler and start the alarm to enforce the function timeout.
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        # If the sampling completes before the timeout, return the result and set success to True.
        result = posterior_sampler(posterior, simulation_output, n_samples)
        success = True
    except TimeoutError:
        # If the timeout is reached, return None and set success to False.
        result = None
        success = False
    finally:
        # Cancel the timer whether an exception was raised or not to prevent the signal from being sent after
        # completion.
        signal.alarm(0)

    return result, success
