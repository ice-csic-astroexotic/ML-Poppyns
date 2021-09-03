"""
Network benchmarking.

Utility functions for benchmarking model performance.
https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b

Authors:

    Alberto Garcia Garcia (garciagarcia@ice.csic.es)
    Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import time
import typing

import numpy as np
import torch


def measure(
    model: torch.nn.Module,
    device: torch.device,
    input_dummy: torch.tensor,
    output_dummy: torch.tensor,
) -> typing.Tuple[float, float]:

    """
    Measure timing for one single forward and backward pass with the model and
    the specified device.

    Args:
        model (torch.module): model to benchmark.
        device (torch.device): device in which the model will be executed.
        input_dummy (torch.tensor): dummy tensor for input purposes.
        output_dummy (torch.tensor): dummy tensor for output purposes.

    Returns:
        A tuple (float, float) that contains the time spent in the forward pass
        and the runtime of the backward pass, both in seconds.

    """

    # Synchronize gpu time and measure forward pass.
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    y_pred = model(input_dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_forward = time.time() - t0

    # Zero gradients, synchronize time and measure backward pass.
    model.zero_grad()
    t0 = time.time()
    y_pred.backward(output_dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_backward = time.time() - t0

    return elapsed_forward, elapsed_backward


def benchmark(
    model: torch.nn.Module,
    device: torch.device,
    input_dummy: torch.tensor,
    output_dummy: torch.tensor,
) -> typing.Tuple[float, float]:

    """
    Measure median time for forward/backward passes of a model on a device.

    Args:
        model (torch.module): model to benchmark.
        device (torch.device): device in which the model will be executed.
        input_dummy (torch.tensor): dummy tensor for input purposes.
        output_dummy (torch.tensor): dummy tensor for output purposes.

    Returns:
        A tuple (float, float) that contains the median time spent in the
        forward pass and the backward pass, both in milliseconds.

    """

    # Move dummies to the required device (CPU or GPU).
    input_dummy = input_dummy.to(device)
    output_dummy = output_dummy.to(device)

    # Dry runs.
    num_dry_runs = 5
    for i in range(num_dry_runs):
        _, _ = measure(model, device, input_dummy, output_dummy)

    # Benchmarking for a defined number of repetitions.
    num_repetitions = 1024
    t_forward = []
    t_backward = []

    for i in range(num_repetitions):
        t_fp, t_bp = measure(model, device, input_dummy, output_dummy)
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # Compute medians for forward and backward passes, convert to milliseconds.
    t_forward = np.median(np.asarray(t_forward) * 1e3)
    t_backward = np.median(np.asarray(t_backward) * 1e3)

    return t_forward, t_backward


def measure_multimodal(
    model: torch.nn.Module,
    device: torch.device,
    input_dummy_1: torch.tensor,
    input_dummy_2: torch.tensor,
    output_dummy: torch.tensor,
) -> typing.Tuple[float, float]:

    """
    Measure timing for one single forward and backward pass with a multimodal model and
    the specified device.

    Args:
        model (torch.module): model to benchmark.
        device (torch.device): device in which the model will be executed.
        input_dummy_1 (torch.tensor): dummy tensor for input 1.
        input_dummy_2 (torch.tensor): dummy tensor for input 2.
        output_dummy (torch.tensor): dummy tensor for output purposes.

    Returns:
        A tuple (float, float) that contains the time spent in the forward pass
        and the runtime of the backward pass, both in seconds.

    """

    # Synchronize gpu time and measure forward pass.
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    y_pred = model(input_dummy_1, input_dummy_2)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_forward = time.time() - t0

    # Zero gradients, synchronize time and measure backward pass.
    model.zero_grad()
    t0 = time.time()
    y_pred.backward(output_dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_backward = time.time() - t0

    return elapsed_forward, elapsed_backward


def benchmark_multimodal(
    model: torch.nn.Module,
    device: torch.device,
    input_dummy_1: torch.tensor,
    input_dummy_2: torch.tensor,
    output_dummy: torch.tensor,
) -> typing.Tuple[float, float]:

    """
    Measure median time for forward/backward passes of a multimodal model on a device.

    Args:
        model (torch.module): model to benchmark.
        device (torch.device): device in which the model will be executed.
        input_dummy_1 (torch.tensor): dummy tensor for input 1.
        input_dummy_2 (torch.tensor): dummy tensor for input 2.
        output_dummy (torch.tensor): dummy tensor for output purposes.

    Returns:
        A tuple (float, float) that contains the median time spent in the
        forward pass and the backward pass, both in milliseconds.

    """

    # Move dummies to the required device (CPU or GPU).
    input_dummy_1 = input_dummy_1.to(device)
    input_dummy_2 = input_dummy_2.to(device)
    output_dummy = output_dummy.to(device)

    # Dry runs.
    num_dry_runs = 5
    for i in range(num_dry_runs):
        _, _ = measure_multimodal(
            model, device, input_dummy_1, input_dummy_2, output_dummy
        )

    # Benchmarking for a defined number of repetitions.
    num_repetitions = 1024
    t_forward = []
    t_backward = []

    for i in range(num_repetitions):
        t_fp, t_bp = measure_multimodal(
            model, device, input_dummy_1, input_dummy_2, output_dummy
        )
        t_forward.append(t_fp)
        t_backward.append(t_bp)

    # Compute medians for forward and backward passes, convert to milliseconds.
    t_forward = np.median(np.asarray(t_forward) * 1e3)
    t_backward = np.median(np.asarray(t_backward) * 1e3)

    return t_forward, t_backward
