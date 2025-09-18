import contextlib
import logging
from typing import Type
from alibi_detect.utils._types import TorchDeviceType
from packaging import version

import numpy as np
import torch

logger = logging.getLogger(__name__)


def zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """
    Set the diagonal of a matrix to 0

    Parameters
    ----------
    mat
        A 2D square matrix

    Returns
    -------
    A 2D square matrix with zeros along the diagonal
    """
    return mat - torch.diag(mat.diag())


def quantile(sample: torch.Tensor, p: float, type: int = 7, sorted: bool = False) -> float:
    """
    Estimate a desired quantile of a univariate distribution from a vector of samples

    Parameters
    ----------
    sample
        A 1D vector of values
    p
        The desired quantile in (0,1)
    type
        The method for computing the quantile.
        See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    sorted
        Whether or not the vector is already sorted into ascending order

    Returns
    -------
    An estimate of the quantile

    """
    N = len(sample)

    if len(sample.shape) != 1:
        raise ValueError("Quantile estimation only supports vectors of univariate samples.")
    if not 1/N <= p <= (N-1)/N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")

    sorted_sample = sample if sorted else sample.sort().values

    if type == 6:
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    h_floor = int(h)
    quantile = sorted_sample[h_floor-1]
    if h_floor != h:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])

    return float(quantile)


def get_device(device: TorchDeviceType = None) -> torch.device:
    """
    Instantiates a PyTorch device object.

    Parameters
    ----------
    device
        Either `None`, a str ('gpu', 'cuda' or 'cpu') indicating the device to choose, or an already instantiated device
        object. If `None`, the GPU is selected if it is detected, otherwise the CPU is used as a fallback.

    Returns
    -------
    The instantiated device object.
    """
    if isinstance(device, torch.device):  # Already a torch device
        return device
    else:  # Instantiate device
        if device is None or device.lower() in ['gpu', 'cuda']:
            torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch_device.type == 'cpu':
                logger.warning('No GPU detected, fall back on CPU.')
        else:
            torch_device = torch.device('cpu')
            if device.lower() != 'cpu':
                logger.warning('Requested device not recognised, fall back on CPU.')
    return torch_device


def get_optimizer(name: str = 'Adam') -> Type[torch.optim.Optimizer]:
    """
    Get an optimizer class from its name.

    Parameters
    ----------
    name
        Name of the optimizer.

    Returns
    -------
    The optimizer class.
    """
    optimizer = getattr(torch.optim, name, None)

    if optimizer is None:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    return optimizer


# Source of the following function:
# https://github.com/huggingface/transformers/blob/58e13b9f129bb0dccc3b51e5da22f45ef3ff0ae7/src/transformers/trainer.py#L276C1-L292C55
def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    # tests complain about float64DType, so we add it as well
    allowlist += [type(np.dtype(np.float64))]

    return torch.serialization.safe_globals(allowlist)

