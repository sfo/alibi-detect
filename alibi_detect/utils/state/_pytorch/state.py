"""
Submodule to handle saving and loading of detector state dictionaries when the dictionaries contain `torch.Tensor`'s.
"""
from pathlib import Path
import numpy
import torch

from alibi_detect.utils.pytorch.misc import safe_globals


def save_state_dict(state_dict: dict, filepath: Path):
    """
    Utility function to save a detector's state dictionary to a filepath using `torch.save`.

    Parameters
    ----------
    state_dict
        The state dictionary to save.
    filepath
        Directory to save state dictionary to.
    """
    # Save to disk
    torch.save(state_dict, filepath)


def load_state_dict(filepath: Path) -> dict:
    """
    Utility function to load a detector's state dictionary from a filepath with `torch.load`.

    Parameters
    ----------
    filepath
        Directory to load state dictionary from.

    Returns
    -------
    The loaded state dictionary.
    """
    with safe_globals():
        state_dict = torch.load(filepath)
    return state_dict
