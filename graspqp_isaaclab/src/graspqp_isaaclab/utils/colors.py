# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for colors and visualization."""

from __future__ import annotations

import colorsys
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=128)
def get_color_map(length: int) -> list[tuple[float, float, float]]:
    """Generate a color palette of [length] colors.
    Args:
        length (int): Number of colors to generate.
    Returns:
        list[tuple[float, float, float]]: List with different colors ranging
            from [0,1].
    """
    brightness = 0.7
    hsv = [(i / length, 1, brightness) for i in range(length)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    s = np.random.get_state()

    # Permute color to not have smooth color changes
    np.random.seed(0)
    result = [tuple(colors[i]) for i in np.random.permutation(len(colors))]
    np.random.set_state(s)
    return result


def cmap(data: torch.Tensor | np.ndarray, name: str = "viridis", min=None, max=None, with_alpha=True) -> np.ndarray:
    """Convert a tensor to a color map.
    Args:
        data (torch.Tensor | np.ndarray): Tensor to convert to color map.
        name (str): Name of the color map. Defaults to "viridis".
    Returns:
        np.ndarray: Color map.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name)
    if min is None:
        min = data.min()
    if max is None:
        max = data.max()

    data = (data - min) / (max - min)
    if with_alpha:
        return cmap(data)
    else:
        return cmap(data)[:, :3]
