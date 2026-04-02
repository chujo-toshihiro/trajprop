"""SPICE kernel loading utility."""

from __future__ import annotations

import os

import spiceypy as spice


def init_spice(
    kernel_dir: str,
    kernel_list: list[str] | None = None,
) -> None:
    """Load SPICE kernels from a directory.

    Parameters
    ----------
    kernel_dir : str
        Directory containing SPICE kernel files.
    kernel_list : list[str], optional
        Filenames to load.  When ``None``, all ``.bsp``, ``.tls``, and
        ``.tpc`` files in ``kernel_dir`` are loaded.

    Raises
    ------
    FileNotFoundError
        If a specified kernel file does not exist.
    """

    if kernel_list is None:
        kernel_list = [
            f
            for f in os.listdir(kernel_dir)
            if f.lower().endswith((".bsp", ".tls", ".tpc"))
        ]

    for kernel in kernel_list:
        path = os.path.join(kernel_dir, kernel)

        if os.path.exists(path):
            spice.furnsh(path)
        else:
            raise FileNotFoundError(f"kernel not found: {path}.")
