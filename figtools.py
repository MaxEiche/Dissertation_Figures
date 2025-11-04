# figtools.py — helper utilities for unified figure styling
# ----------------------------------------------------------
# Example usage:
#
#   import matplotlib.pyplot as plt
#   from figtools import use_style, new_figure, savefig, PALETTE
#
#   with use_style():
#       fig, ax = new_figure(width="single")
#       ax.plot(x, y, color=PALETTE["primary"], label="Data")
#       ax.set_xlabel("Time [h]")
#       ax.set_ylabel("Signal [a.u.]")
#       ax.legend()
#       savefig(fig, "figures/figure1.pdf")

from contextlib import contextmanager
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# ==========================================================
# Color palette — Okabe–Ito (colorblind-friendly)
# ==========================================================
PALETTE = {
    "primary":   "#0072B2",  # blue
    "secondary": "#E69F00",  # orange
    "accent":    "#009E73",  # green
    "alert":     "#D55E00",  # red-orange
    "magenta":   "#CC79A7",  # magenta
    "sky":       "#56B4E9",  # light blue
    "yellow":    "#F0E442",  # yellow
    "black":     "#000000",
}

# Optional marker/linestyle dictionaries for consistent use
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

# Typical journal column widths in millimeters
COL_WIDTHS_MM = {
    "single": 85,    # single-column (Nature / Cell)
    "onehalf": 114,  # ~1.5 columns
    "double": 174,   # full width
}

# ==========================================================
# Helpers
# ==========================================================
def mm_to_inches(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / 25.4


def _figure_width_inches(width):
    """Accept a keyword ('single', 'double', ...) or numeric mm width."""
    if isinstance(width, str):
        mm = COL_WIDTHS_MM.get(width.lower())
        if mm is None:
            raise ValueError(f"Unknown width preset: {width}")
        return mm_to_inches(mm)
    try:
        return mm_to_inches(float(width))
    except Exception as e:
        raise ValueError("Width must be a preset string or numeric millimeters") from e


def set_size(ax: plt.Axes, width="single", ratio=0.62):
    """Set figure size based on target column width and aspect ratio."""
    w_in = _figure_width_inches(width)
    h_in = w_in * float(ratio)
    fig = ax.figure
    fig.set_size_inches(w_in, h_in)
    return fig


def format_axes(ax: plt.Axes, spines=("top", "right"), tick_params=None):
    """Hide selected spines and apply consistent tick formatting."""
    for side in spines:
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        if side in ax.spines:
            ax.spines[side].set_linewidth(0.8)
            ax.spines[side].set_color("#333333")
    if tick_params is None:
        tick_params = dict(length=3, width=0.8, pad=2)
    ax.tick_params(axis="both", which="both", **tick_params)
    return ax


def apply_small_margins(ax: plt.Axes, x=0.02, y=0.05):
    """Pad axes slightly for nicer layout."""
    try:
        ax.margins(x=x, y=y)
    except Exception:
        pass
    return ax


def savefig(fig: plt.Figure, path, dpi=None, transparent=False):
    """Safe save: creates folders, applies tight bbox and consistent dpi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"transparent": transparent}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(path, **save_kwargs)
    return path


# ==========================================================
# Context manager for applying styles
# ==========================================================
@contextmanager
def use_style(
    style_mpl="figstyle.mplstyle",
    use_tex=False,
    font_family=None,
    overrides=None,
):
    """
    Context manager to apply .mplstyle and optional rcParams overrides.

    Parameters
    ----------
    style_mpl : str
        Path to your .mplstyle file.
    use_tex : bool
        Enable LaTeX text rendering (requires system LaTeX).
    font_family : str or None
        Override font family (e.g., 'Arial', 'Helvetica').
    overrides : dict or None
        Additional rcParams to temporarily apply.
    """
    rc = {}
    if use_tex:
        rc.update({
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{siunitx}\usepackage{amsmath}"
                r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
            ),
        })
    if font_family:
        rc["font.family"] = font_family
    if overrides:
        rc.update(overrides)

    with mpl.rc_context(rc=rc):
        with mpl.style.context(str(style_mpl)):
            yield


# ==========================================================
# Convenience function: create a new, formatted figure
# ==========================================================
def new_figure(width="single", ratio=0.62):
    """Create a preformatted figure + axes pair."""
    fig, ax = plt.subplots()
    set_size(ax, width=width, ratio=ratio)
    format_axes(ax)
    return fig, ax
