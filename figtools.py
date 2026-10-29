"""
figtools.py — helper utilities for unified figure styling in the thesis

Core ideas
----------
- Centralise physical figure widths (thesis, single column, etc.)
- Always go through `use_style(...)` so fonts / ticks / layout come from
  `figstyle.mplstyle`.
- Use `set_size(...)`, `new_figure(...)` and `new_subplots(...)` so that
  all figures share the same physical dimensions and appearance.
- Define a fixed colour map for therapy strategies (NT, CT, AT50, ...).

Typical usage
-------------
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from figtools import use_style, new_figure, savefig, therapy_color

    OUTDIR = Path("output/figures")
    STYLE  = "figstyle.mplstyle"

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    with use_style(STYLE):
        fig, ax = new_figure(width="thesis", ratio=0.62)
        ax.plot(x, y, color=therapy_color("AT50"), label="AT50")
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Tumor burden [a.u.]")
        ax.legend()

        savefig(fig, OUTDIR / "example.svg", dpi=300, transparent=True)
"""

from __future__ import annotations

from pathlib import Path
from data_paths import DATA_ROOT
from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Union, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# Width presets (in mm) for different figure types
# ============================================================

# These are the physical widths that correspond to your thesis layout.
# Adjust them if the FAU template uses different values.
COL_WIDTHS_MM: Dict[str, float] = {
    "thesis": 160.0,   # full text width (~16 cm)
    "single": 85.0,    # single column / half page
    "onehalf": 114.0,  # 1.5 column
    "double": 174.0,   # full double column width
}


def mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches."""
    return float(mm) / 25.4


def _figure_width_inches(width: Union[str, float, int]) -> float:
    """
    Resolve a width preset or a numeric width in mm to inches.

    Parameters
    ----------
    width : {'thesis', 'single', 'onehalf', 'double'} or float
        - If str: looked up in COL_WIDTHS_MM.
        - If numeric: interpreted as millimetres.

    Returns
    -------
    float
        Width in inches.
    """
    if isinstance(width, str):
        if width not in COL_WIDTHS_MM:
            raise KeyError(
                f"Unknown width preset '{width}'. "
                f"Available: {list(COL_WIDTHS_MM.keys())}"
            )
        mm = COL_WIDTHS_MM[width]
    else:
        mm = float(width)
    return mm_to_inches(mm)


# ============================================================
# Colour palette and therapy-strategy colours
# ============================================================

PALETTE: Dict[str, str] = {
    # Core
    "primary": "#1F77B4",    # blue
    "secondary": "#FF7F0E",  # orange
    "tertiary": "#2CA02C",   # green

    # Neutrals
    "black": "#000000",
    "grey": "#7F7F7F",

    # Extra categorical colours if needed
    "cat1": "#1B9E77",
    "cat2": "#D95F02",
    "cat3": "#7570B3",
    "cat4": "#E7298A",
}

# Fixed colours for population-level traces used across figures.
POPULATION_COLORS: Dict[str, str] = {
    "sensitive": "#4169E1",
    "resistant": "#DAA520",
    "total": "#000000",
}

# Neutral colours for state comparisons that are not treatment arms.
NEUTRAL_STATE_COLORS: Dict[str, str] = {
    "inactive": "#9E9E9E",
    "active": "#333333",
}

# Fixed colours for therapy strategies, used across all figures.
# You can tweak these later but keep the mapping stable everywhere.
THERAPY_COLORS = {
    "NT":        "#332288",  # no treatment
    "CT":        "#117733",  # continuous / max dose
    "IT":        "#9467BD",  # muted purple, immune treatment
    "Pulse":     "#AA4499",  # pulse schedule

    # Adaptive therapy variants (optimized for contrast)
    "AT50":      "#88CCEE",  # light blue
    "AT100":     "#CC6677",  # rose
    "AT80":      "#2CA02C",  # green (rarely shown)
    "AT100+":    "#44AA99",  # teal (rarely shown)
    "AT":        "#278c27",  # overall AT green
    "eAT100":    "#37c837",  # eAT100 green

    # Agent-based strategies
    "LV-Agent":  "#882255",  # wine
    "KPP-Agent": "#8C564B",  # brown
}


def therapy_color(name: str) -> str:
    """
    Return the canonical colour for a given therapy strategy.

    Parameters
    ----------
    name : str
        One of: 'NT', 'CT', 'IT', 'AT50', 'AT80', 'AT100', 'AT100+',
        'LV-Agent', 'KPP-Agent'.

    Returns
    -------
    hex_colour : str

    Raises
    ------
    KeyError
        If an unknown strategy name is passed.
    """
    try:
        return THERAPY_COLORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown therapy strategy '{name}'. "
            f"Known: {list(THERAPY_COLORS.keys())}"
        ) from exc


def population_color(name: str) -> str:
    """Return the canonical colour for a population-level trace."""
    key = str(name).strip().lower()
    try:
        return POPULATION_COLORS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown population colour '{name}'. "
            f"Known: {list(POPULATION_COLORS.keys())}"
        ) from exc


def neutral_state_color(name: str) -> str:
    """Return the canonical neutral colour for non-therapy state comparisons."""
    key = str(name).strip().lower()
    try:
        return NEUTRAL_STATE_COLORS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown neutral state colour '{name}'. "
            f"Known: {list(NEUTRAL_STATE_COLORS.keys())}"
        ) from exc


MARKERS: Sequence[str] = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
LINESTYLES: Sequence[str] = ["-", "--", "-.", ":"]

# Thesis-wide visual constants not covered by Matplotlib rcParams.  Keep
# explicit, local overrides to these values rare so figures remain coherent.
PANEL_LABEL_FONTSIZE = 10.0
PANEL_LABEL_WEIGHT = "bold"
ANNOTATION_FONTSIZE = 7.5
ERRORBAR_LINEWIDTH = 1.0
ERRORBAR_CAPSIZE = 3.0
SECONDARY_LINEWIDTH = 0.9
GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.35


# ============================================================
# Layout helpers
# ============================================================

def set_size(
    ax: plt.Axes,
    width: Union[str, float, int] = "thesis",
    ratio: float = 0.62,
) -> mpl.figure.Figure:
    """
    Set the physical size of the figure that an Axes belongs to.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        An axes in the target figure.
    width : str or float
        Width preset ('thesis', 'single', 'double', ...) or numeric mm.
    ratio : float
        Height / width ratio.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resized figure.
    """
    w_in = _figure_width_inches(width)
    h_in = w_in * float(ratio)
    fig = ax.figure
    fig.set_size_inches(w_in, h_in)
    return fig


def format_axes(
    ax: plt.Axes,
    spines: Iterable[str] = ("top", "right"),
    tick_params: Optional[dict] = None,
) -> None:
    """
    Apply unified styling to axes: hide selected spines, set tick appearance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    spines : iterable of {'top', 'right', 'left', 'bottom'}
        Spines to hide (defaults to removing top/right).
    tick_params : dict, optional
        Overrides for ax.tick_params.
    """
    # Hide selected spines
    for spine in spines:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)

    # Ensure remaining spines use the active style's colour / linewidth.
    for loc, spine in ax.spines.items():
        if spine.get_visible():
            spine.set_linewidth(mpl.rcParams["axes.linewidth"])
            spine.set_color(mpl.rcParams["axes.edgecolor"])

    # Default tick settings (can be overridden)
    default_tick_params = dict(
        direction="out",
        length=mpl.rcParams["xtick.major.size"],
        width=mpl.rcParams["xtick.major.width"],
        pad=2,
        top=False,
        right=False,
    )
    if tick_params is not None:
        default_tick_params.update(tick_params)

    ax.tick_params(axis="both", **default_tick_params)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.10,
    y: float = 1.05,
) -> None:
    """Add a standard bold panel label, e.g. ``a`` or ``b``."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight=PANEL_LABEL_WEIGHT,
        va="top",
        ha="right",
    )


def apply_small_margins(ax: plt.Axes, x: float = 0.02, y: float = 0.05) -> None:
    """
    Add small relative margins to axes (for nicer layout in tight figures)."""
    ax.margins(x=x, y=y)


# ============================================================
# Style context
# ============================================================

@contextmanager
def use_style(
    style_mpl: Optional[Union[str, Path]] = "figstyle.mplstyle",
    use_tex: bool = False,
    font_family: Optional[str] = None,
    overrides: Optional[Dict[str, object]] = None,
):
    """
    Context manager applying a Matplotlib style + rcParams for the duration.

    Parameters
    ----------
    style_mpl : str or Path or None
        Path/name of a .mplstyle file to use (e.g. 'figstyle.mplstyle').
        If None, no style file is applied and only overrides are used.
    use_tex : bool
        If True, enable LaTeX rendering via text.usetex (slower!).
    font_family : str, optional
        Override the font family (e.g. 'Arial') inside this context.
    overrides : dict, optional
        Additional rcParams to update inside the context.
    """
    overrides = dict(overrides or {})

    if use_tex:
        overrides.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )

    if font_family is not None:
        overrides["font.family"] = font_family

    # Use rc_context so everything reverts after the with-block
    with mpl.rc_context():
        # Apply style file first
        if style_mpl is not None:
            if str(style_mpl) == "figstyle.mplstyle":
                style_mpl = DATA_ROOT / "config" / "figstyle.mplstyle"
            mpl.style.use(str(style_mpl))

        # Then apply overrides on top
        if overrides:
            mpl.rcParams.update(overrides)

        yield


# ============================================================
# Figure / subplot constructors
# ============================================================

def new_figure(
    width: Union[str, float, int] = "thesis",
    ratio: float = 0.62,
) -> Tuple[mpl.figure.Figure, plt.Axes]:
    """
    Create a new figure + single axes with standard thesis sizing and axes style.

    Parameters
    ----------
    width : str or float
        Width preset ('thesis', 'single', 'double', ...) or numeric mm.
    ratio : float
        Height / width ratio.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots()
    set_size(ax, width=width, ratio=ratio)
    format_axes(ax)
    return fig, ax


def new_subplots(
    nrows: int = 1,
    ncols: int = 1,
    width: Union[str, float, int] = "thesis",
    ratio: float = 0.62,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    **subplot_kwargs,
) -> Tuple[mpl.figure.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """
    Create a multi-panel figure with unified sizing and axis formatting.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions.
    width : str or float
        Width preset ('thesis', 'single', 'double', ...) or numeric mm.
    ratio : float
        Height / width ratio of the full figure.
    sharex, sharey : bool
        Standard plt.subplots behaviour.
    squeeze : bool
        Standard plt.subplots behaviour.
    subplot_kwargs :
        Passed through to plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : Axes or array-like of Axes
    """
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        **subplot_kwargs,
    )

    # Set the size of the figure based on the first axes
    if isinstance(axes, plt.Axes):
        ref_ax = axes
        all_axes = [axes]
    else:
        # Flatten possible numpy array of axes
        all_axes = [ax for ax in (axes.ravel() if hasattr(axes, "ravel") else axes)]
        ref_ax = all_axes[0]

    set_size(ref_ax, width=width, ratio=ratio)

    # Apply unified axis formatting to all panels
    for ax in all_axes:
        format_axes(ax)

    return fig, axes


# ============================================================
# Saving utility
# ============================================================

def savefig(
    fig: mpl.figure.Figure,
    path: Union[str, Path],
    dpi: Optional[int] = None,
    transparent: Optional[bool] = None,
    **savefig_kwargs,
) -> None:
    """
    Save a figure with consistent settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Output path (extension determines format).
    dpi : int, optional
        DPI override for export (defaults to rcParams['savefig.dpi']).
    transparent : bool, optional
        If None, use rcParams['savefig.transparent'].
    savefig_kwargs :
        Additional keyword arguments forwarded to fig.savefig(). Pass
        ``bbox_inches=None`` to retain the exact physical canvas dimensions.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = mpl.rcParams.get("savefig.dpi", 300)

    if transparent is None:
        transparent = mpl.rcParams.get("savefig.transparent", True)

    # Matplotlib resolves ``bbox_inches=None`` back to the global rcParam.
    # Treat an explicitly supplied None as a request for the uncropped,
    # physical figure canvas instead.
    _missing = object()
    bbox_inches = savefig_kwargs.pop("bbox_inches", _missing)
    if bbox_inches is _missing:
        bbox_inches = mpl.rcParams.get("savefig.bbox", "tight")
    elif bbox_inches is None:
        bbox_inches = fig.bbox_inches

    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=mpl.rcParams.get("savefig.pad_inches", 0.02),
        transparent=transparent,
        **savefig_kwargs,
    )
