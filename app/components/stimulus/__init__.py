import os
import streamlit.components.v1 as components

_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
_component = components.declare_component("bci_stimulus", path=_frontend_dir)


def render_ssvep(targets: list[dict], width=900, height=450, key=None):
    """
    targets: [{"label": "SPACE", "freq_hz": 12.0, "x": 650, "y": 100, "w": 150, "h": 60}, ...]
    """
    config = {"targets": targets}
    return _component(mode="ssvep", config=config, width=width, height=height, key=key, default=None)


def render_p300(grid: list[list[str]], flash_ms=100, isi_ms=75, cell_size=80,
                 origin=(50, 50), width=900, height=550, key=None):
    config = {
        "grid": grid,
        "flash_ms": flash_ms,
        "isi_ms": isi_ms,
        "cell_size": cell_size,
        "origin": {"x": origin[0], "y": origin[1]},
    }
    return _component(mode="p300", config=config, width=width, height=height, key=key, default=None)


def render_p300_controlled(grid: list[list[str]], active_row=None, active_col=None,
                            cell_size=80, origin=(50, 50), width=900, height=550, key=None):
    config = {
        "grid": grid,
        "cell_size": cell_size,
        "origin": {"x": origin[0], "y": origin[1]},
        "active_row": active_row,
        "active_col": active_col,
    }
    return _component(mode="p300_controlled", config=config, width=width, height=height, key=key, default=None)
