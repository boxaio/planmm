"""LAMM control vertex indices on hack_template (global vids), keyed by patch id 0..9."""

from __future__ import annotations

# Per-region control vertex indices (shared with demo_hack_control_verts.py).
control_verts_ids: dict[str, list[int]] = {
    '0': [2316, 4339, 2647, 2430, 2630, 4347, 4291, 3404, 2478, 3085],
    '1': [84, 1079, 23, 189, 1407, 2153, 1185, 421, 1897, 757],
    '2': [7266, 7394, 7400, 6348, 7245, 6858, 6798, 7119,
          8129, 7571, 8058, 7642, 7585, 7544, 7734, 7862],
    '3': [10674, 10795, 11062, 10781, 9008, 8837, 9156],
    '4': [7, 963, 2191, 473, 437, 2598, 4390],
    '5': [10517, 9581, 9664],
    '6': [1643, 8320, 8547],
    '7': [3589, 2811, 4129, 2932, 2833, 3101, 3283, 3332,
          1374, 570, 612, 1022, 591, 581, 1063, 1109,
          3647, 2367, 2590, 2626, 1993, 2037, 349, 83],
    '8': [977, 1307, 1348, 4840, 4857, 4777, 4556, 5196, 5312, 4849, 4853],
    '9': [10, 5945, 5515, 6460, 6456, 5970, 5966, 5526, 5525, 6132, 6225, 5627, 5720, 6155, 5621],
}


def default_control_vertices() -> dict[int, list[int]]:
    """Return LAMM-style control_vertices dict (int region id -> vids)."""
    return {int(k): [int(v) for v in vals] for k, vals in control_verts_ids.items()}


def resolve_control_vertices(model_cfg) -> dict[int, list[int]]:
    """Use MODEL.control_vertices if set, else hack defaults."""
    cfg = model_cfg.config if hasattr(model_cfg, 'config') else model_cfg
    cv = cfg.get('control_vertices') if hasattr(cfg, 'get') else {}
    if not cv and hasattr(model_cfg, '__getitem__'):
        try:
            cv = model_cfg['control_vertices']
        except (KeyError, TypeError):
            cv = {}
    if cv:
        return {int(k): [int(v) for v in vals] for k, vals in sorted(cv.items())}
    return default_control_vertices()
