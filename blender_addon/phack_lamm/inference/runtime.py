"""
LAMM inference for the Blender addon.

Mirrors ``exp_poisson/LAMM_generate_local.py``:
  - random global identity from ``gaussian_id.pickle``
  - local region manipulation via control-vertex displacements
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# blender_addon/phack_lamm/inference/runtime.py -> PHACK_code
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from networks.lamm import LAMM
from utils.torch_utils import load_from_checkpoint

# Keep local copy to avoid importing dataset.hack_seg_fids (pulls polyscope/igl).
REGION_NAMES: dict[int, str] = {
    0: "LeftFace",
    1: "RightFace",
    2: "Ears",
    3: "Neck",
    4: "Chin",
    5: "Skull",
    6: "Forehead",
    7: "Eyes",
    8: "Nose",
    9: "Mouth",
}

CONTROL_VERTICES: dict[int, list[int]] = {
    0: [2316, 4339, 2647, 2430, 2630, 4347, 4291, 3404, 2478, 3085],
    1: [84, 1079, 23, 189, 1407, 2153, 1185, 421, 1897, 757],
    2: [7266, 7394, 7400, 6348, 7245, 6858, 6798, 7119,
        8129, 7571, 8058, 7642, 7585, 7544, 7734, 7862],
    3: [10674, 10795, 11062, 10781, 9008, 8837, 9156],
    4: [7, 963, 2191, 473, 437, 2598, 4390],
    5: [10517, 9581, 9664],
    6: [1643, 8320, 8547],
    7: [3589, 2811, 4129, 2932, 2833, 3101, 3283, 3332,
        1374, 570, 612, 1022, 591, 581, 1063, 1109,
        3647, 2367, 2590, 2626, 1993, 2037, 349, 83],
    8: [977, 1307, 1348, 4840, 4857, 4777, 4556, 5196, 5312, 4849, 4853],
    9: [10, 5945, 5515, 6460, 6456, 5970, 5966, 5526, 5525,
        6132, 6225, 5627, 5720, 6155, 5621],
}


def default_paths(repo_root: Path | None = None) -> dict[str, Path]:
    """Generate with AE weights; edit with manip + residual composition.

    ``best_alpha_max.pth`` decode alone has patch seams. AE ``best.pth`` generates
    clean meshes; local edit applies ``src + (manip(δ) - manip(0))`` so seams cancel.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    ae_dir = root / "results" / "lamm_hack_ae" / "20260706_094816"
    manip_dir = root / "results" / "lamm_hack_manipulation"
    return {
        "repo_root": root,
        "config": root / "exp_poisson" / "lamm_hack_manipulation.yaml",
        "ae_config": root / "exp_poisson" / "lamm_hack.yaml",
        "ae_checkpoint": ae_dir / "best.pth",
        "ae_save_dir": ae_dir,
        "checkpoint": manip_dir / "best_alpha_max.pth",
        "save_dir": manip_dir,
        "template": root / "dataset" / "hack_template.obj",
        "region_info": root / "dataset" / "hack_region_info.pkl",
    }


def _read_obj_faces(obj_path: Path) -> np.ndarray:
    """Lightweight OBJ face reader (avoids utils.mesh → igl/polyscope)."""
    fvs: list[list[int]] = []
    with open(obj_path, encoding="utf8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            fv: list[int] = []
            for tok in line[2:].strip().split():
                fv.append(int(tok.split("/")[0]) - 1)
            if len(fv) >= 3:
                for i in range(2, len(fv)):
                    fvs.append([fv[0], fv[i - 1], fv[i]])
    return np.asarray(fvs, dtype=np.int64)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_arg}")
    return torch.device("cpu")


def sample_random_displacements(
    delta_stats: dict,
    key: int,
    control_lms: dict[int, list[int]],
    k_std: float = 1.0,
    device: str | torch.device = "cpu",
    rng: np.random.Generator | None = None,
) -> list[torch.Tensor]:
    """Sample control-vertex displacements for one region; others are zero."""
    rng = rng or np.random.default_rng()
    delta: list[torch.Tensor] = []
    for idx in control_lms.keys():
        n = 3 * len(control_lms[idx])
        if idx != key:
            delta.append(torch.zeros(n, device=device))
        else:
            sample = rng.multivariate_normal(
                delta_stats[key]["mean"],
                k_std * delta_stats[key]["std"],
                size=1,
            )
            delta.append(torch.tensor(sample.reshape(-1), dtype=torch.float32, device=device))
    return delta


def displacements_from_region_offsets(
    control_lms: dict[int, list[int]],
    offsets: dict[int, np.ndarray],
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """
    Build per-region delta tensors from disk translations.

    ``offsets[region_id]`` is a (3,) translation applied uniformly to all
    control vertices of that region (LAMM disk-style local edit).
    """
    delta: list[torch.Tensor] = []
    for idx in control_lms.keys():
        n_v = len(control_lms[idx])
        if idx not in offsets:
            delta.append(torch.zeros(3 * n_v, device=device))
            continue
        off = np.asarray(offsets[idx], dtype=np.float32).reshape(-1)
        if off.size == 3:
            flat = np.tile(off, n_v)
        elif off.size == 3 * n_v:
            flat = off
        else:
            raise ValueError(
                f"region {idx}: offset size {off.size} "
                f"expected 3 or {3 * n_v}"
            )
        delta.append(torch.tensor(flat, dtype=torch.float32, device=device))
    return delta


class LammRuntime:
    """Lazy-loaded LAMM AE + manipulation models for Blender / CLI."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        ae_checkpoint: str | Path | None = None,
        device: str = "0",
        config_file: str | Path | None = None,
    ):
        paths = default_paths(Path(repo_root) if repo_root else None)
        self.repo_root = paths["repo_root"]
        self.config_file = Path(config_file) if config_file else paths["config"]
        # Generate uses AE weights; local edit uses manipulation weights.
        self.ae_checkpoint = Path(ae_checkpoint) if ae_checkpoint else paths["ae_checkpoint"]
        self.checkpoint = Path(checkpoint) if checkpoint else paths["checkpoint"]
        self.ae_save_dir = self.ae_checkpoint.parent
        self.save_dir = paths["save_dir"]
        self.device = _resolve_device(device)
        self.control_lms = dict(CONTROL_VERTICES)

        config = read_yaml(self.config_file)
        model_cfg = dict(config["MODEL"])
        model_cfg["region_info_file"] = str(paths["region_info"])
        model_cfg["reference_obj"] = str(paths["template"])
        model_cfg["control_vertices"] = {
            str(k): v for k, v in self.control_lms.items()
        }
        self.model_cfg = model_cfg

        self.faces = _read_obj_faces(paths["template"])

        # Gaussian for sampling identities: prefer AE run dir (matches ae_checkpoint).
        gaussian_path = self.ae_save_dir / "gaussian_id.pickle"
        if not gaussian_path.is_file():
            gaussian_path = self.save_dir / "gaussian_id.pickle"
        with open(gaussian_path, "rb") as f:
            self.gaussian_id = pickle.load(f, encoding="latin1")

        delta_path = self.save_dir / "displacement_stats.pickle"
        with open(delta_path, "rb") as f:
            self.delta_stats = pickle.load(f)

        self._ae: LAMM | None = None
        self._manip: LAMM | None = None
        self._source_verts: np.ndarray | None = None

    @property
    def source_verts(self) -> np.ndarray | None:
        return self._source_verts

    def _ensure_ae(self) -> LAMM:
        if self._ae is None:
            cfg = dict(self.model_cfg)
            cfg["manipulation"] = False
            net = LAMM(cfg).to(self.device)
            load_from_checkpoint(
                net, str(self.ae_checkpoint), partial_restore=True, device=str(self.device),
            )
            net.eval()
            self._ae = net
        return self._ae

    def _ensure_manip(self) -> LAMM:
        if self._manip is None:
            cfg = dict(self.model_cfg)
            cfg["manipulation"] = True
            net = LAMM(cfg).to(self.device)
            load_from_checkpoint(
                net, str(self.checkpoint), partial_restore=True, device=str(self.device),
            )
            net.eval()
            self._manip = net
        return self._manip

    def control_centers(self, verts: np.ndarray) -> dict[int, np.ndarray]:
        """Mean of control vertices per region (disk rest positions)."""
        centers: dict[int, np.ndarray] = {}
        for key, vids in self.control_lms.items():
            centers[key] = np.asarray(verts[vids], dtype=np.float64).mean(axis=0)
        return centers

    def control_radii(self, verts: np.ndarray, min_radius: float = 0.008) -> dict[int, float]:
        """Disk radius from control-vertex spread, clamped to face scale."""
        bbox = np.asarray(verts, dtype=np.float64).max(axis=0) - np.asarray(verts, dtype=np.float64).min(axis=0)
        face_scale = float(np.linalg.norm(bbox))
        max_r = max(face_scale * 0.055, min_radius)
        radii: dict[int, float] = {}
        centers = self.control_centers(verts)
        for key, vids in self.control_lms.items():
            pts = np.asarray(verts[vids], dtype=np.float64)
            c = centers[key]
            if len(pts) == 0:
                radii[key] = min_radius
                continue
            dists = np.linalg.norm(pts - c, axis=1)
            r = float(np.median(dists) * 1.25) if len(dists) else min_radius
            radii[key] = float(np.clip(r, min_radius, max_r))
        return radii

    @torch.inference_mode()
    def generate_random(
        self,
        *,
        k_std: float = 1.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random HACK identity. Returns (verts [V,3], faces [F,3])."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        ae = self._ensure_ae()
        mu = self.gaussian_id["mean"]
        sigma = self.gaussian_id["sigma"]
        z = torch.tensor(
            np.random.multivariate_normal(mu, k_std * sigma, 1),
            dtype=torch.float32,
            device=self.device,
        )
        verts = ae.decode(z.unsqueeze(1))[-1][0].detach().cpu().numpy().astype(np.float64)
        self._source_verts = verts.copy()
        return verts, self.faces.copy()

    def set_source(self, verts: np.ndarray) -> None:
        self._source_verts = np.asarray(verts, dtype=np.float64).copy()

    @torch.inference_mode()
    def manipulate(
        self,
        *,
        region_key: int | None = None,
        offsets: dict[int, Any] | None = None,
        random_region: bool = False,
        k_std: float = 1.0,
        seed: int | None = None,
        source_verts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Apply local manipulation to the current source identity.

        Provide either:
          - ``random_region=True`` + ``region_key``: sample from displacement_stats
          - ``offsets``: {region_id: (3,) translation} from disk drags

        Returns (verts, faces, info) where info may include ``applied_offsets``.
        """
        src = source_verts if source_verts is not None else self._source_verts
        if src is None:
            raise RuntimeError("no source mesh; call generate_random() first")

        if seed is not None:
            np.random.seed(seed)

        manip = self._ensure_manip()
        x = torch.tensor(src, dtype=torch.float32, device=self.device).unsqueeze(0)
        info: dict[str, Any] = {}

        if random_region:
            if region_key is None:
                raise ValueError("region_key required for random_region")
            delta = sample_random_displacements(
                self.delta_stats, int(region_key), self.control_lms, k_std, self.device,
            )
            flat = delta[list(self.control_lms.keys()).index(int(region_key))]
            arr = flat.detach().cpu().numpy().reshape(-1, 3)
            info["applied_offsets"] = {int(region_key): arr.mean(axis=0).tolist()}
        elif offsets is not None:
            clean = {
                int(k): np.asarray(v, dtype=np.float32)
                for k, v in offsets.items()
                if v is not None and np.linalg.norm(np.asarray(v).reshape(-1)[:3]) > 1e-12
            }
            delta = displacements_from_region_offsets(self.control_lms, clean, self.device)
            info["applied_offsets"] = {
                int(k): np.asarray(v, dtype=np.float32).reshape(-1)[:3].tolist()
                for k, v in clean.items()
            }
        else:
            raise ValueError("provide offsets=... or random_region=True")

        delta_b = []
        for d in delta:
            if d.ndim == 1:
                delta_b.append(d.unsqueeze(0))
            else:
                delta_b.append(d)

        zeros = [
            torch.zeros(1, size, device=self.device)
            for size in manip.control_region_sizes
        ]
        out_delta = manip((x, delta_b))[-1][0].detach().cpu().numpy().astype(np.float64)
        out_zero = manip((x, zeros))[-1][0].detach().cpu().numpy().astype(np.float64)
        # Residual: keep AE source topology, add only the manip-induced change.
        out = np.asarray(src, dtype=np.float64) + (out_delta - out_zero)
        info["residual"] = True
        return out, self.faces.copy(), info

    def region_meta(self, verts: np.ndarray | None = None) -> list[dict[str, Any]]:
        """UI metadata for the 10 patch disks."""
        v = verts if verts is not None else self._source_verts
        if v is None:
            raise RuntimeError("no verts for region metadata")
        centers = self.control_centers(v)
        radii = self.control_radii(v)
        meta = []
        for key in sorted(self.control_lms.keys()):
            meta.append({
                "key": key,
                "name": REGION_NAMES[key],
                "center": centers[key].tolist(),
                "radius": float(radii[key]),
                "n_controls": len(self.control_lms[key]),
            })
        return meta


def _save_npz(path: Path, verts: np.ndarray, faces: np.ndarray, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"verts": verts, "faces": faces}
    if meta is not None:
        kwargs["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(str(path), **kwargs)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="PHACK LAMM Blender inference CLI")
    parser.add_argument("--mode", choices=["generate", "manipulate", "meta"], required=True)
    parser.add_argument("--repo_root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Manipulation checkpoint (local edit).")
    parser.add_argument("--ae_checkpoint", type=Path, default=None,
                        help="AE checkpoint (random generate).")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--k_std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--region", type=int, default=None)
    parser.add_argument("--random_region", action="store_true")
    parser.add_argument("--offsets_json", type=str, default=None,
                        help='JSON dict of region offsets, e.g. {"8":[0,0.01,0]}')
    parser.add_argument("--source_npz", type=Path, default=None)
    parser.add_argument("--out_npz", type=Path, required=True)
    args = parser.parse_args()

    rt = LammRuntime(
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        ae_checkpoint=args.ae_checkpoint,
        device=args.device,
    )

    if args.mode == "generate":
        verts, faces = rt.generate_random(k_std=args.k_std, seed=args.seed)
        meta = {"regions": rt.region_meta(verts), "mode": "generate"}
        _save_npz(args.out_npz, verts, faces, meta)
        print(json.dumps({"ok": True, "verts": int(verts.shape[0]), "out": str(args.out_npz)}))
        return

    if args.source_npz is None or not args.source_npz.is_file():
        raise SystemExit("--source_npz required for manipulate/meta")

    data = np.load(str(args.source_npz), allow_pickle=True)
    src = np.asarray(data["verts"], dtype=np.float64)
    rt.set_source(src)

    if args.mode == "meta":
        meta = {"regions": rt.region_meta(src), "mode": "meta"}
        _save_npz(args.out_npz, src, np.asarray(data["faces"]), meta)
        print(json.dumps({"ok": True, "regions": len(meta["regions"])}))
        return

    offsets = None
    if args.offsets_json:
        raw = json.loads(args.offsets_json)
        offsets = {int(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}

    verts, faces, info = rt.manipulate(
        region_key=args.region,
        offsets=offsets,
        random_region=args.random_region,
        k_std=args.k_std,
        seed=args.seed,
        source_verts=src,
    )
    meta = {
        "regions": rt.region_meta(src),
        "mode": "manipulate",
        "region": args.region,
        **info,
    }
    _save_npz(args.out_npz, verts, faces, meta)
    print(json.dumps({"ok": True, "verts": int(verts.shape[0]), "out": str(args.out_npz)}))


if __name__ == "__main__":
    _cli()
