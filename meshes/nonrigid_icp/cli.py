"""CLI for nonrigid ICP example / batch_opt."""

from __future__ import annotations

import argparse

from nicp_config import POSTPROCESS_DEFAULTS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one HACK raw OBJ mesh to the corresponding BNI target mesh."
    )
    parser.add_argument("--stem", default="Geng5", help="sample stem, e.g. 000063_52")
    parser.add_argument("--raw-path", default=None, help="explicit source OBJ path")
    parser.add_argument("--target-path", default=None, help="explicit target OBJ path")
    parser.add_argument("--raw-dir", default=None, help="directory containing *_phack_raw.obj")
    parser.add_argument("--target-dir", default=None, help="directory containing target .obj files")
    parser.add_argument("--bni-lmks-dir", default=None, help="directory containing *_bni_lmks.npz")
    parser.add_argument("--bni-lmks-path", default=None, help="explicit BNI landmarks npz path")
    parser.add_argument("--bni-seg-dir", default=None, help="directory containing *_bni_seg.npz")
    parser.add_argument("--bni-seg-path", default=None, help="explicit BNI segmentation npz path")
    parser.add_argument(
        "--target-face-mask",
        action="store_true",
        help="restrict target correspondence search to BNI face regions",
    )
    parser.add_argument(
        "--no-target-face-mask",
        action="store_true",
        help="disable target-side BNI face mask",
    )
    parser.add_argument("--hack-lmks478-bary-path", default=None, help="path to hack_lmks478_bary.pkl")
    parser.add_argument("--seg28-pkl-path", default=None, help="path to HACK_seg_28.pkl")
    parser.add_argument("--hack-cavity-pkl-path", default=None, help="path to HACK_cavity.pkl")
    parser.add_argument("--output-dir", default="./test_nicp", help="output directory")
    parser.add_argument("--output-path", default=None, help="explicit fitted OBJ output path")

    parser.add_argument("--quick", action="store_true", help="short smoke-test schedule")
    parser.add_argument("--quiet", action="store_true", help="suppress per-iteration logs")
    parser.add_argument("--visualize", action="store_true", help="open polyscope visualization")
    parser.add_argument("--visualize-correspondences", action="store_true", help="show correspondences")
    parser.add_argument(
        "--visualize-correspondence-limit",
        type=int,
        default=2000,
        help="max correspondence links in polyscope; <=0 shows all",
    )

    parser.add_argument("--stiffness", default=None, help="comma-separated stiffness schedule")
    parser.add_argument("--max-iterations", type=int, default=None, help="ICP iterations per stage")
    parser.add_argument("--gamma", type=float, default=None, help="translation row smoothness multiplier")
    parser.add_argument(
        "--correspondence-mode",
        choices=("vertex", "surface"),
        default=None,
        help="nearest target query backend",
    )
    parser.add_argument("--normal-angle", type=float, default=None, help="normal rejection threshold (deg)")
    parser.add_argument("--distance-threshold", type=float, default=None, help="absolute correspondence cutoff")
    parser.add_argument(
        "--distance-threshold-scale",
        type=float,
        default=None,
        help="relative distance cutoff as bbox diagonal fraction",
    )
    parser.add_argument("--trim-percentile", type=float, default=None, help="trim distances above percentile")
    parser.add_argument(
        "--solver",
        choices=("normal_cholesky", "spsolve", "lsmr"),
        default=None,
        help="linear least-squares solver",
    )
    parser.add_argument("--damping", type=float, default=None, help="normal-equation damping")
    parser.add_argument("--solver-tolerance", type=float, default=None, help="LSMR tolerance")
    parser.add_argument("--solver-max-iterations", type=int, default=None, help="LSMR max iterations")
    parser.add_argument("--affine-prior-weight", type=float, default=None)
    parser.add_argument("--affine-prior-decay", type=float, default=None)
    parser.add_argument("--position-prior-weight", type=float, default=None)
    parser.add_argument("--position-prior-decay", type=float, default=None)
    parser.add_argument("--laplacian-weight", type=float, default=None)
    parser.add_argument("--vertex-mass-weight", type=float, default=None)
    parser.add_argument("--edge-vector-weight", type=float, default=None)
    parser.add_argument("--geometry-weight-decay", type=float, default=None)
    parser.add_argument(
        "--landmark-weight",
        type=float,
        default=1.0,
        help="global landmark constraint multiplier",
    )

    # Common post-process overrides (full defaults in nicp_config.PostprocessConfig)
    parser.add_argument(
        "--no-nonface-smooth",
        action="store_true",
        help="disable post-ICP non-face seam smoothing",
    )
    parser.add_argument(
        "--no-mandatory-final-smooth",
        action="store_true",
        help="skip mandatory localized seam smooth",
    )
    parser.add_argument(
        "--no-mandatory-face-interior-smooth",
        action="store_true",
        help="skip mandatory face-interior smooth",
    )
    parser.add_argument(
        "--no-nonface-region-smooth",
        action="store_true",
        help="disable dedicated non-face Taubin pass",
    )
    parser.add_argument(
        "--no-regional-darap-normal-deform",
        action="store_true",
        help="skip post-fit regional dARAP deformation toward smoothed normals",
    )
    parser.add_argument(
        "--no-postprocess-lite-with-darap",
        action="store_true",
        help="run full post-ICP seam/bump passes even when regional dARAP is enabled",
    )
    parser.add_argument(
        "--example-fast",
        action="store_true",
        help="batch-oriented fast path: skip ICP invert repair, post-ICP non-face smooth, "
        "quality evals, and per-mesh history JSON when regional dARAP is on",
    )
    parser.add_argument(
        "--no-example-fast",
        action="store_true",
        help="disable example_fast even if enabled elsewhere (e.g. batch_opt)",
    )
    parser.add_argument(
        "--no-icp-camera-depth-attenuation",
        action="store_true",
        help="disable per-iteration down-weighting of correspondences far from the camera plane",
    )
    parser.add_argument(
        "--no-regional-darap-camera-depth-attenuation",
        action="store_true",
        help="disable camera-depth scaling of regional dARAP Procrustes lambda",
    )
    parser.set_defaults(**POSTPROCESS_DEFAULTS)
    return parser
