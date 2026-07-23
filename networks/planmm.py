"""PLANMM: LAMM autoencoder with PoissonBlock patch tokenization.

Architecture is identical to ``networks.lamm.LAMM`` (MLPMixer encoder/decoder,
per-layer mesh outputs, multilayer L1 training), with two Poisson options:

1. ``xyz_to_token`` → shared PoissonBlock stack + region mass-pool (always on).
2. ``mesh_readout``: ``linear`` | ``njf`` | ``poisson`` | ``hybrid``
   ``hybrid`` (default ``hybrid_final='poisson_lp'``):
     mid-layers = LAMM ``token_to_xyz`` (hard region merge);
     final = **one global Poisson-integrated mesh** whose Grad residual is
     anchored at ``G@lowpass(Linear)`` (coarse pose base), with full Linear
     only as Grad conditioning + COM — never added in R³.
     ``poisson`` = Grad base at template (ablation; under-deforms).
     ``residual`` = ``scaffold + detail`` (ablation; inherits Linear HF).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import ConfigParams
from meshes.operators import construct_mesh_operators
from networks.module_mlpmixer import MLPMixerPerLayerOut
from networks.PoissonNet import NJFHead, PoissonBlock, bmm_G
from networks.base import vertices_to_faces
from utils.mesh import read_obj


HACK = {
    'verts': 14062,
    'faces': 28068,
}


def _mesh_connected_components(n_verts: int, faces: torch.Tensor) -> list[torch.Tensor]:
    """Union-find CCs over local vertex indices; return member index tensors."""
    parent = list(range(n_verts))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for tri in faces.detach().reshape(-1, 3).tolist():
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    touched = set()
    for tri in faces.detach().reshape(-1, 3).tolist():
        touched.update(int(x) for x in tri)

    buckets: dict[int, list[int]] = {}
    for i in touched:
        buckets.setdefault(find(i), []).append(i)
    return [
        torch.tensor(sorted(members), dtype=torch.long)
        for members in buckets.values()
        if members
    ]


def resolve_control_vertices(config):
    """Resolve control vertices from config or demo defaults."""
    cfg = config.config if hasattr(config, 'config') else config
    cv = cfg.get('control_vertices') or {}
    if cv:
        return {int(k): [int(v) for v in vals] for k, vals in sorted(cv.items())}
    from exp_poisson.hack_control_vertices import default_control_vertices

    return default_control_vertices()


def build_layer_blend_weights(
    encoder_depth: int,
    decoder_depth: int,
    *,
    device: torch.device | None = None,
):
    """Per-layer GT blend λ in [0, 1]: encoder 1→0, decoder 0→1."""
    n_layers = encoder_depth + decoder_depth + 2
    w = torch.cat((
        torch.linspace(1, 0, encoder_depth + 1),
        torch.linspace(0, 1, decoder_depth + 1),
    ))
    if device is not None:
        w = w.to(device)
    return w.view(n_layers, 1, 1, 1)


def build_layer_loss_weights(
    encoder_depth: int,
    decoder_depth: int,
    *,
    weights: list[float] | None = None,
    device: torch.device | None = None,
):
    """Per-layer loss multipliers (separate from blend λ); defaults to ones."""
    n_layers = encoder_depth + decoder_depth + 2
    if weights is not None:
        if len(weights) != n_layers:
            raise ValueError(f'layer_loss weights length {len(weights)} != n_layers {n_layers}')
        w = torch.tensor(weights, dtype=torch.float32)
    else:
        w = torch.ones(n_layers, dtype=torch.float32)
    if device is not None:
        w = w.to(device)
    return w.view(n_layers, 1, 1, 1)


def mass_weighted_pool(feat: torch.Tensor, vertex_mass: torch.Tensor) -> torch.Tensor:
    """Pool vertex features to one vector: [B, V, C] + [B, V] → [B, C]."""
    mass = vertex_mass.unsqueeze(-1)
    return (feat * mass).sum(dim=1) / mass.sum(dim=1).clamp_min(1e-8)


class PoissonXYZToToken(nn.Module):
    """PoissonBlocks on mesh → region patch tokens.

    Shared Poisson stack over all vertices, then mass-weighted pool per region
    and a Linear projection to ``dim``.
    """

    def __init__(self, dim: int, C_width: int, n_blocks: int, region_vert_ids: list[torch.Tensor], config: dict | None = None):
        super().__init__()
        config = config or {}
        self.region_vert_ids = list(region_vert_ids)
        self.proj_in = nn.Linear(3, C_width)
        self.blocks = nn.ModuleList([
            PoissonBlock(
                in_c=C_width,
                out_c=C_width,
                width=C_width,
                extra_feats=0,
                config=config,
            )
            for _ in range(n_blocks)
        ])
        self.patch_proj = nn.Linear(C_width, dim)
        self.gradient_checkpoint = bool(config.get('gradient_checkpoint', False))

    def forward(self, x: torch.Tensor, M, G, solver, faces, vertex_mass):
        """Args:
            x: [B, V, 3] input mesh vertices.

        Returns:
            [B, K, dim] patch tokens (K = number of regions).
        """
        feat = self.proj_in(x)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                def _block_fwd(f, blk, m, g, sol, fc, vm):
                    out, _ = blk(f, m, g, sol, fc, vm, extra_features=None)
                    return out

                feat = checkpoint(
                    _block_fwd, feat, block, M, G, solver, faces, vertex_mass,
                    use_reentrant=False,
                )
            else:
                feat, _ = block(
                    feat, M, G, solver, faces, vertex_mass, extra_features=None,
                )

        tokens = []
        device = feat.device
        for vids in self.region_vert_ids:
            vids = vids.to(device)
            region_feat = feat[:, vids]
            region_mass = vertex_mass[:, vids]
            pooled = mass_weighted_pool(region_feat, region_mass)
            tokens.append(self.patch_proj(pooled))
        return torch.stack(tokens, dim=1)


class SmoothMeshReadout(nn.Module):
    """Soft-scatter patch tokens → (optional PoissonBlocks) → NJF field.

    Call modes:
      - Absolute (``njf`` / ``poisson`` / hybrid ``poisson``): mass-centered
        displacement; caller places it (template and/or Linear COM).
      - Additive scaffold residual (``scaffold=...``, ablation): Grad head
        anchored at ``G@scaffold``; returns mass-centered detail for
        ``mesh = scaffold + detail``. Algebraically a translate of the
        Poisson field, but grads inherit Linear's integrable high-frequency.
    """

    def __init__(
        self,
        dim: int,
        C_width: int,
        n_blocks: int,
        use_poisson_blocks: bool,
        poisson_cfg: dict | None = None,
        region_face_ids: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        poisson_cfg = dict(poisson_cfg or {})
        self.use_poisson_blocks = bool(use_poisson_blocks)
        self.grad_residual = bool(poisson_cfg.get('grad_residual', False))
        self.regional_grad = bool(poisson_cfg.get('regional_grad', False))
        self.proj_in = nn.Linear(dim, C_width)
        self.blocks = nn.ModuleList()
        if self.use_poisson_blocks:
            self.blocks = nn.ModuleList([
                PoissonBlock(
                    in_c=C_width,
                    out_c=C_width,
                    width=C_width,
                    extra_feats=0,
                    config=poisson_cfg,
                )
                for _ in range(n_blocks)
            ])

        self.use_r3_njf = self.grad_residual or self.regional_grad
        self.to_r3: nn.Linear | None = None
        if self.use_r3_njf:
            self.to_r3 = nn.Linear(C_width, 3)
            njf_in = 3
        else:
            njf_in = C_width
        self.njf_head = NJFHead(
            in_c=njf_in,
            out_c=3,
            width=C_width,
            config=poisson_cfg,
            region_face_ids=region_face_ids if self.regional_grad else None,
        )
        self.gradient_checkpoint = bool(poisson_cfg.get('gradient_checkpoint', False))

    def set_region_face_ids(self, region_face_ids: list[torch.Tensor]):
        self.njf_head.set_region_face_ids(region_face_ids)

    def _run_blocks(self, feat: torch.Tensor, M, G, solver, faces, vertex_mass):
        x = self.proj_in(feat)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                def _block_fwd(f, blk, m, g, sol, fc, vm):
                    out, _ = blk(f, m, g, sol, fc, vm, extra_features=None)
                    return out

                x = checkpoint(
                    _block_fwd, x, block, M, G, solver, faces, vertex_mass,
                    use_reentrant=False,
                )
            else:
                x, _ = block(x, M, G, solver, faces, vertex_mass, extra_features=None)
        return x

    def forward(self, feat: torch.Tensor, M, G, solver, faces, vertex_mass, scaffold=None):
        """Args:
            feat: [B, V, dim] soft-scattered patch features.
            scaffold: optional [B, V, 3] Linear coarse shape. When set, returns
                mass-centered integrable detail for ``scaffold + detail``.

        Returns:
            [B, V, 3] displacement (mass-centered absolute, or scaffold detail).
        """
        x = self._run_blocks(feat, M, G, solver, faces, vertex_mass)

        if scaffold is not None:
            # Strategy 2: NJF residual around Linear coarse pose.
            if self.to_r3 is None:
                raise RuntimeError(
                    'scaffold residual requires grad_residual or regional_grad '
                    '(to_r3 for token conditioning)'
                )
            cond = vertices_to_faces(self.to_r3(x), faces)
            detail, _ = self.njf_head(
                scaffold, M, G, solver, faces, vertex_mass,
                cond_faces=cond, return_detail=True,
            )
            return detail

        if self.to_r3 is not None:
            x = self.to_r3(x)
        disp, _ = self.njf_head(x, M, G, solver, faces, vertex_mass)
        return disp


class PLANMM(nn.Module):

    default_cfg = {
        'scale_dim': 4,
        'scale_dim_token': 0.5,
        'dropout': 0,
        'bottleneck_dim': 256,
        'depth': 3,
        'encoder_depth': 3,
        'decoder_depth': 3,
        'Dinput': 3,
        'Dlms': 8,
        'heads': 8,
        'Npatches': 10,
        'dim': 512,
        'region_info_file': _REPO_ROOT / 'dataset' / 'hack_region_info.pkl',
        'reference_obj': _REPO_ROOT / 'dataset' / 'hack_template.obj',
        'center_template': True,
        'manipulation': False,
        'control_vertices': {},
        # Poisson tokenizer
        'C_width': 128,
        'poisson_token_blocks': 2,
        'operator_high_precision': False,
        # Mesh readout: linear | njf | poisson | hybrid
        # hybrid: mid = Linear token_to_xyz; final depends on hybrid_final
        'mesh_readout': 'linear',
        'mesh_readout_blocks': 1,
        'region_soft_edge_mult': 2.0,
        # poisson_lp (default) = Poisson mesh; Grad base = lowpass(Linear)
        # poisson = Grad base at template (ablation; under-deforms)
        # residual / additive = soft Linear + global NJF detail (ablation)
        # regional_residual = Linear + per-CC NJF (ablation; can bump interiors)
        # regional = per-CC template+NJF; njf = global template+NJF
        'hybrid_final': 'poisson_lp',
        'scaffold_lowpass_iters': 4,
        'scaffold_lowpass_blend': 0.5,
        'poisson_cfg': {
            'cmlp_nlayers': 2,
            'cmlp_modulate': True,
            'mass_norm': True,
            'inner_prod_features': False,
            'drop_path': 0.0,
            'dropout_p': 0.0,
            'mlp_norm': False,
            'gradient_checkpoint': False,
            # NJF Grad head: residual + shared ComplexMLP + region embedding
            'grad_residual': True,
            'regional_grad': True,
        },
    }

    def __init__(self, config: dict):
        super().__init__()

        self.config = ConfigParams(config, self.default_cfg)
        self.Npatches = self.config['Npatches']
        self.dim = self.config['dim']
        self.depth = self.config['depth']
        self.encoder_depth = self.config['encoder_depth']
        self.decoder_depth = self.config['decoder_depth']
        if self.encoder_depth is None:
            self.encoder_depth = self.depth
        if self.decoder_depth is None:
            self.decoder_depth = self.depth

        self.heads = self.config['heads']
        self.scale_dim = self.config['scale_dim']
        self.scale_dim_token = self.config['scale_dim_token']
        self.dropout = self.config['dropout']
        self.bottleneck_dim = self.config['bottleneck_dim']
        self.Dinput = self.config['Dinput']
        self.manipulation = bool(self.config['manipulation'])
        self.mesh_readout = str(self.config['mesh_readout']).lower()
        if self.mesh_readout not in ('linear', 'njf', 'poisson', 'hybrid'):
            raise ValueError(
                f"mesh_readout must be 'linear'|'njf'|'poisson'|'hybrid', got {self.mesh_readout!r}"
            )
        self.hybrid_final = str(self.config['hybrid_final']).lower()
        if self.hybrid_final == 'regional_absolute':
            self.hybrid_final = 'regional'
        if self.hybrid_final == 'additive':
            self.hybrid_final = 'residual'
        _hybrid_final_ok = (
            'poisson_lp', 'poisson', 'residual', 'regional',
            'regional_residual', 'njf',
        )
        if self.hybrid_final not in _hybrid_final_ok:
            raise ValueError(
                "hybrid_final must be "
                "'poisson_lp'|'poisson'|'residual'|'additive'|"
                "'regional_residual'|'regional'|'regional_absolute'|'njf', "
                f"got {self.hybrid_final!r}"
            )
        self.region_soft_edge_mult = float(self.config['region_soft_edge_mult'])
        self.scaffold_lowpass_iters = int(self.config['scaffold_lowpass_iters'])
        self.scaffold_lowpass_blend = float(self.config['scaffold_lowpass_blend'])
        self.C_width = int(self.config['C_width'])
        # Uniform 1-ring graph for Linear lowpass (filled in register_template_mesh).
        self.register_buffer('_lp_src', torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer('_lp_dst', torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer('_lp_deg', torch.zeros(0, dtype=torch.float32), persistent=False)

        region_info_file = Path(self.config['region_info_file'])
        with open(region_info_file, 'rb') as f:
            region_info = pickle.load(f)

        region_info = {k: v for k, v in region_info.items() if 'inner' not in k}
        self.region_names = list(region_info.keys())
        self.region_face_ids = [torch.tensor(_dict['fids']) for _dict in region_info.values()]
        assert len(self.region_face_ids) == self.Npatches, 'number of patches does not match.'
        faces = torch.cat(self.region_face_ids).unique()
        assert len(faces) == HACK['faces'], 'number of faces does not match.'

        self.region_vert_ids = [torch.tensor(_dict['vids']) for _dict in region_info.values()]
        self.region_n_verts = [v.shape[0] for v in self.region_vert_ids]
        self.n_vertices = len(torch.cat(self.region_vert_ids).unique())
        assert self.n_vertices == HACK['verts'], 'number of vertices does not match.'

        self.encoder = MLPMixerPerLayerOut(
            dim=self.dim,
            depth=self.encoder_depth,
            num_patches=self.Npatches + 1,
            expansion_factor=self.scale_dim,
            expansion_factor_token=self.scale_dim_token,
            dropout=0,
        )

        poisson_cfg = dict(self.config['poisson_cfg'])
        self.xyz_to_token = PoissonXYZToToken(
            dim=self.dim,
            C_width=int(self.config['C_width']),
            n_blocks=int(self.config['poisson_token_blocks']),
            region_vert_ids=self.region_vert_ids,
            config=poisson_cfg,
        )

        self.semantic_embedding = nn.Parameter(torch.randn(self.Npatches, self.dim))
        self.id_token = nn.Parameter(torch.randn(1, 1, self.dim))

        if self.bottleneck_dim is not None:
            self.bottleneck_down = nn.Linear(self.dim, self.bottleneck_dim)
            self.bottleneck_up = nn.Linear(self.bottleneck_dim, self.dim)
        else:
            self.bottleneck_down = nn.Identity()
            self.bottleneck_up = nn.Identity()

        self.learned_decoder_tokens = nn.Parameter(torch.randn(1, self.Npatches, self.dim))

        if self.manipulation:
            self._init_manipulation_modules()

        self.decoder = MLPMixerPerLayerOut(
            dim=self.dim,
            depth=self.decoder_depth,
            num_patches=self.Npatches + 1,
            expansion_factor=self.scale_dim,
            expansion_factor_token=self.scale_dim_token,
            dropout=0,
        )

        # Mesh readout modules (construct only what is needed → DDP-safe).
        self.token_to_xyz: nn.ModuleList | None = None
        self.mesh_readout_net: SmoothMeshReadout | None = None
        need_linear = self.mesh_readout in ('linear', 'hybrid')
        need_smooth = self.mesh_readout in ('njf', 'poisson', 'hybrid')
        if need_linear:
            self.token_to_xyz = nn.ModuleList([
                nn.Linear(self.dim, 3 * patch_dim) for patch_dim in self.region_n_verts
            ])
        if need_smooth:
            # hybrid final decode also runs PoissonBlocks before NJF when blocks>0.
            self.mesh_readout_net = SmoothMeshReadout(
                dim=self.dim,
                C_width=self.C_width,
                n_blocks=int(self.config['mesh_readout_blocks']),
                use_poisson_blocks=(self.mesh_readout in ('poisson', 'hybrid')),
                poisson_cfg=poisson_cfg,
                region_face_ids=self.region_face_ids,
            )

        # Template mesh + cached operators for Poisson tokenize / smooth readout
        self.register_buffer('template_verts', torch.zeros(0, 3), persistent=False)
        self.register_buffer('template_faces', torch.zeros(0, 3, dtype=torch.long), persistent=False)
        self.register_buffer('_soft_patch_weight', torch.zeros(0), persistent=False)
        self.register_buffer('seam_vert_mask', torch.zeros(0, dtype=torch.bool), persistent=False)
        self.register_buffer('seam_face_mask', torch.zeros(0, dtype=torch.bool), persistent=False)
        self._ops_cache: dict = {}

    @property
    def n_loss_layers(self):
        return self.encoder_depth + self.decoder_depth + 2

    def _init_manipulation_modules(self):
        control_vertices = resolve_control_vertices(self.config)
        if not control_vertices:
            raise ValueError(
                'manipulation=True requires MODEL.control_vertices or '
                'defaults from exp_poisson/hack_control_vertices.py.'
            )

        self.control_vertices = control_vertices
        self.control_region_keys = list(control_vertices.keys())
        self.control_region_sizes = [3 * len(v) for v in control_vertices.values()]
        self.delta_control_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size, 64, bias=False),
                nn.GELU(),
                nn.Linear(64, 256, bias=False),
                nn.GELU(),
                nn.Linear(256, self.dim, bias=False),
            )
            for size in self.control_region_sizes
        ])
        self._control_vid_tensors: list[torch.Tensor] = [
            torch.tensor(self.control_vertices[key], dtype=torch.long)
            for key in self.control_region_keys
        ]

    def control_vid_tensors(self, device: torch.device | None = None) -> list[torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        return [vids.to(device) for vids in self._control_vid_tensors]

    def register_template_mesh(self, verts: torch.Tensor, faces: torch.Tensor):
        """Cache template topology and cotangent operators for Poisson tokenize."""
        if verts.ndim == 2:
            verts = verts.unsqueeze(0)
        if faces.ndim == 2:
            faces = faces.unsqueeze(0)

        device = verts.device
        tv = verts[0].detach().to(device).float()
        if bool(self.config['center_template']):
            tv = tv - tv.mean(dim=0, keepdim=True)
        self.template_verts = tv
        self.template_faces = faces[0].detach().to(device).long()
        self.n_vertices = int(self.template_verts.shape[0])
        self._ops_cache = {}

        self.region_vert_ids = [vids.to(device) for vids in self.region_vert_ids]
        self.region_face_ids = [fids.to(device) for fids in self.region_face_ids]
        self.xyz_to_token.region_vert_ids = list(self.region_vert_ids)
        self._build_soft_patch_weights(device)
        self._build_seam_masks(device)
        self._build_lowpass_graph(device)
        if self.mesh_readout_net is not None:
            self.mesh_readout_net.set_region_face_ids(self.region_face_ids)
        if self.mesh_readout == 'hybrid' and self.hybrid_final in (
            'regional', 'regional_residual',
        ):
            self._cache_region_operators(device)

        if self.manipulation:
            self._control_vid_tensors = [
                torch.tensor(self.control_vertices[key], dtype=torch.long, device=device)
                for key in self.control_region_keys
            ]

    def _cache_region_operators(self, device: torch.device):
        """Build per-connected-component submesh ops (template geometry).

        Semantic regions that are disconnected (e.g. ``Ears`` = left+right)
        are split so each Poisson solve has a single gauge.
        """
        if self.template_verts.numel() == 0:
            raise RuntimeError('template_verts required before region operators')
        parts_key = ('region_parts', self._device_key(device))
        if parts_key in self._ops_cache:
            return

        high_prec = bool(self.config['operator_high_precision'])
        V = int(self.template_verts.shape[0])
        parts: list[dict] = []
        for ri, (vids, fids) in enumerate(zip(self.region_vert_ids, self.region_face_ids)):
            vids = vids.to(device).long()
            fids = fids.to(device).long()
            g2l = torch.full((V,), -1, dtype=torch.long, device=device)
            g2l[vids] = torch.arange(vids.shape[0], device=device)
            gfaces = self.template_faces.to(device)[fids]
            local_faces = g2l[gfaces]
            if (local_faces < 0).any():
                missing = int((local_faces < 0).sum().item())
                raise RuntimeError(
                    f'region {ri}: {missing} face-corner vids not in region_vert_ids'
                )
            n_local = int(vids.shape[0])
            for members in _mesh_connected_components(n_local, local_faces):
                members = members.to(device)
                member_set = set(members.tolist())
                # Faces fully inside this CC, remapped to 0..n_cc-1
                keep = []
                for tri in local_faces.tolist():
                    if all(int(v) in member_set for v in tri):
                        keep.append(tri)
                if not keep:
                    continue
                keep_f = torch.tensor(keep, dtype=torch.long, device=device)
                l2c = torch.full((n_local,), -1, dtype=torch.long, device=device)
                l2c[members] = torch.arange(members.shape[0], device=device)
                cc_faces = l2c[keep_f]
                cc_vids = vids[members]
                cc_verts = self.template_verts.to(device)[cc_vids].unsqueeze(0)
                mass, solver, G, M = construct_mesh_operators(
                    cc_verts,
                    cc_faces.unsqueeze(0),
                    high_precision=high_prec,
                )
                parts.append({
                    'patch': ri,
                    'vids': cc_vids,
                    'faces': cc_faces,
                    'mass': mass,
                    'solver': solver,
                    'G': G,
                    'M': M,
                })
        self._ops_cache[parts_key] = parts

    def _iter_region_parts(self, batch_size: int, device: torch.device):
        self._cache_region_operators(device)
        parts = self._ops_cache[('region_parts', self._device_key(device))]
        for pack in parts:
            faces = pack['faces'].unsqueeze(0).expand(batch_size, -1, -1)
            yield (
                pack['patch'],
                pack['vids'],
                pack['mass'],
                pack['solver'],
                pack['G'],
                pack['M'],
                faces,
            )

    def _build_soft_patch_weights(self, device: torch.device):
        """Gaussian soft membership over regions (for smooth mesh readout)."""
        verts = self.template_verts.to(device)
        faces = self.template_faces.to(device)
        edge_len = (verts[faces[:, 0]] - verts[faces[:, 1]]).norm(dim=-1).median()
        sigma = (self.region_soft_edge_mult * edge_len).clamp_min(1e-4)
        logits = []
        for vids in self.region_vert_ids:
            vids = vids.to(device)
            dist = torch.cdist(verts, verts[vids]).min(dim=1).values
            logits.append(-dist.pow(2) / (2.0 * sigma.pow(2)))
        self._soft_patch_weight = torch.softmax(torch.stack(logits, dim=-1), dim=-1)

    def _build_seam_masks(self, device: torch.device):
        """Region-boundary seam band: verts/faces on cross-region edges + 1-ring."""
        faces = self.template_faces.to(device).long()
        V = int(self.template_verts.shape[0])
        F = int(faces.shape[0])
        face_region = torch.full((F,), -1, dtype=torch.long, device=device)
        for ri, fids in enumerate(self.region_face_ids):
            face_region[fids.to(device).long()] = ri
        if (face_region < 0).any():
            raise RuntimeError('region_face_ids leave some faces unassigned')

        e01 = torch.stack((faces[:, 0], faces[:, 1]), dim=1)
        e12 = torch.stack((faces[:, 1], faces[:, 2]), dim=1)
        e20 = torch.stack((faces[:, 2], faces[:, 0]), dim=1)
        all_e = torch.cat((e01, e12, e20), dim=0)
        all_fi = torch.arange(F, device=device).repeat(3)
        ea = all_e.min(dim=1).values
        eb = all_e.max(dim=1).values
        keys = ea * V + eb
        order = torch.argsort(keys)
        keys_s = keys[order]
        fi_s = all_fi[order]
        ea_s = ea[order]
        eb_s = eb[order]

        seam_vert = torch.zeros(V, dtype=torch.bool, device=device)
        seam_face = torch.zeros(F, dtype=torch.bool, device=device)
        i = 0
        n = keys_s.shape[0]
        while i < n:
            j = i + 1
            while j < n and keys_s[j] == keys_s[i]:
                j += 1
            fis = fi_s[i:j]
            if fis.numel() == 2 and int(face_region[fis[0]]) != int(face_region[fis[1]]):
                seam_face[fis] = True
                seam_vert[ea_s[i]] = True
                seam_vert[eb_s[i]] = True
            i = j

        # 1-ring: any face touching a seam vert; verts of those faces.
        if seam_vert.any():
            touch = (
                seam_vert[faces[:, 0]]
                | seam_vert[faces[:, 1]]
                | seam_vert[faces[:, 2]]
            )
            seam_face = seam_face | touch
            ring_verts = torch.zeros(V, dtype=torch.bool, device=device)
            for c in range(3):
                ring_verts[faces[touch, c]] = True
            seam_vert = seam_vert | ring_verts

        self.seam_vert_mask = seam_vert
        self.seam_face_mask = seam_face
        print(
            f'[planmm] seam band: verts={int(seam_vert.sum())}/{V}, '
            f'faces={int(seam_face.sum())}/{F}'
        )

    def _build_lowpass_graph(self, device: torch.device):
        """Cache undirected 1-ring edges for uniform Laplacian smoothing."""
        faces = self.template_faces.to(device).long()
        V = int(self.template_verts.shape[0])
        e = torch.cat(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            dim=0,
        )
        ea = e.min(dim=1).values
        eb = e.max(dim=1).values
        keys = ea * V + eb
        uniq = torch.unique(keys)
        a = uniq // V
        b = uniq % V
        src = torch.cat((a, b), dim=0)
        dst = torch.cat((b, a), dim=0)
        deg = torch.bincount(src, minlength=V).to(dtype=torch.float32).clamp_min(1.0)
        self._lp_src = src
        self._lp_dst = dst
        self._lp_deg = deg

    def _lowpass_verts(
        self,
        x: torch.Tensor,
        n_iters: int | None = None,
        blend: float | None = None,
    ) -> torch.Tensor:
        """Uniform 1-ring Jacobi smoothing on template connectivity.

        ``x_{t+1} = (1-λ) x_t + λ A x_t`` with A the neighbor-average matrix.
        Differentiable w.r.t. ``x`` (Linear scaffold).
        """
        if self._lp_src.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before lowpass.')
        n_iters = self.scaffold_lowpass_iters if n_iters is None else int(n_iters)
        blend = self.scaffold_lowpass_blend if blend is None else float(blend)
        if n_iters <= 0 or blend <= 0.0:
            return x
        blend = min(max(blend, 0.0), 1.0)
        src = self._lp_src.to(device=x.device)
        dst = self._lp_dst.to(device=x.device)
        deg = self._lp_deg.to(device=x.device, dtype=x.dtype).view(1, -1, 1)
        out = x
        for _ in range(n_iters):
            acc = torch.zeros_like(out)
            acc.index_add_(1, src, out[:, dst])
            neigh = acc / deg
            out = (1.0 - blend) * out + blend * neigh
        return out

    def _device_key(self, device: torch.device):
        return device.index if device.type == 'cuda' else -1

    def _ensure_batch_operators(self, device: torch.device):
        key = ('full', self._device_key(device))
        if key in self._ops_cache:
            return

        if self.template_verts.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before forward().')

        verts = self.template_verts.unsqueeze(0).to(device)
        faces = self.template_faces.unsqueeze(0).to(device)
        mass, solver, G, M = construct_mesh_operators(
            verts,
            faces,
            high_precision=bool(self.config['operator_high_precision']),
        )
        self._ops_cache[key] = (mass, solver, G, M)

    def _get_batch_operators(self, batch_size: int, device: torch.device):
        """Return template operators (B=1 shared) + faces expanded to batch.

        ``bmm_G`` / ``apply_poisson_solve`` already broadcast B=1 G/solver.
        """
        self._ensure_batch_operators(device)
        mass, solver, G, M = self._ops_cache[('full', self._device_key(device))]
        faces = self.template_faces.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return mass, solver, G, M, faces

    def tokenize(self, x: torch.Tensor):
        """PoissonBlock tokenize → [B, K, dim]."""
        B = x.shape[0]
        mass, solver, G, M, faces = self._get_batch_operators(B, x.device)
        return self.xyz_to_token(x, M, G, solver, faces, mass)

    def encode(self, x: torch.Tensor, return_layer_outputs: bool = True):
        B = x.shape[0]
        encoder_tokens = self.tokenize(x)
        encoder_tokens = encoder_tokens + self.semantic_embedding
        encoder_tokens = torch.cat((self.id_token.repeat(B, 1, 1), encoder_tokens), dim=1)
        encoder_tokens = self.encoder(encoder_tokens)
        id_token = encoder_tokens[-1][:, 0].unsqueeze(1)
        id_token = self.bottleneck_down(id_token)
        if not return_layer_outputs:
            return id_token, None
        # hybrid: encoder side is intermediate → always Linear (NJF only on final decode).
        if self.mesh_readout == 'hybrid':
            return id_token, self._merge_linear(encoder_tokens)
        return id_token, self.merge(encoder_tokens)

    def decode(self, id_token: torch.Tensor, delta_lms: list[torch.Tensor] | None = None, return_layer_outputs: bool=True):
        B = id_token.shape[0]
        id_token = self.bottleneck_up(id_token)
        patch_tokens = self.learned_decoder_tokens.repeat(B, 1, 1)
        if self.manipulation and delta_lms is not None:
            delta_token = [fc(delta) for fc, delta in zip(self.delta_control_net, delta_lms)]
            for i, idx in enumerate(self.control_region_keys):
                patch_tokens[:, idx] = patch_tokens[:, idx] + delta_token[i]
        decoder_tokens = torch.cat((id_token, patch_tokens), dim=1)
        decoder_tokens = self.decoder(decoder_tokens)
        if self.mesh_readout == 'hybrid':
            return self._merge_hybrid_decode(decoder_tokens, return_layer_outputs)
        if not return_layer_outputs:
            return self.merge(decoder_tokens[-1:])[0]
        return self.merge(decoder_tokens)

    def merge(self, tokens: list[torch.Tensor]):
        """Map mixer layer tokens → multilayer meshes [L, B, V, 3].

        ``linear``: LAMM per-region ``token_to_xyz`` + scatter-average.
        ``njf`` / ``poisson``: soft-scatter → smooth NJF on every layer.
        ``hybrid``: use ``encode`` / ``_merge_hybrid_decode`` instead.
        """
        if self.mesh_readout in ('linear', 'hybrid'):
            # hybrid encoder path also lands here only via _merge_linear explicitly.
            return self._merge_linear(tokens)
        return self._merge_smooth(tokens)

    def poisson_grad_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'mean',
        face_mask: torch.Tensor | None = None,
    ):
        """Face-area-weighted L1 on intrinsic gradients (Poisson / NJF domain).

        Cotangent ``G`` and face-area ``M`` are built from the **target** mesh
        (template topology, target vertex positions; operators detached). Then

        ``‖M ⊙ (G pred - G target)‖₁``

        so the metric matches the ground-truth surface rather than the neutral
        template.

        Args:
            face_mask: optional bool ``[F]`` — if set, only those faces contribute
                (interleaved gradient rows for each selected face).
        """
        if pred.shape != target.shape:
            raise ValueError(f'pred/target shape mismatch: {pred.shape} vs {target.shape}')
        flat = pred.ndim == 3
        if flat:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        L, B, V, C = pred.shape
        if self.template_faces.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before poisson_grad_loss().')

        device = pred.device
        tgt_ops = target.reshape(L * B, V, C).detach()
        faces = self.template_faces.unsqueeze(0).expand(L * B, -1, -1).to(device)
        _, _, G, M = construct_mesh_operators(
            tgt_ops,
            faces,
            high_precision=bool(self.config['operator_high_precision']),
            poisson_solve=False,
        )
        g_pred = bmm_G(G, pred.reshape(L * B, V, C))
        g_tgt = bmm_G(G, target.reshape(L * B, V, C))
        err = (g_pred - g_tgt).abs()
        # M: [L*B, 2F] face-area weights for interleaved gradient rows.
        w = M.to(device=err.device, dtype=err.dtype).unsqueeze(-1)
        if face_mask is not None:
            fm = face_mask.to(device=device, dtype=torch.bool).reshape(-1)
            if fm.numel() != self.template_faces.shape[0]:
                raise ValueError(
                    f'face_mask length {fm.numel()} != F={self.template_faces.shape[0]}'
                )
            # Interleaved rows: [A0, A0, A1, A1, ...] → expand mask to 2F.
            row_mask = fm.repeat_interleave(2)  # [2F]
            w = w * row_mask.to(dtype=w.dtype).view(1, -1, 1)
        weighted = err * w
        if reduction == 'none':
            out = weighted.reshape(L, B, -1, C)
            return out[0] if flat else out
        denom = w.sum().clamp_min(1e-8) * C
        return weighted.sum() / denom

    def seam_vertex_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Mean L1 on the seam-band vertices (final mesh only)."""
        if self.seam_vert_mask.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before seam_vertex_loss().')
        if pred.shape != target.shape:
            raise ValueError(f'pred/target shape mismatch: {pred.shape} vs {target.shape}')
        mask = self.seam_vert_mask.to(device=pred.device)
        if not mask.any():
            return pred.new_zeros(())
        return (pred[:, mask] - target[:, mask]).abs().mean()

    def seam_grad_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Area-weighted gradient L1 restricted to seam-band faces."""
        if self.seam_face_mask.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before seam_grad_loss().')
        return self.poisson_grad_loss(pred, target, face_mask=self.seam_face_mask)

    def _merge_hybrid_decode(self, decoder_tokens: list[torch.Tensor], return_layer_outputs: bool):
        """Hybrid decode: mid = Linear ``token_to_xyz``; final = NJF variant.

        ``poisson_lp`` (default): final mesh **is** the global Poisson solution
        with Grad residual anchored at ``G@lowpass(Linear)``. Full Linear only
        conditions GradMLP and sets COM — coarse pose without Linear HF in the
        Jacobian base.
        ``poisson``: same but Grad base at template (ablation; under-deforms).
        ``residual`` / ``additive``: ``scaffold + detail`` ablation.
        ``regional_residual``: Linear + per-CC NJF residual (ablation).
        ``regional`` / ``regional_absolute``: per-CC ``template + absolute NJF``.
        ``njf``: global ``template + NJF`` (no Linear conditioning).
        """
        if self.mesh_readout_net is None or self.token_to_xyz is None:
            raise RuntimeError("mesh_readout='hybrid' requires both linear and NJF heads")
        if self.template_verts.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before hybrid decode.')

        linear_all = self._merge_linear(decoder_tokens)  # [L, B, V, 3]
        layer_tok = decoder_tokens[-1]
        if self.hybrid_final == 'poisson_lp':
            scaffold = self._merge_linear_soft([layer_tok])[0]
            last = self._njf_poisson_mesh_from_tokens(
                layer_tok, scaffold, grad_base='lowpass',
            )
        elif self.hybrid_final == 'poisson':
            scaffold = self._merge_linear_soft([layer_tok])[0]
            last = self._njf_poisson_mesh_from_tokens(
                layer_tok, scaffold, grad_base='template',
            )
        elif self.hybrid_final == 'residual':
            scaffold = self._merge_linear_soft([layer_tok])[0]
            detail = self._njf_disp_from_tokens(layer_tok, scaffold=scaffold)
            last = scaffold + detail
        elif self.hybrid_final == 'regional_residual':
            last = self._njf_regional_mesh_from_tokens(
                layer_tok, scaffold=linear_all[-1],
            )
        elif self.hybrid_final == 'regional':
            last = self._njf_regional_mesh_from_tokens(layer_tok, scaffold=None)
        else:
            last = self._njf_mesh_from_tokens(layer_tok)

        if not return_layer_outputs:
            return last
        if linear_all.shape[0] == 1:
            return last.unsqueeze(0)
        return torch.cat((linear_all[:-1], last.unsqueeze(0)), dim=0)

    def _njf_disp_from_tokens(self, layer_tokens: torch.Tensor, scaffold: torch.Tensor | None = None):
        """Soft-scatter one mixer layer → global NJF displacement [B, V, 3]."""
        if self.mesh_readout_net is None:
            raise RuntimeError('mesh_readout_net is required for NJF displacement')
        B = layer_tokens.shape[0]
        device = layer_tokens.device
        mass, solver, G, M, faces = self._get_batch_operators(B, device)
        patch_tokens = layer_tokens[:, 1:]
        feat = self._soft_scatter_tokens(patch_tokens)
        return self.mesh_readout_net(feat, M, G, solver, faces, mass, scaffold=scaffold)

    def _njf_poisson_mesh_from_tokens(
        self,
        layer_tokens: torch.Tensor,
        scaffold: torch.Tensor,
        grad_base: str = 'lowpass',
    ):
        """Global Poisson mesh; Linear conditions grads + sets COM.

        ``grad_base``:
          - ``lowpass``: residual around ``G@lowpass(scaffold)`` (default).
          - ``template``: residual around ``G@template`` (ablation).
        Soft-scatter tokens + ``scaffold - template`` condition Δ; translational
        gauge follows Linear's mass COM. Output is never ``scaffold + detail``.
        """
        if self.mesh_readout_net is None:
            raise RuntimeError('mesh_readout_net is required for poisson hybrid final')
        if self.mesh_readout_net.to_r3 is None:
            raise RuntimeError(
                "hybrid_final poisson* requires grad_residual or regional_grad "
                '(to_r3 for token conditioning)'
            )
        grad_base = str(grad_base).lower()
        if grad_base not in ('lowpass', 'template'):
            raise ValueError(f"grad_base must be 'lowpass'|'template', got {grad_base!r}")

        B = layer_tokens.shape[0]
        device = layer_tokens.device
        mass, solver, G, M, faces = self._get_batch_operators(B, device)
        template = self.template_verts.unsqueeze(0).expand(B, -1, -1).to(device)
        feat = self._soft_scatter_tokens(layer_tokens[:, 1:])
        net = self.mesh_readout_net
        x = net._run_blocks(feat, M, G, solver, faces, mass)
        cond = vertices_to_faces(net.to_r3(x), faces)
        # Full Linear relative pose as GradMLP cue (detail / identity).
        cond = cond + vertices_to_faces(scaffold - template, faces)

        if grad_base == 'lowpass':
            x_in = self._lowpass_verts(scaffold)
        else:
            x_in = template

        u, _ = net.njf_head(
            x_in, M, G, solver, faces, mass,
            cond_faces=cond, return_detail=False,
        )
        w = mass.unsqueeze(-1)
        com = (scaffold * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return u + com

    def _njf_mesh_from_tokens(self, layer_tokens: torch.Tensor):
        """``template + absolute NJF`` (global) → [B, V, 3]."""
        B = layer_tokens.shape[0]
        device = layer_tokens.device
        template = self.template_verts.unsqueeze(0).expand(B, -1, -1).to(device)
        return template + self._njf_disp_from_tokens(layer_tokens, scaffold=None)

    def _njf_regional_mesh_from_tokens(
        self,
        layer_tokens: torch.Tensor,
        scaffold: torch.Tensor | None = None,
    ):
        """Per-connected-component NJF, soft-blend overlaps.

        ``scaffold is None`` (default regional):
            ``mesh = template[vids] + absolute_NJF(feat)``
        ``scaffold`` given (regional_residual ablation):
            ``mesh = scaffold[vids] + residual_NJF(feat)``
        """
        if self.mesh_readout_net is None:
            raise RuntimeError('mesh_readout_net is required for regional NJF')
        B = layer_tokens.shape[0]
        device = layer_tokens.device
        V = int(self.n_vertices)
        patch_tokens = layer_tokens[:, 1:]
        feat = self._soft_scatter_tokens(patch_tokens)  # [B, V, dim]
        soft_w = self._soft_patch_weight.to(device=device, dtype=feat.dtype)  # [V, K]
        template = self.template_verts.to(device)

        if scaffold is None:
            base_full = template.unsqueeze(0).expand(B, -1, -1)
        else:
            base_full = scaffold

        accum = torch.zeros(B, V, 3, device=device, dtype=feat.dtype)
        weight = torch.zeros(B, V, 1, device=device, dtype=feat.dtype)
        njf = self.mesh_readout_net.njf_head
        old_face_region = njf.face_region_id

        try:
            for ri, vids, mass, solver, G, M, faces in self._iter_region_parts(B, device):
                Fr = faces.shape[1]
                if njf.regional_grad and njf.region_embed is not None:
                    njf.face_region_id = torch.full(
                        (Fr,), ri, dtype=torch.long, device=device,
                    )
                feat_r = feat[:, vids]
                if scaffold is None:
                    disp_r = self.mesh_readout_net(
                        feat_r, M, G, solver, faces, mass, scaffold=None,
                    )
                    mesh_r = template[vids].unsqueeze(0) + disp_r
                else:
                    scaffold_r = scaffold[:, vids]
                    detail_r = self.mesh_readout_net(
                        feat_r, M, G, solver, faces, mass, scaffold=scaffold_r,
                    )
                    mesh_r = scaffold_r + detail_r
                wr = soft_w[vids, ri].view(1, -1, 1).expand(B, -1, 1)
                accum[:, vids] = accum[:, vids] + mesh_r * wr
                weight[:, vids] = weight[:, vids] + wr
        finally:
            njf.face_region_id = old_face_region

        bare = weight.squeeze(-1) < 1e-8
        if bare.any():
            accum = torch.where(bare.unsqueeze(-1), base_full, accum)
            weight = torch.where(bare.unsqueeze(-1), torch.ones_like(weight), weight)
        return accum / weight.clamp_min(1e-8)

    def _merge_linear(self, tokens: list[torch.Tensor]):
        B = tokens[0].shape[0]
        output_tokens = [d[:, 1:] for d in tokens]
        xyz = self.inverse_tokenize(output_tokens)
        outputs = torch.zeros((len(tokens), B, self.n_vertices, 3), device=tokens[0].device)
        counts = torch.zeros((len(tokens), B, self.n_vertices, 1), device=tokens[0].device)
        for i, layer_output in enumerate(xyz):
            for j, face_part in enumerate(layer_output):
                vids = self.region_vert_ids[j].to(tokens[0].device)
                outputs[i, :, vids] += face_part
                counts[i, :, vids] += 1.0
        return outputs / counts.clamp(min=1.0)

    def _merge_linear_soft(self, tokens: list[torch.Tensor]):
        """Linear ``token_to_xyz`` with soft region weights on overlapping verts.

        Falls back to hard merge if soft weights are not registered yet.
        """
        if self._soft_patch_weight.numel() == 0:
            return self._merge_linear(tokens)
        B = tokens[0].shape[0]
        device = tokens[0].device
        dtype = tokens[0].dtype
        soft_w = self._soft_patch_weight.to(device=device, dtype=dtype)
        output_tokens = [d[:, 1:] for d in tokens]
        xyz = self.inverse_tokenize(output_tokens)
        outputs = torch.zeros((len(tokens), B, self.n_vertices, 3), device=device, dtype=dtype)
        counts = torch.zeros((len(tokens), B, self.n_vertices, 1), device=device, dtype=dtype)
        for i, layer_output in enumerate(xyz):
            for j, face_part in enumerate(layer_output):
                vids = self.region_vert_ids[j].to(device)
                w = soft_w[vids, j].view(1, -1, 1)
                outputs[i, :, vids] = outputs[i, :, vids] + face_part * w
                counts[i, :, vids] = counts[i, :, vids] + w
        return outputs / counts.clamp_min(1e-8)

    def _soft_scatter_tokens(self, patch_tokens: torch.Tensor):
        """[B, K, dim] → [B, V, dim] via soft region membership."""
        if self._soft_patch_weight.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before smooth mesh readout.')
        w = self._soft_patch_weight.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        return torch.einsum('vk,bkc->bvc', w, patch_tokens)

    def _merge_smooth(self, tokens: list[torch.Tensor]):
        """``njf``/``poisson``/hybrid-final: ``template + NJF`` (integrable field)."""
        if self.mesh_readout_net is None:
            raise RuntimeError(f'mesh_readout={self.mesh_readout} but readout net is missing')
        if self.template_verts.numel() == 0:
            raise RuntimeError('Call register_template_mesh() before smooth mesh readout.')

        B = tokens[0].shape[0]
        device = tokens[0].device
        mass, solver, G, M, faces = self._get_batch_operators(B, device)
        scaffold = self.template_verts.unsqueeze(0).expand(B, -1, -1).to(device)

        meshes = []
        for layer_tokens in tokens:
            patch_tokens = layer_tokens[:, 1:]  # drop id token
            feat = self._soft_scatter_tokens(patch_tokens)
            disp = self.mesh_readout_net(feat, M, G, solver, faces, mass)
            meshes.append(scaffold + disp)
        return torch.stack(meshes, dim=0)

    def inverse_tokenize(self, y: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        """ Linear token_to_xyz (only when ``mesh_readout='linear'``)."""
        if self.token_to_xyz is None:
            raise RuntimeError("inverse_tokenize requires mesh_readout='linear'")
        B = y[0].shape[0]
        return [
            [
                self.token_to_xyz[i](out_tokens[:, i]).reshape(B, self.region_n_verts[i], 3)
                for i in range(self.Npatches)
            ]
            for out_tokens in y
        ]

    def forward(self, x: torch.Tensor | tuple | list):
        """
        Args:
            x: [B, V, 3] mesh, or (mesh, delta_lms) in manipulation mode.

        Returns:
            [L, B, V, 3] multilayer outputs (encoder + decoder).
        """
        delta_lms = None
        if isinstance(x, (tuple, list)):
            x, delta_lms = x

        id_token, encoder_outputs = self.encode(x)
        decoder_outputs = self.decode(id_token, delta_lms)
        return torch.cat((encoder_outputs, decoder_outputs), dim=0)


def load_template_from_config(config, device='cpu'):
    ref = Path(config.get('reference_obj', PLANMM.default_cfg['reference_obj']))
    obj = read_obj(str(ref), tri=True)
    verts = torch.tensor(obj.vs, dtype=torch.float32, device=device)
    faces = torch.tensor(obj.fvs, dtype=torch.long, device=device)
    if config.get('center_template', PLANMM.default_cfg['center_template']):
        verts = verts - verts.mean(dim=0, keepdim=True)
    return verts, faces