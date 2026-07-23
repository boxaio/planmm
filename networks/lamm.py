"""LAMM: Locally Adaptive Morphable Model with selectable token backbone.

Encoder/decoder default to ``MLPMixerPerLayerOut``; set ``backbone: transformer``
to use ``TransformerPerLayerOut`` instead.

When ``manipulation=False``, standard autoencoder with static ``learned_decoder_tokens``.

When ``manipulation=True``, control-vertex deltas update decoder patch tokens (LAMM-style).
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
from networks.module_mlpmixer import MLPMixerPerLayerOut
from networks.module_transformer import TransformerPerLayerOut


HACK = {
    'verts': 14062,
    'faces': 28068,
}


def resolve_control_vertices(config) -> dict[int, list[int]]:
    """Resolve LAMM-style control vertices from config or demo defaults."""
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
) -> torch.Tensor:
    """Per-layer GT blend λ in [0, 1]: encoder 1→0, decoder 0→1 (LAMM eq.1-2)."""
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
) -> torch.Tensor:
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


class LAMM(nn.Module):
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
        'dim_head': None,  # default: dim // heads
        'Npatches': 10,
        'dim': 512,
        # Token backbone for encoder & decoder: 'mlpmixer' | 'transformer'
        'backbone': 'mlpmixer',
        'region_info_file': _REPO_ROOT / 'dataset' / 'hack_region_info.pkl',
        'reference_obj': _REPO_ROOT / 'dataset' / 'hack_template.obj',
        'center_template': True,
        'manipulation': False,
        'control_vertices': {},
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
        self.backbone = str(self.config['backbone']).lower()
        if self.backbone in ('mlp', 'mixer', 'mlp_mixer'):
            self.backbone = 'mlpmixer'
        if self.backbone not in ('mlpmixer', 'transformer'):
            raise ValueError(
                f"backbone must be 'mlpmixer'|'transformer', got {self.backbone!r}"
            )
        dim_head = self.config['dim_head']
        self.dim_head = int(dim_head) if dim_head is not None else max(1, self.dim // self.heads)

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

        self.encoder = self._build_token_backbone(self.encoder_depth)

        self.xyz_to_token = nn.ModuleList([
            nn.Linear(self.Dinput * patch_dim, self.dim)
            for patch_dim in self.region_n_verts
        ])

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

        self.decoder = self._build_token_backbone(self.decoder_depth)
        self.token_to_xyz = nn.ModuleList([
            nn.Linear(self.dim, 3 * patch_dim) for patch_dim in self.region_n_verts
        ])

    def _build_token_backbone(self, depth: int) -> nn.Module:
        """Build encoder or decoder stack: MLPMixer or Transformer."""
        if self.backbone == 'transformer':
            return TransformerPerLayerOut(
                dim=self.dim,
                depth=depth,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=int(self.dim * self.scale_dim),
                dropout=self.dropout,
                return_input=True,
            )
        return MLPMixerPerLayerOut(
            dim=self.dim,
            depth=depth,
            num_patches=self.Npatches + 1,
            expansion_factor=self.scale_dim,
            expansion_factor_token=self.scale_dim_token,
            dropout=0,
        )

    @property
    def n_loss_layers(self) -> int:
        return self.encoder_depth + self.decoder_depth + 2

    def _init_manipulation_modules(self):
        control_vertices = resolve_control_vertices(self.config)
        if not control_vertices:
            raise ValueError(
                'manipulation=True requires MODEL.control_vertices or '
                'defaults from exp_poisson/demo_hack_control_verts.py.'
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
        """Control vertex index tensors on ``device`` (always available in manipulation mode)."""
        if device is None:
            device = next(self.parameters()).device
        return [vids.to(device) for vids in self._control_vid_tensors]

    def get_regions(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, N, D = x.shape
        return [x[:, ids].reshape(B, -1) for ids in self.region_vert_ids]

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        regions = self.get_regions(x)
        return torch.stack(
            [self.xyz_to_token[i](regions[i]) for i in range(self.Npatches)],
            dim=1,
        )

    def encode(self, x: torch.Tensor):
        B = x.shape[0]
        encoder_tokens = self.tokenize(x)
        encoder_tokens = encoder_tokens + self.semantic_embedding
        encoder_tokens = torch.cat((self.id_token.repeat(B, 1, 1), encoder_tokens), dim=1)
        encoder_tokens = self.encoder(encoder_tokens)
        id_token = encoder_tokens[-1][:, 0].unsqueeze(1)
        id_token = self.bottleneck_down(id_token)
        outputs = self.merge(encoder_tokens)
        return id_token, outputs

    def decode(self, id_token: torch.Tensor, delta_lms: list[torch.Tensor] | None = None):
        B = id_token.shape[0]
        id_token = self.bottleneck_up(id_token)
        patch_tokens = self.learned_decoder_tokens.repeat(B, 1, 1)
        if self.manipulation and delta_lms is not None:
            delta_token = [fc(delta) for fc, delta in zip(self.delta_control_net, delta_lms)]
            for i, idx in enumerate(self.control_region_keys):
                patch_tokens[:, idx] = patch_tokens[:, idx] + delta_token[i]
        decoder_tokens = torch.cat((id_token, patch_tokens), dim=1)
        decoder_tokens = self.decoder(decoder_tokens)
        return self.merge(decoder_tokens)

    def merge(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        B = tokens[0].shape[0]
        output_tokens = [d[:, 1:] for d in tokens]
        xyz = self.inverse_tokenize(output_tokens)
        outputs = torch.zeros((len(tokens), B, self.n_vertices, 3), device=tokens[0].device)
        counts = torch.zeros((len(tokens), B, self.n_vertices, 1), device=tokens[0].device)
        for i, layer_output in enumerate(xyz):
            for j, face_part in enumerate(layer_output):
                outputs[i, :, self.region_vert_ids[j]] += face_part
                counts[i, :, self.region_vert_ids[j]] += 1.0
        return outputs / counts.clamp(min=1.0)

    def inverse_tokenize(self, y: list[torch.Tensor]) -> list[list[torch.Tensor]]:
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
               delta_lms: list of [B, 3*N] flattened control-vertex displacements
               per control region (LAMM convention).

        Returns:
            [L, B, V, 3] multilayer outputs (encoder + decoder, LAMM-style).
        """
        delta_lms = None
        if isinstance(x, (tuple, list)):
            x, delta_lms = x

        id_token, encoder_outputs = self.encode(x)
        decoder_outputs = self.decode(id_token, delta_lms)
        return torch.cat((encoder_outputs, decoder_outputs), dim=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae_cfg = {
        'Npatches': 10,
        'dim': 256,
        'encoder_depth': 3,
        'decoder_depth': 3,
        'bottleneck_dim': 256,
        'manipulation': False,
    }
    for backbone in ('mlpmixer', 'transformer'):
        model = LAMM({**ae_cfg, 'backbone': backbone}).to(device)
        x = torch.randn(2, HACK['verts'], 3, device=device)
        out = model(x)
        print(
            f'{backbone}: params={sum(p.numel() for p in model.parameters()):,}, '
            f'out={tuple(out.shape)}'
        )
        assert out.shape == (model.n_loss_layers, 2, HACK['verts'], 3)

    manip_cfg = {
        **ae_cfg,
        'backbone': 'transformer',
        'manipulation': True,
    }
    manip = LAMM(manip_cfg).to(device)
    delta = [
        torch.zeros(2, size, device=device)
        for size in manip.control_region_sizes
    ]
    recon = manip((x, delta))
    print(f'Manip(transformer) output: {tuple(recon.shape)}')
    assert recon.shape == (manip.n_loss_layers, 2, HACK['verts'], 3)
    print('smoke test passed')
