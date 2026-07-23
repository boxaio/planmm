import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import PoissonSolver, PoissonBlockMLP, output_at, vertices_to_faces, faces_to_vertices


def _solver_at(solvers, batch_idx: int):
    """Pick batch solver; reuse solvers[0] when operators are shared (B=1 cache)."""
    return solvers[0] if len(solvers) == 1 else solvers[batch_idx]


def bmm_G(G: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Intrinsic gradient G @ x. G may be [1, 2F, V] shared across batch."""
    if G.ndim == 3 and G.shape[0] == 1 and x.shape[0] > 1:
        g0 = G[0]
        if g0.is_sparse:
            return torch.stack([torch.sparse.mm(g0, x[b]) for b in range(x.shape[0])], dim=0)
        return torch.bmm(g0.unsqueeze(0).expand(x.shape[0], -1, -1), x)
    if G.ndim == 2 and G.is_sparse:
        return torch.stack([torch.sparse.mm(G, x[b]) for b in range(x.shape[0])], dim=0)
    return torch.bmm(G, x)


def bmm_GT(G: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """G^T @ x where x is [B, 2F, C]."""
    if G.ndim == 3 and G.shape[0] == 1 and x.shape[0] > 1:
        g0 = G[0]
        if g0.is_sparse:
            gt = g0.transpose(0, 1)
            return torch.stack([torch.sparse.mm(gt, x[b]) for b in range(x.shape[0])], dim=0)
        gt = g0.transpose(0, 1).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.bmm(gt, x)
    if G.ndim == 2 and G.is_sparse:
        gt = G.transpose(0, 1)
        return torch.stack([torch.sparse.mm(gt, x[b]) for b in range(x.shape[0])], dim=0)
    gt = G.transpose(1, 2)
    if gt.shape[0] == 1 and x.shape[0] > 1:
        return torch.bmm(gt.expand(x.shape[0], -1, -1), x)
    return torch.bmm(gt, x)


def apply_poisson_solve(solvers, rhs: torch.Tensor) -> torch.Tensor:
    """Solve Lu = rhs per batch item (shared solver when len(solvers)==1)."""
    u = torch.zeros_like(rhs)
    for b in range(rhs.shape[0]):
        sb = _solver_at(solvers, b)
        if rhs.shape[-1] > 128:
            for j in range(0, rhs.shape[-1], 128):
                u[b][:, j:j + 128] = PoissonSolver.apply(sb, rhs[b][:, j:j + 128].contiguous())
        else:
            u[b] = PoissonSolver.apply(sb, rhs[b].contiguous())
    return u


class NJFHead(nn.Module):
    """Predict face gradients then Poisson-integrate to vertex displacements.

    Config flags (in ``config``):
      - ``grad_residual``: if True, ``grads = grads_in + Δ(...)``. Requires
        ``in_c == out_c`` (typically map features → R³ before this head).
      - ``regional_grad``: if True, a **shared** ComplexMLP conditioned on a
        per-face region embedding (one batched forward; requires
        ``region_face_ids``). With residual:
        ``grads = G@x + Δ(G@x, x_face + emb_r)``.
    """

    def __init__(
        self,
        in_c,
        out_c,
        width,
        config=None,
        region_face_ids: list[torch.Tensor] | None = None,
    ):
        super().__init__()
        config = config or {}
        cmlp_nlayers = config.get('cmlp_nlayers', 2)
        cmlp_modulate = bool(config.get('cmlp_modulate', False))
        self.grad_residual = bool(config.get('grad_residual', False))
        self.regional_grad = bool(config.get('regional_grad', False))
        self.in_c = in_c
        self.out_c = out_c

        if self.grad_residual and in_c != out_c:
            raise ValueError(
                f'grad_residual requires in_c == out_c, got in_c={in_c}, out_c={out_c}. '
                'Project vertex features to R^3 before NJFHead.'
            )

        # Region conditioning is injected via ComplexMLP's face modulator.
        if self.regional_grad:
            cmlp_modulate = True

        self.grad_mlp = ComplexMLP(
            in_c=in_c, out_c=out_c, width=width,
            num_layers=cmlp_nlayers, modulate=cmlp_modulate,
        )
        if self.grad_residual:
            _zero_init_complex_mlp_last(self.grad_mlp)

        self.n_regions = 0
        self.region_embed: nn.Embedding | None = None
        self.register_buffer('face_region_id', torch.zeros(0, dtype=torch.long), persistent=False)

        if self.regional_grad:
            if not region_face_ids:
                raise ValueError('regional_grad=True requires region_face_ids')
            self.n_regions = len(region_face_ids)
            self.region_embed = nn.Embedding(self.n_regions, in_c)
            nn.init.zeros_(self.region_embed.weight)
            self._rebuild_face_region_id(region_face_ids)

    def _rebuild_face_region_id(self, region_face_ids: list[torch.Tensor]):
        # Keep buffer on the module device so DDP/NCCL sync does not see CPU tensors
        # after register_template_mesh() (called after .to(cuda)).
        if self.region_embed is not None:
            device = self.region_embed.weight.device
        elif self.face_region_id.is_cuda or self.face_region_id.device.type != 'cpu':
            device = self.face_region_id.device
        else:
            device = torch.device('cpu')

        fids_list = [f.detach().long().reshape(-1).to(device) for f in region_face_ids]
        n_faces = int(max(int(f.max().item()) for f in fids_list) + 1)
        face_region = torch.full((n_faces,), -1, dtype=torch.long, device=device)
        for r, fids in enumerate(fids_list):
            face_region[fids] = r
        if (face_region < 0).any():
            missing = int((face_region < 0).sum().item())
            raise ValueError(f'region_face_ids leave {missing} faces unassigned')
        self.face_region_id = face_region

    def set_region_face_ids(self, region_face_ids: list[torch.Tensor]):
        """Update face→region map (e.g. after device move)."""
        if not self.regional_grad:
            return
        if len(region_face_ids) != self.n_regions:
            raise ValueError(
                f'region_face_ids length {len(region_face_ids)} != n_regions {self.n_regions}'
            )
        self._rebuild_face_region_id(region_face_ids)

    def _predict_grads(self, grads_in: torch.Tensor, x_faces: torch.Tensor) -> torch.Tensor:
        """grads_in / out: [B, 2F, C] interleaved; x_faces: [B, F, C]."""
        x_cond = x_faces
        if self.regional_grad:
            emb = self.region_embed(self.face_region_id.to(device=x_faces.device))  # [F, C]
            x_cond = x_faces + emb.unsqueeze(0)
        out = self.grad_mlp(grads_in, x_cond)
        if self.grad_residual:
            return grads_in + out
        return out

    def forward(self, x_in, M, G, solver, faces, vertex_mass, original_grads=None, **kwargs):
        """Integrate predicted grads; pin translational gauge by mass-mean.

        Kwargs:
          - ``cond_faces``: optional [B, F, C] added to face conditioning
            (e.g. token features while ``x_in`` is a Linear scaffold).
          - ``return_detail``: if True, return mass-centered ``u - x_in``
            (integrable residual on top of a coarse scaffold).
        """
        cond_faces = kwargs.pop('cond_faces', None)
        return_detail = bool(kwargs.pop('return_detail', False))

        grads_in = bmm_G(G, x_in)   # (B, 2F, C)
        x_faces = vertices_to_faces(x_in, faces)   # (B, F, C)
        if cond_faces is not None:
            x_faces = x_faces + cond_faces
        grads = self._predict_grads(grads_in, x_faces)
        if original_grads is not None:
            grads = grads + original_grads

        # Solve Poisson equation Lu = ∇^T @ face_areas * grads
        rhs = bmm_GT(G, M.unsqueeze(-1) * grads)  # (B, V, C)
        u = apply_poisson_solve(solver, rhs)

        if return_detail:
            detail = u - x_in
            w = vertex_mass.unsqueeze(-1)
            detail = detail - (detail * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return detail, grads

        # Absolute displacement: mass-weighted global mean removal.
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) \
                  / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)
        return u, grads


def _zero_init_complex_mlp_last(mlp: 'ComplexMLP'):
    """Zero last ComplexLayer so residual Grad starts as identity (Δ≈0)."""
    last = mlp.layers[-1]
    nn.init.zeros_(last.lin_real.weight)
    nn.init.zeros_(last.lin_imag.weight)


class ComplexRotationScale(nn.Module):
    '''
    Modulation layer that applies a per-channel rotation and scale 
    to complex features that is dependent on scalar features x.
    Eq.(4) in the main paper.
    '''
    def __init__(self, in_c, out_c):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(in_c, in_c),
            nn.GELU(),
            nn.Linear(in_c, 2*out_c),   # output interpretted as [phase, scale]
        )
        self.scale_softplus = nn.Softplus()

    def forward(self, x, f_real, f_imag):
        # x: (B, F, C) scalar features averaged onto faces
        # f_real, f_imag: (B, F, C) real and imaginary parts of vector features.

        phase, scale = self.modulator(x).chunk(2, dim=-1)
        scale = self.scale_softplus(scale) + 1e-8   # map scale into positive values
        cos = torch.cos(phase)
        sin = torch.sin(phase)

        # Rotate by phase, and apply scale
        real_out = (f_real * cos - f_imag * sin) * scale
        imag_out = (f_real * sin + f_imag * cos) * scale
        return real_out, imag_out


class ComplexLayer(nn.Module):
    """
    A complex-valued linear map (C x C), plus an optional magnitude-based nonlinearity.
    Eq.(3) in the main paper.
    Operation:
        z_out := W @ z_in ,  where W is CxC complex mat.
        Then optionally apply:  z_out <- GELU(|z_out| + mag_bias) * (z_out / |z_out|).
    """
    def __init__(self, in_c, out_c, nonlin=True):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.nonlin = nonlin

        # Real and imaginary parts of complex weight matrix
        self.lin_real = nn.Linear(in_c, out_c, bias=False)
        self.lin_imag = nn.Linear(in_c, out_c, bias=False)

        if self.nonlin:
            self.mag_bias = nn.Parameter(torch.zeros(out_c), requires_grad=True)
            self.gelu = nn.GELU()

    def forward(self, f_real, f_imag):
        # f_real, f_imag: (B, F, C) real and imaginary parts of vector features.

        y_real = self.lin_real(f_real) - self.lin_imag(f_imag) # (B, F, C)
        y_imag = self.lin_real(f_imag) + self.lin_imag(f_real) # (B, F, C)

        if self.nonlin:
            r = torch.sqrt(y_real**2 + y_imag**2 + 1e-8)
            r_activated = self.gelu(r + self.mag_bias)
            scale = r_activated / (r + 1e-8)
            y_real = y_real * scale
            y_imag = y_imag * scale

        return y_real, y_imag


class ComplexMLP(nn.Module):
    def __init__(self, in_c, out_c, width, num_layers=3, modulate=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.layers = nn.ModuleList()

        for i in range(num_layers-1):
            self.layers.append(ComplexLayer(in_c if i == 0 else width, width, nonlin=True))
        self.layers.append(ComplexLayer(width, out_c, nonlin=False))
        
        self.modulator = None
        if modulate:
            self.modulator = ComplexRotationScale(in_c, in_c)

    def forward(self, f, x=None):
        """
        f: (B, 2F, C) with interleaved real/imag convention, OR (f_real, f_imag) tuple
        x: (B, F, C) scalar face-based features, used for modulation layer if enabled
        Returns: (B, 2F, C) interleaved transformed vector features
        """
        if isinstance(f, tuple):
            f_real, f_imag = f
        else:
            f_real = f[:, 0::2, :]
            f_imag = f[:, 1::2, :]
        B_, F_, C_ = f_real.shape

        out_re, out_im = f_real, f_imag

        if self.modulator is not None:
            out_re, out_im = self.modulator(x, out_re, out_im)

        for layer in self.layers:
            out_re, out_im = layer(out_re, out_im)

        out = torch.empty(B_, 2*F_, self.out_c, device=f_real.device)
        out[:, 0::2, :] = out_re
        out[:, 1::2, :] = out_im
        return out


class PoissonBlock(nn.Module):
    def __init__(self, in_c, out_c, width, extra_feats=0, config={}):
        super().__init__()
        drop_path = config.get('drop_path', 0.0)
        dropout_p = config.get('dropout_p', 0.0)
        cmlp_nlayers = config.get('cmlp_nlayers', 2)
        mlp_norm = config.get('mlp_norm', False)
        self.mass_norm = config.get('mass_norm', False)
        self.inner_prod_features = config.get('inner_prod_features', False)
        self.grad_hf_features = config.get('grad_hf_features', False)
        self.cmlp_modulate = config.get('cmlp_modulate', True)

        self.grad_mlp = ComplexMLP(
            in_c=in_c, out_c=out_c, width=width, 
            num_layers=cmlp_nlayers, modulate=self.cmlp_modulate,
        )

        mlp_in = in_c + out_c   # [x_in, pde_sol]
        use_grad_feats = self.inner_prod_features or self.grad_hf_features
        if use_grad_feats:
            mlp_in += out_c
        if self.inner_prod_features:   # optional
            self.grad_features = ComplexLayer(in_c=out_c, out_c=out_c, nonlin=False)
            self.grad_scaler = nn.Parameter(1e-2 * torch.ones(out_c), requires_grad=True)
        if self.grad_hf_features:
            self.grad_hf_scaler = nn.Parameter(5e-3 * torch.ones(out_c), requires_grad=True)
        if self.inner_prod_features and self.grad_hf_features:
            self.grad_feat_fuse = nn.Linear(2 * out_c, out_c, bias=False)
            nn.init.normal_(self.grad_feat_fuse.weight, std=0.02)
            
        self.vert_mlp = PoissonBlockMLP(
            in_c=mlp_in, out_c=out_c, width=width, drop_path=drop_path, drop=dropout_p, 
            grad_inputs=use_grad_feats, norm=mlp_norm, extra_feats=extra_feats,
        )

    def forward(self, x_in, M, G, solver, faces, vertex_mass, extra_features=None, **kwargs):
        '''
        - x_in:             (B, V, C)   scalar vertex features
        - M:                (B, 2F)     interleaved face areas [A0, A0, A1, A1, ...]
        - G:                (B, 2F, V)  intrinsic gradient operator
        - solver:           [B,]        list of solver objects
        - faces:            (B, F, 3)   face indices
        - vertex_mass:      (B, V)      lumped vertex masses
        - extra_features:   (B, V, C)   additional features to be concatenated to the input of the MLP

        1. Computes transformed gradient features:
                grads = VectorMLP(∇x_in, x_in)
        2. Solve Poisson equation: 
                Lu = ∇ ⋅ (M * grads)
        3. Optional grad-domain HF residual (bypasses Poisson, feeds vert MLP)
        4. Compute new vertex features:
                out = MLP([x_in, u, grad_feats]) + x_in
        See Fig. 2 in the paper for a diagram of the block.
        '''
        B, V, C = x_in.shape
        F = M.shape[1] // 2

        grads_in = bmm_G(G, x_in) # (B, 2F, C)
        x_face = vertices_to_faces(x_in, faces) # (B, F, C)
        grads = self.grad_mlp(grads_in, x_face) # (B, 2F, C)

        # Solve Poisson equation Lu = ∇^T @ face_areas * grads
        rhs = bmm_GT(G, M.unsqueeze(-1) * grads) # (B, V, C)
        u = apply_poisson_solve(solver, rhs)

        # nullify area-weighted mean:
        u = u - torch.sum(u * vertex_mass.unsqueeze(-1), dim=1, keepdim=True) / torch.sum(vertex_mass, dim=1, keepdim=True).unsqueeze(-1)

        # Optionally compute inner products of transformed gradient features 
        # -- see DiffusionNet inner-product features https://github.com/nmwsharp/diffusion-net
        gradient_features = None
        if self.inner_prod_features:
            ginX, ginY = grads_in[:, 0::2, :], grads_in[:, 1::2, :]
            gX, gY = grads[:, 0::2, :], grads[:, 1::2, :]
            gX, gY = self.grad_features(gX, gY)
            inner_prod = gX * ginX + gY * ginY   # [B, F, C]
            face_areas = M[:, 0::2]   # [B, F]
            gradient_features = faces_to_vertices(inner_prod, faces, face_areas, num_vertices=V)  # [B, V, C]
            gradient_features = torch.tanh(gradient_features * self.grad_scaler)

        grad_hf_vert = None
        if self.grad_hf_features:
            # Gradient delta magnitude: non-integrable HF bypasses Poisson solve.
            delta = grads - grads_in
            dX, dY = delta[:, 0::2, :], delta[:, 1::2, :]
            hf_mag = torch.sqrt(dX * dX + dY * dY + 1e-8)
            face_areas = M[:, 0::2]
            grad_hf_vert = faces_to_vertices(hf_mag, faces, face_areas, num_vertices=V)
            grad_hf_vert = torch.tanh(grad_hf_vert * self.grad_hf_scaler)

        grad_feats_for_mlp = None
        if gradient_features is not None and grad_hf_vert is not None:
            grad_feats_for_mlp = self.grad_feat_fuse(
                torch.cat((gradient_features, grad_hf_vert), dim=-1),
            )
        elif gradient_features is not None:
            grad_feats_for_mlp = gradient_features
        elif grad_hf_vert is not None:
            grad_feats_for_mlp = grad_hf_vert

        out = self.vert_mlp(
            x=x_in, 
            pde_sol=u, 
            grad_features=grad_feats_for_mlp, 
            extra_features=extra_features, 
            mass=vertex_mass if self.mass_norm else None,
        )

        return out, grads
    

class PoissonNet(nn.Module):
    def __init__(self, 
        C_in, C_out, C_width=128, n_blocks=4, head_type='njf', extra_features=0, outputs_at='vertices', 
        last_activation=nn.Identity(), config={}, **kwargs,
    ):   
        super().__init__()
        assert head_type.lower() in ['linear', 'mlp', 'njf'], \
            f"Invalid head type: {head_type.lower()}. Choose from ['linear', 'mlp', 'njf']."
        self.outputs_at = outputs_at
        self.last_act = last_activation

        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = n_blocks
        self.head_type = head_type.lower()

        # Bias-free cond MLP: zeros → zeros (avoids unconditional mean-face shortcut).
        if extra_features > 0:
            self.extra_feat_mlp = nn.Sequential(
                nn.Linear(extra_features, C_width, bias=False),
                nn.GELU(),
                nn.Linear(C_width, C_width, bias=False),
            )
            extra_features = C_width
        else:
            self.extra_feat_mlp = None

        self.multilayer_readout = bool(config.get('multilayer_readout', True))
        self.gradient_checkpoint = bool(config.get('gradient_checkpoint', False))

        self.proj_in = nn.Linear(C_in, C_width)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(
                PoissonBlock(in_c=C_width, out_c=C_width, width=C_width, extra_feats=extra_features, config=config)
            )

        # Nonzero init required: zero-init head ⇒ ∂L/∂feat = Wᵀ = 0, so id/cond
        # get no gradient and decode never learns expression (see run 112234).
        self.block_readout = nn.Linear(C_width, 3, bias=False) if self.multilayer_readout else None
        if self.block_readout is not None:
            nn.init.normal_(self.block_readout.weight, std=1e-2)

        self.head = nn.Identity()
        if self.head_type == 'njf':
            self.head = NJFHead(in_c=C_width, out_c=3, width=C_width, config=config)
        elif self.head_type == 'linear':
            self.head = nn.Linear(C_width, C_out, bias=False)
            nn.init.normal_(self.head.weight, std=1e-2)
        elif self.head_type == 'mlp':
            self.head = nn.Linear(60 * C_width, C_out, bias=False)
            nn.init.normal_(self.head.weight, std=1e-2)
        else:
            raise ValueError(f"Invalid head type: {self.head_type}. Choose from ['njf', 'linear'].")

    def _block_readout_xyz(self, x_in, feat):
        """Map block features to R^3; optional residual w.r.t. input scaffold."""
        delta = self.block_readout(feat)
        if getattr(self, '_njf_residual', False):
            return x_in + delta
        return delta

    def forward(self, x_in, M, G, solver, faces, vertex_mass, extra_features=None, return_layer_outputs=False, **kwargs):
        '''
        - x_in:             (B, V, C)   input scalar vertex features
        - M:                (B, 2F)     interleaved face areas [A0, A0, A1, A1, ...]
        - G:                (B, 2F, V)  intrinsic gradient operator
        - solver:           [B,]        list of solver objects
        - faces:            (B, F, 3)   face indices
        - vertex_mass:      (B, V)      lumped vertex masses
        - extra_features:   (B, C)   additional features to be concatenated to the input of block MLPs
        - njf_zero_anchor:  if True, integrate NJF from zero grad field (residual mode)
        - njf_residual:     if True, return x_in + poisson_out (displacement from x_in scaffold)
        '''
        njf_zero_anchor = bool(kwargs.pop('njf_zero_anchor', False))
        njf_residual = bool(kwargs.pop('njf_residual', False))
        self._njf_residual = njf_residual

        if self.head_type == 'njf':
            with torch.no_grad():
                if njf_zero_anchor:
                    grad_in = torch.zeros(
                        x_in.shape[0], M.shape[1], x_in.shape[-1],
                        device=x_in.device, dtype=x_in.dtype,
                    )
                else:
                    grad_in = bmm_G(G, x_in)

        if extra_features is not None and self.extra_feat_mlp is not None:
            extra_features = self.extra_feat_mlp(extra_features)
            if extra_features.ndim == 2:  # Assume (B, C)
                extra_features = extra_features.unsqueeze(1).expand(-1, x_in.shape[1], -1)
        elif extra_features is not None and self.extra_feat_mlp is None:
            if extra_features.ndim == 2:
                extra_features = extra_features.unsqueeze(1).expand(-1, x_in.shape[1], -1)

        x = self.proj_in(x_in) # (B, V, C_width)
        layer_outputs = []
        # decoder_depth blocks + final head → decoder_depth+1 (LAMM)
        for block in self.blocks:
            if self.gradient_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                def _block_fwd(feat, blk, m, g, sol, fc, vm, extra):
                    out, _ = blk(feat, m, g, sol, fc, vm, extra_features=extra, **kwargs)
                    return out

                x = checkpoint(
                    _block_fwd, x, block, M, G, solver, faces, vertex_mass, extra_features,
                    use_reentrant=False,
                )
            else:
                x, _ = block(
                    x, M, G, solver, faces, vertex_mass, extra_features=extra_features, **kwargs,
                )

            if return_layer_outputs:
                if self.head_type == 'njf':
                    # Intermediate layers must also go through NJF (linear readout
                    # produces non-integrable Δ and blows up multilayer training).
                    mid, _ = self.head(
                        x, M, G, solver, faces, vertex_mass, original_grads=grad_in,
                    )
                    if njf_residual:
                        mid = x_in + mid
                    layer_outputs.append(mid)
                elif self.block_readout is not None:
                    layer_outputs.append(self._block_readout_xyz(x_in, x))

        if self.head_type == 'njf':
            out, grads = self.head(x, M, G, solver, faces, vertex_mass, original_grads=grad_in)
            if njf_residual:
                out = x_in + out
            if return_layer_outputs:
                layer_outputs.append(out)
                return out, grads, layer_outputs
            return out, grads

        if self.head_type == 'linear':
            out = self.head(x)
        elif self.head_type == 'mlp':
            out = self.head(x)
        else:
            out = x

        out = output_at(out, faces, vertex_mass, domain=self.outputs_at)
        out = self.last_act(out)
        if njf_residual and out.shape[-1] == x_in.shape[-1]:
            out = x_in + out
        if return_layer_outputs:
            layer_outputs.append(out)
            return out, None, layer_outputs
        return out, None