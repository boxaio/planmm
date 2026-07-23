import sys
import numpy as np
import torch

from pathlib import Path
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer


class TrackerRasterizer(MeshRasterizer):
    def __init__(self, image_size, cameras) -> None:
        settings = RasterizationSettings()
        settings.image_size = (image_size, image_size)
        settings.perspective_correct = True
        settings.cull_backfaces = True

        super().__init__(cameras, settings)
        self.reset()

    def reset(self):
        self.bary_coords = None
        self.pix_to_face = None
        self.zbuffer = None

    def is_rasterize(self):
        return self.bary_coords is None and self.pix_to_face is None and self.zbuffer is None

    def forward(self, meshes, attributes, **kwargs):
        if self.is_rasterize():
            fragments = super().forward(meshes, **kwargs)
            self.bary_coords = fragments.bary_coords.detach()
            self.pix_to_face = fragments.pix_to_face.detach()
            self.zbuffer = fragments.zbuf.permute(0, 3, 1, 2).detach()

        vismask = (self.pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()  # (N, F, 3, D)
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        mask = self.pix_to_face == -1
        pix_to_face = self.pix_to_face.clone()  # [N, H, W, K]
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)  # [N, H, W, 1, D]
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)  # [N, D, H, W]
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)  # [N, D+1, H, W]

        return pixel_vals, self.zbuffer
