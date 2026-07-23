import math
import os
import sys
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import lr_scheduler
from typing import Type, Any, Callable, Union, List, Optional, Dict
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def conv1x1(in_planes: int, out_planes: int, stride: int=1, bias: bool=False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# adapted from https://github.com/pytorch/vision/edit/master/torchvision/models/resnet.py
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        use_last_fc: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, 
        block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1, dilate: bool=False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, 
                self.base_width, previous_dilation, norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups, base_width=self.base_width, 
                    dilation=self.dilation, norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


class _SharedResidualBlock(nn.Module):
    """Pre-LN + linear + ReLU residual on the shared embedding (lightweight FFN)."""

    def __init__(self, dim: int, use_layer_norm: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.lin = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin(self.norm(x)))


class SharedTrunk(nn.Module):
    """
    Backbone pooled map -> task-shared embedding.
    LayerNorm stabilizes scale before splitting to shape / expr / pose / trans heads.
    Optional residual blocks add depth without widening heads.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(1)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            _SharedResidualBlock(hidden_dim, use_layer_norm)
            for _ in range(num_residual_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class ParamPredictionHead(nn.Module):
    """
    Shape / expression head without skip connections (residuals hurt low-dim FLAME regression).
    Linear -> [LN] -> GELU -> [Dropout] -> (optional extra Linear+[LN]+GELU blocks) -> Linear(out).

    Output layer uses small random init so early predictions stay near zero (DECA-style).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_layer_norm: bool = False,
        num_extra_hidden: int = 0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_linear = nn.Linear(in_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.in_act = nn.GELU()
        self.in_drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        mid: List[nn.Module] = []
        for _ in range(num_extra_hidden):
            mid.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                mid.append(nn.LayerNorm(hidden_dim))
            mid.append(nn.GELU())
            if dropout_p > 0:
                mid.append(nn.Dropout(dropout_p))
        self.mid = nn.Sequential(*mid)
        self.out_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_act(self.in_norm(self.in_linear(x)))
        h = self.in_drop(h)
        h = self.mid(h)
        return self.out_linear(h)


def _make_param_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    use_layer_norm: bool,
    num_extra_hidden: int,
    dropout_p: float = 0.0,
) -> nn.Module:
    """
    Legacy: Sequential(Linear, ReLU, Linear) -> checkpoint keys shape_head.0 / .1 / .2.
    Improved: ParamPredictionHead (GELU MLP, no residual). num_extra_hidden = extra Linear blocks
    in hidden space (not residual count).
    """
    if not use_layer_norm and num_extra_hidden == 0 and dropout_p <= 0:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    return ParamPredictionHead(
        in_dim,
        hidden_dim,
        out_dim,
        use_layer_norm,
        num_extra_hidden,
        dropout_p,
    )


class NeckPoseHead(nn.Module):
    """
    6-D neck pose (two 3-D blocks for HACK) from shared embedding.
    Per-sample LayerNorm on z removes coupling to global feature scale; a thin GELU MLP
    avoids an under-expressive single Linear that can leave one sample poorly fit per batch.
    """

    def __init__(self, shared_dim: int, hidden_dim: int, use_layer_norm: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shared_dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(shared_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 6)

    def forward(self, z: Tensor) -> Tensor:
        z = self.norm(z)
        return self.fc2(self.act(self.fc1(z)))


class TranslationHead(nn.Module):
    """
    3-D translation from shared embedding. LayerNorm on z (default on) stabilizes scale;
    a thin GELU MLP reduces underfitting vs a single Linear when trans competes with shape/expr.
    """

    def __init__(self, shared_dim: int, hidden_dim: int, use_layer_norm: bool = False) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(shared_dim) if use_layer_norm else nn.Identity()
        self.fc1 = nn.Linear(shared_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, z: Tensor) -> Tensor:
        z = self.norm(z)
        return self.fc2(self.act(self.fc1(z)))


class PHACKEncoderResNet(nn.Module):
    """
    DECA-style encoder: Backbone -> shared 512-d -> task heads (shape/expr MLP or ParamPredictionHead).
    Shape/Expression: legacy 2-layer ReLU MLP by default; optional GELU+LN MLP (no residual) via head_* flags.
    Neck pose: LayerNorm + small GELU MLP to 6-D (default), or legacy Linear if neck_pose_hidden=0.
    Trans: TranslationHead (LN + GELU MLP -> 3) by default, or legacy Linear if trans_hidden=0.
    All params are exposed for optimizer (backbone, shared, shape_head, expression_head, neck_pose_head, trans_head).

    shared_use_layer_norm: LayerNorm on the shared embedding (improves multi-head stability; new key names).
    shared_num_residual_blocks: Pre-LN residual Linear blocks on that embedding (0 = off).
        When both are False/0, use legacy Flatten→Linear→ReLU so old .pth shared.* still loads.
    head_use_layer_norm: shape & expression use GELU+LN head (no residual).
    head_num_hidden_residual: number of extra hidden Linear blocks (name kept for config compat; not residual).
    head_dropout: dropout after first hidden activation (0 = off). Legacy stack if LN/extra/dropout all off.
    neck_pose_hidden: hidden width for neck MLP; 0 = single Linear(shared_dim, 6) (old checkpoint layout).
    trans_hidden: hidden width for translation MLP; 0 = single Linear(shared_dim, 3).
    trans_use_layer_norm: when using TranslationHead, apply LayerNorm on z (recommended for trans).
    """
    fc_dim = 264
    def __init__(self, 
        net_recon='resnet50', use_last_fc=False, 
        backbone_path='/media/ubuntu/xb/cv_resnet50_face-reconstruction/init_model/resnet50-0676ba61.pth',
        pretrained: bool = False,
        ckpt_path: Optional[str] = None,
        shared_dim: int = 512,
        head_hidden: int = 512,
        shared_use_layer_norm: bool = False,
        shared_num_residual_blocks: int = 0,
        head_use_layer_norm: bool = False,
        head_num_hidden_residual: int = 0,
        head_dropout: float = 0.0,
        neck_pose_hidden: int = 128,
        trans_hidden: int = 128,
        trans_use_layer_norm: bool = True,
        n_shape: int = 200,
        n_expression: int = 55,
    ):
        super().__init__()
        self.use_last_fc = use_last_fc

        if net_recon not in func_dict:
            raise NotImplementedError(f'network [{net_recon}] is not implemented')
        func, last_dim = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)

        if backbone_path and os.path.isfile(backbone_path):
            state_dict = filter_state_dict(torch.load(backbone_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("loading init net_recon %s from %s" %(net_recon, backbone_path))

        self.backbone = backbone

        # Legacy stack matches old checkpoints (shared.0/1/2). Improved trunk uses new key names.
        if shared_use_layer_norm or shared_num_residual_blocks > 0:
            self.shared = SharedTrunk(
                last_dim,
                shared_dim,
                use_layer_norm=shared_use_layer_norm,
                num_residual_blocks=shared_num_residual_blocks,
            )
        else:
            self.shared = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(last_dim, shared_dim),
                nn.ReLU(inplace=True),
            )

        _hd = float(head_dropout)
        self.shape_head = _make_param_head(
            shared_dim,
            head_hidden,
            n_shape,
            head_use_layer_norm,
            head_num_hidden_residual,
            _hd,
        )
        self.expression_head = _make_param_head(
            shared_dim,
            head_hidden,
            n_expression,
            head_use_layer_norm,
            head_num_hidden_residual,
            _hd,
        )
        nph = int(neck_pose_hidden)
        if nph <= 0:
            self.neck_pose_head = nn.Linear(shared_dim, 6)
        else:
            self.neck_pose_head = NeckPoseHead(shared_dim, nph)
        nth = int(trans_hidden)
        if nth <= 0:
            self.trans_head = nn.Linear(shared_dim, 3)
        else:
            self.trans_head = TranslationHead(
                shared_dim, nth, use_layer_norm=bool(trans_use_layer_norm),
            )

        if pretrained:
            self._load_pretrained(ckpt_path)
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        if isinstance(self.shared, SharedTrunk):
            nn.init.kaiming_normal_(self.shared.proj.weight, mode='fan_out', nonlinearity='relu')
            if self.shared.proj.bias is not None:
                nn.init.zeros_(self.shared.proj.bias)
            for blk in self.shared.blocks:
                nn.init.xavier_uniform_(blk.lin.weight, gain=0.25)
                if blk.lin.bias is not None:
                    nn.init.zeros_(blk.lin.bias)
        else:
            for m in self.shared.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self._init_param_head(self.shape_head)
        self._init_param_head(self.expression_head)
        self._init_neck_pose_head()
        self._init_trans_head()

    def _init_neck_pose_head(self) -> None:
        try:
            gelu_gain = nn.init.calculate_gain('gelu')
        except (ValueError, AttributeError):
            gelu_gain = math.sqrt(2.0)
        if isinstance(self.neck_pose_head, NeckPoseHead):
            nn.init.xavier_uniform_(self.neck_pose_head.fc1.weight, gain=gelu_gain)
            if self.neck_pose_head.fc1.bias is not None:
                nn.init.zeros_(self.neck_pose_head.fc1.bias)
            nn.init.normal_(self.neck_pose_head.fc2.weight, mean=0.0, std=5e-3)
            if self.neck_pose_head.fc2.bias is not None:
                nn.init.zeros_(self.neck_pose_head.fc2.bias)
        else:
            nn.init.xavier_uniform_(self.neck_pose_head.weight, gain=0.25)
            if self.neck_pose_head.bias is not None:
                nn.init.zeros_(self.neck_pose_head.bias)

    def _init_trans_head(self) -> None:
        try:
            gelu_gain = nn.init.calculate_gain('gelu')
        except (ValueError, AttributeError):
            gelu_gain = math.sqrt(2.0)
        if isinstance(self.trans_head, TranslationHead):
            nn.init.xavier_uniform_(self.trans_head.fc1.weight, gain=gelu_gain)
            if self.trans_head.fc1.bias is not None:
                nn.init.zeros_(self.trans_head.fc1.bias)
            # Slightly larger than shape neck output: translation can span a wider numeric range.
            nn.init.normal_(self.trans_head.fc2.weight, mean=0.0, std=1e-2)
            if self.trans_head.fc2.bias is not None:
                nn.init.zeros_(self.trans_head.fc2.bias)
        else:
            nn.init.xavier_uniform_(self.trans_head.weight, gain=0.25)
            if self.trans_head.bias is not None:
                nn.init.zeros_(self.trans_head.bias)

    def _init_param_head(self, head: nn.Module) -> None:
        try:
            gelu_gain = nn.init.calculate_gain('gelu')
        except (ValueError, AttributeError):
            gelu_gain = math.sqrt(2.0)

        if isinstance(head, ParamPredictionHead):
            nn.init.xavier_uniform_(head.in_linear.weight, gain=gelu_gain)
            if head.in_linear.bias is not None:
                nn.init.zeros_(head.in_linear.bias)
            for m in head.mid.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=gelu_gain)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Small output init: start near zero FLAME coeffs (stable fine-tuning).
            nn.init.normal_(head.out_linear.weight, mean=0.0, std=1e-3)
            if head.out_linear.bias is not None:
                nn.init.zeros_(head.out_linear.bias)
        else:
            linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
            for i, m in enumerate(linears):
                if i < len(linears) - 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _load_pretrained(self, ckpt_path: Optional[str], device='cuda') -> None:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when pretrained=True")
        if not ckpt_path.endswith(".pth"):
            raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}. Only .pth is allowed")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            loaded = ckpt.get('phack_encoder', ckpt)
            missing, unexpected = self.load_state_dict(loaded, strict=False)
            if missing:
                print(f"[PHACKEncoder] pretrained: {len(missing)} missing keys (head layers may differ)")
            if unexpected:
                print(f"[PHACKEncoder] pretrained: {len(unexpected)} unexpected keys ignored")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model: {str(e)}") from e
    
    def forward(self, x):
        feat = self.backbone(x)       # [B, 2048, 1, 1]
        z = self.shared(feat)         # [B, hidden_dim]
        outputs = {
            "shape_params": self.shape_head(z),
            "expression_params": self.expression_head(z),
            "neck_pose": self.neck_pose_head(z),
            "trans": self.trans_head(z),
        }
        return outputs



func_dict = {
    'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048)
}


def model_config_encoder_kwargs(model_cfg: Any) -> Dict[str, Any]:
    """
    Build ``PHACKEncoderResNet`` keyword arguments from the training config ``Model`` section
    (``ConfigObject`` or ``dict``). Used by trainers and eval so checkpoint architecture matches yaml.
    """
    def _g(obj: Any, key: str, default=None):
        if obj is None:
            return default
        v = getattr(obj, key, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(key, None)
        return default if v is None else v

    backbone = _g(model_cfg, 'backbone', 'resnet50')
    if backbone not in func_dict:
        backbone = 'resnet50'

    return {
        'net_recon': backbone,
        'shared_use_layer_norm': bool(_g(model_cfg, 'encoder_shared_layer_norm', False)),
        'shared_num_residual_blocks': int(_g(model_cfg, 'encoder_shared_residual_blocks', 0)),
        'head_use_layer_norm': bool(_g(model_cfg, 'encoder_head_layer_norm', False)),
        'head_num_hidden_residual': int(_g(model_cfg, 'encoder_head_hidden_residual', 0)),
        'head_dropout': float(_g(model_cfg, 'encoder_head_dropout', 0.0)),
        'neck_pose_hidden': int(_g(model_cfg, 'encoder_neck_pose_hidden', 128)),
        'trans_hidden': int(_g(model_cfg, 'encoder_trans_hidden', 128)),
        'trans_use_layer_norm': bool(_g(model_cfg, 'encoder_trans_use_layer_norm', True)),
        'n_shape': int(_g(model_cfg, 'n_shape', 200)),
        'n_expression': int(_g(model_cfg, 'n_expression', 55)),
    }


if __name__ == '__main__':

    recon_net = PHACKEncoderResNet(net_recon='resnet50', use_last_fc=False)

    x = torch.randn((2, 3, 224, 224))
    y = recon_net(x)