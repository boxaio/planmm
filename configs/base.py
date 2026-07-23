import yaml
import sys
import importlib
from pathlib import Path
from dataclasses import asdict, dataclass, is_dataclass, field
from typing import Any, Optional, Literal, Tuple, Union, Dict, List
from yacs.config import CfgNode as CN

# sys.path.append(str(Path(__file__).parent.parent))

# from utils.log import get_logger


def import_module(module_name: str):
    module_name, class_name = module_name.rsplit(".", 1)
    module = getattr(importlib.import_module(module_name), class_name)
    return module

class Config:
    def __getitem__(self, __name: str):
        if hasattr(self, __name):
            return getattr(self, __name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{__name}'")

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


def cfg_to_dict(cfg_node):
    result = {}
    for key, value in cfg_node.items():
        if isinstance(value, CN):
            result[key] = cfg_to_dict(value)
        else:
            result[key] = value
    return result


class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            elif isinstance(value, list):
                setattr(self, key, [ConfigObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        return None
    
    def __getitem__(self, key):
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            setattr(self, key, ConfigObject(value))
        elif isinstance(value, list):
            setattr(self, key, [ConfigObject(item) if isinstance(item, dict) else item for item in value])
        else:
            setattr(self, key, value)
            
    def __contains__(self, key):
        return hasattr(self, key)
    
    def keys(self):
        return list(self.__dict__.keys())
    
    def items(self):
        return list(self.__dict__.items())
    
    def values(self):
        return list(self.__dict__.values())
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConfigObject) else item for item in value]
            else:
                result[key] = value
        return result

def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return ConfigObject(data) if data else ConfigObject({})
    except FileNotFoundError:
        print(f"Error: cannot find {file_path}")
    except yaml.YAMLError as e:
        print(f"Error: parsing yaml file - {e}")
    return ConfigObject({})


if __name__ == "__main__":
    cfg = load_config("/media/ubuntu/Elements/MyExp/THACK/config/fit_flame.yml")
    
    calibrated = cfg.Data.calibrated  
    print(f"Calibrated: {calibrated}")

    stage_cfg = cfg.Stages
    
    





@dataclass()
class DataConfig(Config):
    root_folder: Path
    """The root folder for the dataset."""
    sequence: str
    """The sequence name"""
    _target: str = "dataset.meadDataset.MEADDataset"
    """The target dataset class"""
    division: Optional[str] = None
    subset: Optional[str] = None
    calibrated: bool = False
    """Whether the cameras parameters are available"""
    align_cameras_to_axes: bool = True
    """Adjust how cameras distribute in the space with a global rotation"""
    camera_convention_conversion: str = 'opencv->opengl'
    target_extrinsic_type: Literal['w2c', 'c2w'] = 'w2c'
    n_downsample_rgb: Optional[int] = None
    """Load from downsampled RGB images to save data IO time"""
    scale_factor: float = 1.0
    """Further apply a scaling transformation after the downsampling of RGB"""
    background_color: Optional[Literal['white', 'black']] = 'white'
    use_alpha_map: bool = False
    use_landmark: bool = True


@dataclass()
class ImageDataConfig(Config):
    # The root folder for the dataset
    data_folder: Path = Path("/media/ubuntu/BoxAI/Datasets/nerf_synthetic/lego/")
    # The identity name
    identity_name: str = "lego"
    extension: str = ".jpg"
    # subsampling resolution, default is 1
    resolution: int = 1
    background_color: Optional[Literal['white', 'black']] = 'white'
    use_alpha_map: bool = False
    use_landmark: bool = True


@dataclass()
class ModelConfig(Config):
    n_shape_params: int = 300
    n_expr_params: int = 100
    n_texture_params: int = 100

    flame_params_ckpt: Path = None

    """Optimize static offsets on top of FLAME vertices in the canonical space"""
    use_static_offset: bool = True
    """Optimize dynamic offsets on top of the FLAME vertices in the canonical space""" 
    use_dynamic_offset: bool = False
    
    use_teeth: bool = True
    remove_lip_inside: bool = False

    """The resolution of the extra texture map"""
    texture_resolution: int = 2048
    """Use a painted texture map instead the pca texture space as the base texture map"""
    texture_painted: bool = True
    """Optimize an extra texture map as the base texture map or the residual texture map"""
    texture_extra: bool = True
    # tex_clusters: tuple[str, ...] = ("skin", "hair", "sclerae", "lips_tight", "boundary")
    """Regions that are supposed to share a similar color inside"""
    texture_clusters: tuple[str, ...] = (
        "skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises",
    )
    """Use the extra texture map as a residual component on top of the base texture"""
    residual_texture: bool = True

    """The regions that are occluded by the hair or garments"""
    occluded: tuple[str, ...] = ()  # to be used for updating stage configs in __post_init__
    
    device: Literal['cuda', 'cpu'] = 'cuda'
    eval: bool = False



@dataclass()
class RenderConfig(Config):
    # The rendering backend
    backend: Literal['nvdiffrast', 'pytorch3d'] = 'nvdiffrast'
    use_opengl: bool = False
    # Background color/image for training
    background_train: Literal['white', 'black', 'target'] = 'target'
    # The rate of disturbance for the foreground
    disturb_rate_fg: Optional[float] = 0.5
    # The rate of disturbance for the background. 0.6 best for multi-view, 0.3 best for single-view
    disturb_rate_bg: Optional[float] = 0.5
    # Background color/image for evaluation
    background_eval: Literal['white', 'black', 'target'] = 'target'
    # The type of lighting
    lighting_type: Literal['constant', 'front', 'front-range', 'SH'] = 'SH'
    # The space of lighting
    lighting_space: Literal['world', 'camera'] = 'world'


# @dataclass()
# class OptimizationConfig(Config):
#     iterations: int = 30_000
#     position_lr_init: float = 0.00016
#     position_lr_final: float = 0.0000016
#     position_lr_delay_mult: float = 0.01
#     position_lr_max_steps: int = 30_000
#     feature_lr: float = 0.0025
#     opacity_lr: float = 0.05
#     scaling_lr: float = 0.005
#     rotation_lr: float = 0.001
#     percent_dense: float = 0.01
#     lambda_dssim: float = 0.2
#     densification_interval: int = 100
#     opacity_reset_interval: int = 3000
#     densify_from_iter: int = 500
#     densify_until_iter: int = 15_000
#     densify_grad_threshold: float = 0.0002
#     random_background: bool = False


@dataclass()
class LearningRateConfig(Config):
    base: float = 5e-3
    """shape, texture, rotation, eyes, neck, jaw"""
    translation: float = 1e-3
    expr: float = 5e-2
    static_offset: float = 5e-4
    dynamic_offset: float = 5e-4
    camera: float = 5e-3
    lights: float = 5e-3


@dataclass()
class LossWeightConfig(Config):
    landmark: Optional[float] = 10.
    always_enable_jawline_landmarks: bool = True
    """Always enable the landmark loss for the jawline landmarks. Ignore disable_jawline_landmarks in stages."""

    photo: Optional[float] = 30.

    # L2 regularization
    reg_shape: float = 3e-1
    reg_neck: float = 3e-1
    reg_jaw: float = 3e-1
    reg_eyes: float = 3e-2
    reg_expr: float = 3e-2

    # regularize the texture map
    reg_tex_res_clusters: Optional[float] = 1e1
    """Regularize the residual texture map inside each texture cluster"""
    reg_tex_res_for: tuple[str, ...] = ("sclerae", "teeth")
    """Regularize the residual texture map for the clusters specified"""
    reg_tex_tv: Optional[float] = 1e4  # important to split regions apart
    """Regularize the total variation of the texture map"""
    reg_tex_pca: float = 1e-4  # will make it hard to model hair color when too high
    """Regularize the pca texture map (not effective when model.tex_painted is True)"""

    # regularize the lighting
    reg_light: Optional[float] = None
    """Regularize lighting parameters"""
    reg_diffuse: Optional[float] = 1e2
    """Regularize lighting parameters by the diffuse term"""

    # L2 regularization for static_offset
    reg_offset: Optional[float] = 3e2
    """Regularize the norm of offsets"""
    reg_offset_relax_coef: float = 1.
    """The coefficient for relaxing reg_offset for the regions specified"""
    reg_offset_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    # laplacian regularization for static_offset
    reg_offset_lap: Optional[float] = 1e6
    """Regularize the difference of laplacian coordinate caused by offsets"""
    reg_offset_lap_relax_coef: float = 0.1
    """The coefficient for relaxing reg_offset_lap for the regions specified"""
    reg_offset_lap_relax_for: tuple[str, ...] = ("hair", "ears")
    """Relax the offset loss for the regions specified"""

    # local rigidity regularization for static_offset
    reg_offset_rigid: Optional[float] = 3e2
    """Regularize the the offsets to be as-rigid-as-possible"""
    reg_offset_rigid_for: tuple[str, ...] = ("left_ear", "right_ear", "neck", "left_eye", "right_eye", "lips_tight")
    """Regularize the the offsets to be as-rigid-as-possible for the regions specified"""

    reg_offset_dynamic: Optional[float] = 3e5
    """Regularize the dynamic offsets to be temporally smooth"""

    blur_iter: int = 0
    """The number of iterations for blurring vertex weights"""
    
    # temporal smoothness
    smooth_trans: float = 3e2
    """global translation"""
    smooth_rot: float = 3e1
    """global rotation"""
    smooth_neck: float = 3e1
    """neck joint"""
    smooth_jaw: float = 1e-1
    """jaw joint"""
    smooth_eyes: float = 0
    """eyes joints"""
    smooth_expr: float = 1e0
    """expression"""
    

@dataclass()
class LogConfig(Config):
    log_path: Path = Path(__file__).parent.parent/"logs"
    interval_scalar: Optional[int] = 100
    """The step interval of scalar logging. Using an interval of stage_tracking.num_steps // 5 unless specified."""
    interval_media: Optional[int] = 500
    """The step interval of media logging. Using an interval of stage_tracking.num_steps unless specified."""
    image_format: Literal['jpg', 'png'] = 'jpg'
    """Output image format"""
    view_indices: Tuple[int, ...] = ()
    """Manually specify the view indices for log"""
    max_num_views: int = 3
    """The maximum number of views for log"""
    stack_views_in_rows: bool = True


@dataclass()
class ExperimentConfig(Config):
    output_folder: Path = Path(__file__).parent.parent/"output/track"
    reuse_landmarks: bool = True
    keyframes: Tuple[int, ...] = tuple()
    photometric: bool = True
    """enable photometric optimization, otherwise only landmark optimization"""



@dataclass()
class StageConfig(Config):
    disable_jawline_landmarks: bool = False
    """Disable the landmark loss for the jawline landmarks since they are not accurate"""

@dataclass()
class StageLmkInitRigidConfig(StageConfig):
    """The stage for initializing the rigid parameters"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose")

@dataclass()
class StageLmkInitAllConfig(StageConfig):
    """The stage for initializing all the parameters optimizable with landmark loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr")

@dataclass()
class StageLmkSequentialTrackConfig(StageConfig):
    """The stage for sequential tracking with landmark loss"""
    num_steps: int = 50
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr")

@dataclass()
class StageLmkGlobalTrackConfig(StageConfig):
    """The stage for global tracking with landmark loss"""
    num_epochs: int = 30
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr")

@dataclass()
class PhotometricStageConfig(StageConfig):
    align_texture_except: tuple[str, ...] = ()
    """Align the inner region of rendered FLAME to the image, except for the regions specified"""
    align_boundary_except: tuple[str, ...] = ("bottomline",)  # necessary to avoid the bottomline of FLAME from being stretched to the bottom of the image
    """Align the boundary of FLAME to the image, except for the regions specified"""

@dataclass()
class StageRgbInitTextureConfig(PhotometricStageConfig):
    """The stage for initializing the texture map with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "shape", "texture", "lights")
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    align_boundary_except: tuple[str, ...] = ("hair", "boundary")

@dataclass()
class StageRgbInitAllConfig(PhotometricStageConfig):
    """The stage for initializing all the parameters except the offsets with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights")
    disable_jawline_landmarks: bool = True
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")
    align_boundary_except: tuple[str, ...] = ("hair", "bottomline")

@dataclass()
class StageRgbInitOffsetConfig(PhotometricStageConfig):
    """The stage for initializing the offsets with photometric loss"""
    num_steps: int = 500
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights", "static_offset")
    disable_jawline_landmarks: bool = True
    align_texture_except: tuple[str, ...] = ("hair", "boundary", "neck")

@dataclass()
class StageRgbSequentialTrackConfig(PhotometricStageConfig):
    """The stage for sequential tracking with photometric loss"""
    num_steps: int = 50
    optimizable_params: tuple[str, ...] = ("pose", "joints", "expr", "texture", "dynamic_offset")
    disable_jawline_landmarks: bool = True

@dataclass()
class StageRgbGlobalTrackConfig(PhotometricStageConfig):
    """The stage for global tracking with photometric loss"""
    num_epochs: int = 30
    optimizable_params: tuple[str, ...] = ("cam", "pose", "shape", "joints", "expr", "texture", "lights", "static_offset", "dynamic_offset")
    disable_jawline_landmarks: bool = True

@dataclass()
class PipelineConfig(Config):
    lmk_init_rigid: StageLmkInitRigidConfig
    lmk_init_all: StageLmkInitAllConfig
    lmk_sequential_track: StageLmkSequentialTrackConfig
    lmk_global_track: StageLmkGlobalTrackConfig
    rgb_init_texture: StageRgbInitTextureConfig
    rgb_init_all: StageRgbInitAllConfig
    rgb_init_offset: StageRgbInitOffsetConfig
    rgb_sequential_track: StageRgbSequentialTrackConfig
    rgb_global_track: StageRgbGlobalTrackConfig


@dataclass()
class BaseTrackConfig(Config):
    data: DataConfig
    model: ModelConfig
    render: RenderConfig
    log: LogConfig
    exp: ExperimentConfig
    lr: LearningRateConfig
    w: LossWeightConfig
    pipeline: PipelineConfig

    """Begin from the specified stage for debugging"""
    begin_stage: Optional[str] = None
    """Begin from the specified frame index for debugging"""
    begin_frame_idx: int = 0
    """Allow asynchronous function calls for speed up"""
    async_func: bool = True
    device: Literal['cuda', 'cpu'] = 'cuda'

    def get_occluded(self):
        occluded_table = {
        }
        if self.data.sequence in occluded_table:
            logger.info(f"Automatically setting cfg.model.occluded to {occluded_table[self.data.sequence]}")
            self.model.occluded = occluded_table[self.data.sequence]

    def __post_init__(self):
        self.get_occluded()

        if not self.model.use_static_offset and not self.model.use_dynamic_offset:
            self.model.occluded = tuple(list(self.model.occluded) + ['hair'])  # disable boundary alignment for the hair region if no offset is used

        for cfg_stage in self.pipeline.__dict__.values():
            if isinstance(cfg_stage, PhotometricStageConfig):
                cfg_stage.align_texture_except = tuple(list(cfg_stage.align_texture_except) + list(self.model.occluded))
                cfg_stage.align_boundary_except = tuple(list(cfg_stage.align_boundary_except) + list(self.model.occluded))

        if self.begin_stage is not None:
            skip = True
            for cfg_stage in self.pipeline.__dict__.values():
                if cfg_stage.__class__.__name__.lower() == self.begin_stage:
                    skip = False
                if skip:
                    cfg_stage.num_steps = 0




def dataclass_to_dict(obj: Any, use_AttrDict: bool=True) -> Union[Dict, 'AttrDict']:
    if is_dataclass(obj):
        data = asdict(obj)
    else:
        data = dict(obj) if hasattr(obj, '__dict__') else obj

    if not isinstance(data, dict):
        return data

    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)
        elif is_dataclass(v) or hasattr(v, '__dict__'):
            data[k] = dataclass_to_dict(v, use_AttrDict)
        elif isinstance(v, tuple):
            data[k] = [dataclass_to_dict(i, use_AttrDict) for i in v]

    return AttrDict(data) if use_AttrDict else data


def save_configs_to_yaml(file_path: str="./logs/config.yaml"):
    try:
        config_dicts_to_yaml = {}
        for k, v in config_dicts.items():
            config_dicts_to_yaml[k] = dataclass_to_dict(v, False)

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(config_dicts_to_yaml, f, 
                      sort_keys=False, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {file_path}: {str(e)}")
