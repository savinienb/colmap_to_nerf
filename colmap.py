import typing
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
from utils import Indices
from _colmap_utils import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
from _colmap_utils import read_cameras_text, read_images_text, read_points3D_text, Image, Camera, Point3D
import sys
from abc import abstractmethod
import typing
from typing import Optional, Iterable, List, Dict, Any, cast, Union, Sequence, TYPE_CHECKING, overload, TypeVar, Iterator, Callable, Tuple
from dataclasses import dataclass
import dataclasses
import os
import numpy as np
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
try:
    from typing import Generic
except ImportError:
    from typing_extensions import Generic
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
try:
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import runtime_checkable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typing import get_args as get_args
    from typing import get_origin as get_origin
except ImportError:
    from typing_extensions import get_args as get_args
    from typing_extensions import get_origin as get_origin
try:
    from typing import NotRequired
    from typing import Required
    from typing import TypedDict
except ImportError:
    from typing_extensions import NotRequired
    from typing_extensions import Required
    from typing_extensions import TypedDict
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet


if TYPE_CHECKING:
    import torch


WG_PREFIX = os.path.expanduser(os.environ.get("WILD_GAUSSIANS_PREFIX", "~/.cache/nerfbaselines"))
ColorSpace = Literal["srgb", "linear"]
CameraModel = Literal["pinhole", "opencv", "opencv_fisheye", "full_opencv"]
DatasetFeature = Literal["color", "points3D_xyz", "points3D_rgb"]
TTensor = TypeVar("TTensor", np.ndarray, "torch.Tensor")
TTensor_co = TypeVar("TTensor_co", np.ndarray, "torch.Tensor", covariant=True)


@overload
def _get_xnp(tensor: np.ndarray):
    return np


@overload
def _get_xnp(tensor: 'torch.Tensor'):
    return cast('torch', sys.modules["torch"])


def _get_xnp(tensor: TTensor):
    if isinstance(tensor, np.ndarray):
        return np
    if tensor.__module__ == "torch":
        return cast('torch', sys.modules["torch"])
    raise ValueError(f"Unknown tensor type {type(tensor)}")


def camera_model_to_int(camera_model: CameraModel) -> int:
    camera_models = get_args(CameraModel)
    if camera_model not in camera_models:
        raise ValueError(f"Unknown camera model {camera_model}, known models are {camera_models}")
    return get_args(CameraModel).index(camera_model)


def camera_model_from_int(i: int) -> CameraModel:
    camera_models = get_args(CameraModel)
    if i >= len(camera_models):
        raise ValueError(f"Unknown camera model with index {i}, known models are {camera_models}")
    return get_args(CameraModel)[i]


class GenericCameras(Protocol[TTensor_co]):
    @property
    def poses(self) -> TTensor_co:
        """Camera-to-world matrices, [N, (R, t)]"""
        ...

    @property
    def intrinsics(self) -> TTensor_co:
        """Intrinsics, [N, (fx,fy,cx,cy)]"""
        ...

    @property
    def camera_models(self) -> TTensor_co:
        """Camera types, [N]"""
        ...

    @property
    def distortion_parameters(self) -> TTensor_co:
        """Distortion parameters, [N, num_params]"""
        ...

    @property
    def image_sizes(self) -> TTensor_co:
        """Image sizes, [N, 2]"""
        ...

    @property
    def nears_fars(self) -> Optional[TTensor_co]:
        """Near and far planes, [N, 2]"""
        ...

    @property
    def metadata(self) -> Optional[TTensor_co]:
        """Metadata, [N, ...]"""
        ...

    def __len__(self) -> int:
        ...

    def item(self) -> Self:
        """Returns a single camera if there is only one. Otherwise raises an error."""
        ...

    def __getitem__(self, index) -> Self:
        ...

    def __setitem__(self, index, value: Self) -> None:
        ...

    def __iter__(self) -> Iterator[Self]:
        ...

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        ...

    def replace(self, **changes) -> Self:
        ...

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCameras[TTensor]':
        ...


@runtime_checkable
class Cameras(GenericCameras[np.ndarray], Protocol):
    pass


@dataclass(frozen=True)
class GenericCamerasImpl(Generic[TTensor_co]):
    poses: TTensor_co  # [N, (R, t)]
    intrinsics: TTensor_co  # [N, (fx,fy,cx,cy)]

    camera_models: TTensor_co  # [N]
    distortion_parameters: TTensor_co  # [N, num_params]
    image_sizes: TTensor_co  # [N, 2]

    nears_fars: Optional[TTensor_co]  # [N, 2]
    metadata: Optional[TTensor_co] = None

    def __len__(self) -> int:
        return 1 if len(self.poses.shape) == 2 else len(self.poses)

    def item(self):
        assert len(self) == 1, "Cameras must have exactly one element to be converted to a single camera"
        return self if len(self.poses.shape) == 2 else self[0]

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            intrinsics=self.intrinsics[index],
            camera_models=self.camera_models[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index],
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
            metadata=self.metadata[index] if self.metadata is not None else None,
        )

    def __setitem__(self, index, value: Self) -> None:
        assert (self.image_sizes is None) == (value.image_sizes is None), "Either both or none of the cameras must have image sizes"
        assert (self.nears_fars is None) == (value.nears_fars is None), "Either both or none of the cameras must have nears and fars"
        self.poses[index] = value.poses
        self.intrinsics[index] = value.intrinsics
        self.camera_models[index] = value.camera_models
        self.distortion_parameters[index] = value.distortion_parameters
        self.image_sizes[index] = value.image_sizes
        if self.nears_fars is not None:
            self.nears_fars[index] = cast(TTensor_co, value.nears_fars)
        if self.metadata is not None:
            self.metadata[index] = cast(TTensor_co, value.metadata)

    def __iter__(self) -> Iterator[Self]:
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def cat(cls, values: Sequence[Self]) -> Self:
        xnp = _get_xnp(values[0].poses)
        nears_fars: Optional[TTensor_co] = None
        metadata: Optional[TTensor_co] = None
        if any(v.nears_fars is not None for v in values):
            assert all(v.nears_fars is not None for v in values), "Either all or none of the cameras must have nears and fars"
            nears_fars = xnp.concatenate([cast(TTensor_co, v.nears_fars) for v in values])
        if any(v.metadata is not None for v in values):
            assert all(v.metadata is not None for v in values), "Either all or none of the cameras must have metadata"
            metadata = xnp.concatenate([cast(TTensor_co, v.metadata) for v in values])
        return cls(
            poses=xnp.concatenate([v.poses for v in values]),
            intrinsics=xnp.concatenate([v.intrinsics for v in values]),
            camera_models=xnp.concatenate([v.camera_models for v in values]),
            distortion_parameters=xnp.concatenate([v.distortion_parameters for v in values]),
            image_sizes=xnp.concatenate([cast(TTensor_co, v.image_sizes) for v in values]),
            nears_fars=nears_fars,
            metadata=metadata,
        )

    def replace(self, **changes) -> Self:
        return dataclasses.replace(self, **changes)

    def apply(self, fn: Callable[[TTensor_co, str], TTensor]) -> 'GenericCamerasImpl[TTensor]':
        return GenericCamerasImpl[TTensor](
            poses=fn(self.poses, "poses"),
            intrinsics=fn(self.intrinsics, "intrinsics"),
            camera_models=fn(self.camera_models, "camera_models"),
            distortion_parameters=fn(self.distortion_parameters, "distortion_parameters"),
            image_sizes=fn(self.image_sizes, "image_sizes"),
            nears_fars=fn(cast(TTensor_co, self.nears_fars), "nears_fars") if self.nears_fars is not None else None,
            metadata=fn(cast(TTensor_co, self.metadata), "metadata") if self.metadata is not None else None,
        )


def new_cameras(
    *,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    camera_models: np.ndarray,
    distortion_parameters: np.ndarray,
    image_sizes: np.ndarray,
    nears_fars: Optional[np.ndarray] = None,
    metadata: Optional[np.ndarray] = None,
) -> Cameras:
    return GenericCamerasImpl[np.ndarray](
        poses=poses,
        intrinsics=intrinsics,
        camera_models=camera_models,
        distortion_parameters=distortion_parameters,
        image_sizes=image_sizes,
        nears_fars=nears_fars,
        metadata=metadata)
    

class _IncompleteDataset(TypedDict, total=True):
    cameras: Cameras  # [N]

    image_paths: List[str]
    image_paths_root: str
    sampling_mask_paths: Optional[List[str]]
    sampling_mask_paths_root: Optional[str]
    metadata: Dict
    sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]]  # [N][H, W]
    points3D_xyz: Optional[np.ndarray]  # [M, 3]
    points3D_rgb: Optional[np.ndarray]  # [M, 3]
    images_points3D_indices: Optional[List[np.ndarray]]  # [N][<M]


class UnloadedDataset(_IncompleteDataset):
    images: NotRequired[Optional[Union[np.ndarray, List[np.ndarray]]]]  # [N][H, W, 3]


class Dataset(_IncompleteDataset):
    images: Union[np.ndarray, List[np.ndarray]]  # [N][H, W, 3]


class RenderOutput(TypedDict, total=False):
    color: Required[np.ndarray]  # [h w 3]
    raw_color: np.ndarray  # [h w 3]
    depth: np.ndarray  # [h w]
    accumulation: np.ndarray  # [h w]


class OptimizeEmbeddingOutput(TypedDict):
    embedding: np.ndarray
    metrics: NotRequired[Dict[str, Sequence[float]]]


class MethodInfo(TypedDict, total=False):
    method_id: Required[str]
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet


class ModelInfo(TypedDict, total=False):
    method_id: Required[str]
    num_iterations: Required[int]
    loaded_step: Optional[int]
    loaded_checkpoint: Optional[str]
    batch_size: int
    eval_batch_size: int
    required_features: FrozenSet[DatasetFeature]
    supported_camera_models: FrozenSet
    hparams: Dict[str, Any]


@runtime_checkable
class Method(Protocol):
    def __init__(self, 
                 *,
                 checkpoint: Union[str, None] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[Dict[str, Any]] = None):
        pass

    @classmethod
    def install(cls):
        """
        Install the method.
        """
        pass

    @classmethod
    @abstractmethod
    def get_method_info(cls) -> MethodInfo:
        """
        Get method info needed to initialize the datasets.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """
        Get method defaults for the trainer.

        Returns:
            Method info.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for the given image index.

        Args:
            index: Image index.

        Returns:
            Image embedding.
        """
        return None

    @abstractmethod
    def optimize_embedding(self, dataset: Dataset, *, embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        """
        Optimize embeddings for single image (passed as dataset slice).

        Args:
            dataset: Dataset.
            embedding: Optional initial embedding.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, camera: Cameras, *, options: Optional[Dict] = None) -> RenderOutput:  # [h w c]
        """
        Render images.

        Args:
            cameras: Cameras.
            options: Render options
        """
        raise NotImplementedError()

    @abstractmethod
    def train_iteration(self, step: int):
        """
        Train one iteration.

        Args:
            step: Current step.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save.
        """
        raise NotImplementedError()


@runtime_checkable
class EvaluationProtocol(Protocol):
    def get_name(self) -> str:
        ...
        
    def render(self, method: Method, dataset: Dataset) -> RenderOutput:
        ...

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        ...

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        ...


class DatasetSpecMetadata(TypedDict, total=False):
    id: str
    name: str
    description: str
    paper_title: str
    paper_authors: List[str]
    paper_link: str
    link: str
    metrics: List[str]
    default_metric: str
    scenes: List[Dict[str, str]]


class LoadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 split: str, 
                 features: Optional[FrozenSet[DatasetFeature]] = None, 
                 **kwargs) -> UnloadedDataset:
        ...


class DownloadDatasetFunction(Protocol):
    def __call__(self, 
                 path: str, 
                 output: str) -> None:
        ...


class TrajectoryFrameAppearance(TypedDict, total=False):
    embedding: Optional[np.ndarray]
    embedding_train_index: Optional[int]


class TrajectoryFrame(TypedDict, total=True):
    pose: np.ndarray
    intrinsics: np.ndarray
    appearance_weights: NotRequired[np.ndarray]


class TrajectoryKeyframe(TypedDict, total=True):
    pose: np.ndarray
    fov: Optional[float]
    transition_duration: NotRequired[Optional[float]]
    appearance: NotRequired[TrajectoryFrameAppearance]


TrajectoryInterpolationType = Literal["kochanek-bartels", "none"]


class ImageSetInterpolationSource(TypedDict, total=True):
    type: Literal["interpolation"]
    interpolation: Literal["none"]
    keyframes: List[TrajectoryKeyframe]
    default_fov: float
    default_transition_duration: float
    default_appearance: NotRequired[Optional[TrajectoryFrameAppearance]]


class KochanekBartelsInterpolationSource(TypedDict, total=True):
    type: Literal["interpolation"]
    interpolation: Literal["kochanek-bartels"]
    is_cycle: bool
    tension: float
    keyframes: List[TrajectoryKeyframe]
    default_fov: float
    default_transition_duration: float
    default_appearance: NotRequired[Optional[TrajectoryFrameAppearance]]


TrajectoryInterpolationSource = Union[ImageSetInterpolationSource, KochanekBartelsInterpolationSource]


class Trajectory(TypedDict, total=True):
    camera_model: CameraModel
    image_size: Tuple[int, int]
    frames: List[TrajectoryFrame]
    appearances: NotRequired[List[TrajectoryFrameAppearance]]
    fps: float
    source: NotRequired[Optional[TrajectoryInterpolationSource]]


@runtime_checkable
class LoggerEvent(Protocol):
    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        ...

    def add_text(self, tag: str, text: str) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, display_name: Optional[str] = None, description: Optional[str] = None, **kwargs) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...

    def add_plot(self, tag: str, *data: np.ndarray,
                 axes_labels: Optional[Sequence[str]] = None, 
                 title: Optional[str] = None,
                 **kwargs) -> None:
        ...

    def add_histogram(self, tag: str, values: np.ndarray, *, num_bins: Optional[int] = None) -> None:
        ...


@runtime_checkable
class Logger(Protocol):
    def add_event(self, step: int) -> typing.ContextManager[LoggerEvent]:
        ...

    def add_scalar(self, tag: str, value: Union[float, int], step: int) -> None:
        ...

    def add_text(self, tag: str, text: str, step: int) -> None:
        ...

    def add_image(self, tag: str, image: np.ndarray, step: int, *, display_name: Optional[str] = None, description: Optional[str] = None) -> None:
        ...

    def add_embedding(self, tag: str, embeddings: np.ndarray, step: int, *, 
                      images: Optional[List[np.ndarray]] = None, 
                      labels: Union[None, List[Dict[str, str]], List[str]] = None) -> None:
        ...
import warnings
import logging
import os
import struct
import numpy as np
import PIL.Image
import PIL.ExifTags
from tqdm import tqdm
from typing import Optional, TypeVar, Tuple, Union, List, Sequence, Dict, cast, overload


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
    )
    return np.eye(3, dtype=a.dtype) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def viewmatrix(
    lookdir,
    up,
    position,
    lock_up = False,
):
    """Construct lookat view matrix."""
    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def orthogonal_dir(a, b): 
        return normalize(np.cross(a, b))

    vecs = [None, normalize(up), normalize(lookdir)]
    # x-axis is always the normalized cross product of `lookdir` and `up`.
    vecs[0] = orthogonal_dir(vecs[1], vecs[2])
    # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
    ax = 2 if lock_up else 1
    # Set the not-locked axis to be orthogonal to the other two.
    vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
    m = np.stack(vecs + [position], axis=1)
    return m


TDataset = TypeVar("TDataset", bound=Union[Dataset, UnloadedDataset])


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def get_transform_poses_pca(poses):
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is positive
    if poses_recentered.mean(axis=0)[2, 1] > 0:
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return transform


def focus_point_fn(poses, xnp = np):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = xnp.eye(3) - directions * xnp.transpose(directions, [0, 2, 1])
    mt_m = xnp.transpose(m, [0, 2, 1]) @ m
    focus_pt = xnp.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def get_default_viewer_transform(poses, dataset_type: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_type == "object-centric":
        transform = get_transform_poses_pca(poses)

        poses = apply_transform(transform, poses)
        lookat = focus_point_fn(poses)

        poses[:, :3, 3] -= lookat
        transform[:3, 3] -= lookat
        return transform, poses[0][..., :3, :4]

    elif dataset_type == "forward-facing":
        raise NotImplementedError("Forward-facing dataset type is not supported")
    elif dataset_type is None:
        # Unknown dataset type
        # We move all center the scene on the mean of the camera origins
        # and reorient the scene so that the average camera up is up
        origins = poses[..., :3, 3]
        mean_origin = np.mean(origins, 0)
        translation = mean_origin
        up = np.mean(poses[:, :3, 1], 0)
        up = -up / np.linalg.norm(up)

        rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
        transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)

        # Scale so that cameras fit in a 2x2x2 cube centered at the origin
        maxlen = np.quantile(np.abs(poses[..., 0:3, 3] - mean_origin[None]).max(-1), 0.95) * 1.1
        dataparser_scale = float(1 / maxlen)
        transform = np.diag([dataparser_scale, dataparser_scale, dataparser_scale, 1]) @ transform

        camera = apply_transform(transform, poses[0])
        return transform, camera[..., :3, :4]
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


METADATA_COLUMNS = ["exposure"]
DatasetType = Literal["object-centric", "forward-facing"]


def get_scene_scale(cameras: Cameras, dataset_type: Optional[DatasetType]):
    if dataset_type == "object-centric":
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))

    elif dataset_type == "forward-facing":
        assert cameras.nears_fars is not None, "Forward-facing dataset must set z-near and z-far"
        return float(cameras.nears_fars.mean())

    elif dataset_type is None:
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))
    
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def get_image_metadata(image: PIL.Image.Image):
    # Metadata format: [ exposure, ]
    values = {}
    try:
        exif_pil = image.getexif()
    except AttributeError:
        exif_pil = image._getexif()  # type: ignore
    if exif_pil is not None:
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in PIL.ExifTags.TAGS}
        if "ExposureTime" in exif and "ISOSpeedRatings" in exif:
            shutters = exif["ExposureTime"]
            isos = exif["ISOSpeedRatings"]
            exposure = shutters * isos / 1000.0
            values["exposure"] = exposure
    return np.array([values.get(c, np.nan) for c in METADATA_COLUMNS], dtype=np.float32)


def _dataset_rescale_intrinsics(dataset: Dataset, image_sizes: np.ndarray):
    cameras = dataset["cameras"]
    if np.any(cameras.image_sizes != image_sizes):
        logging.info("Image sizes do not match camera sizes. Resizing cameras to match image sizes.")

        if np.any(cameras.image_sizes % image_sizes != 0):
            warnings.warn("Downscaled image sizes are not a multiple of camera sizes.")

        multx, multy = np.moveaxis(
            image_sizes.astype(np.float64) / cameras.image_sizes.astype(np.float64), -1, 0)

        if "downscale_factor" in dataset["metadata"]:
            # Downscale factor is passed, we will use it for focal lengths
            # Not for the center of the image, because there could have been rounding
            # which would move the center from the center of the image
            downscale_factor = dataset["metadata"]["downscale_factor"]
            low = np.floor(cameras.image_sizes * np.stack([multx, multy], -1))
            high = np.ceil(cameras.image_sizes * np.stack([multx, multy], -1))
            if np.any(image_sizes < low) or np.any(image_sizes > high):
                raise RuntimeError(f"Downscaled image sizes do not match the downscale_factor of {downscale_factor}.")

        # NOTE: In previous versions of NerfBaselines, we scaled the parameters differently
        # We used:
        #   cx <- cx * multx,  cy <- cy * multy
        #   fx <- fx * multx,  fy <- fy * multx
        # This renders changes slightly the results on the MipNeRF 360 dataset

        multipliers = np.stack([multx, multy, multx, multy], -1)
        dataset["cameras"] = cameras.replace(
            image_sizes=image_sizes, 
            intrinsics=(cameras.intrinsics * multipliers).astype(cameras.intrinsics.dtype))


def dataset_load_features(
    dataset: UnloadedDataset, features=None, supported_camera_models=None
) -> Dataset:
    if features is None:
        features = frozenset(("color",))
    if supported_camera_models is None:
        supported_camera_models = frozenset(("pinhole",))
    images: List[np.ndarray] = []
    image_sizes = []
    all_metadata = []
    resize = dataset["metadata"].get("downscale_loaded_factor")
    if resize == 1:
        resize = None

    i = 0
    logging.info(f"Loading images from {dataset.get('image_paths_root')}")

    for p in tqdm(dataset["image_paths"], desc="loading images", dynamic_ncols=True):
        if str(p).endswith(".bin"):
            assert dataset["metadata"]["color_space"] == "linear"
            with open(p, "rb") as f:
                data_bytes = f.read()
                h, w = struct.unpack("ii", data_bytes[:8])
                image = (
                    np.frombuffer(
                        data_bytes, dtype=np.float16, count=h * w * 4, offset=8
                    )
                    .astype(np.float32)
                    .reshape([h, w, 4])
                )
            metadata = np.array(
                [np.nan for _ in range(len(METADATA_COLUMNS))], dtype=np.float32
            )
        else:
            assert dataset["metadata"]["color_space"] == "srgb"
            pil_image = PIL.Image.open(p)
            metadata = get_image_metadata(pil_image)
            if resize is not None:
                w, h = pil_image.size
                new_size = round(w/resize), round(h/resize)
                pil_image = pil_image.resize(new_size, PIL.Image.Resampling.BICUBIC)
                warnings.warn(f"Resized image with a factor of {resize}")

            image = np.array(pil_image, dtype=np.uint8)
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])
        all_metadata.append(metadata)
        i += 1

    logging.debug(f"Loaded {len(images)} images")

    if dataset["sampling_mask_paths"] is not None:
        sampling_masks = []
        for p in tqdm(dataset["sampling_mask_paths"], desc="loading sampling masks", dynamic_ncols=True):
            sampling_mask = PIL.Image.open(p).convert("L")
            if resize is not None:
                w, h = sampling_mask.size
                new_size = round(w*resize), round(h*resize)
                sampling_mask = sampling_mask.resize(new_size, PIL.Image.Resampling.NEAREST)
                warnings.warn(f"Resized sampling mask with a factor of {resize}")

            sampling_masks.append(np.array(sampling_mask, dtype=np.uint8).astype(bool))
        dataset["sampling_masks"] = sampling_masks  # padded_stack(sampling_masks)
        logging.debug(f"Loaded {len(sampling_masks)} sampling masks")

    if resize is not None:
        # Replace all paths with the resized paths
        dataset["image_paths"] = [
            os.path.join("/resized", os.path.relpath(p, dataset["image_paths_root"])) 
            for p in dataset["image_paths"]]
        dataset["image_paths_root"] = "/resized"
        if dataset["sampling_mask_paths"] is not None:
            dataset["sampling_mask_paths"] = [
                os.path.join("/resized-sampling-masks", os.path.relpath(p, dataset["sampling_mask_paths_root"])) 
                for p in dataset["sampling_mask_paths"]]
            dataset["sampling_mask_paths_root"] = "/resized-sampling-masks"

    dataset["images"] = images  # padded_stack(images)

    # Replace image sizes and metadata
    image_sizes = np.array(image_sizes, dtype=np.int32)

    _dataset_rescale_intrinsics(cast(Dataset, dataset), image_sizes)

    if supported_camera_models is not None and supported_camera_models != set(("pinhole",)):
        raise RuntimeError(
            "Some cameras models are not supported by the method."
        )
    return cast(Dataset, dataset)


class DatasetNotFoundError(Exception):
    pass


class MultiDatasetError(DatasetNotFoundError):
    def __init__(self, errors, message):
        self.errors = errors
        self.message = message
        super().__init__(message + "\n" + "".join(f"\n  {name}: {error}" for name, error in errors.items()))

    def write_to_logger(self, color=True, terminal_width=None):
        if terminal_width is None:
            terminal_width = 120
            try:
                terminal_width = min(os.get_terminal_size().columns, 120)
            except OSError:
                pass
        message = self.message
        if color:
            message = "\33[0m\33[31m" + message + "\33[0m"
        for name, error in self.errors.items():
            prefix = f"   {name}: "
            mlen = terminal_width - len(prefix)
            prefixlen = len(prefix)
            if color:
                prefix = f"\33[96m{prefix}\33[0m"
            rows = [error[i : i + mlen] for i in range(0, len(error), mlen)]
            mdetail = f'\n{" "*prefixlen}'.join(rows)
            message += f"\n{prefix}{mdetail}"
        logging.error(message)


def dataset_index_select(dataset: TDataset, i: Union[slice, int, list, np.ndarray]) -> TDataset:
    assert isinstance(i, (slice, int, list, np.ndarray))
    dataset_len = len(dataset["image_paths"])

    def index(key, obj):
        if obj is None:
            return None
        if key == "cameras":
            if len(obj) == 1:
                return obj if isinstance(i, int) else obj
            return obj[i]
        if isinstance(obj, np.ndarray):
            if obj.shape[0] == 1:
                return obj[0] if isinstance(i, int) else obj
            obj = obj[i]
            return obj
        if isinstance(obj, list):
            indices = np.arange(dataset_len)[i]
            if indices.ndim == 0:
                return obj[indices]
            return [obj[i] for i in indices]
        raise ValueError(f"Cannot index object of type {type(obj)} at key {key}")

    _dataset = cast(Dict, dataset.copy())
    _dataset.update({k: index(k, v) for k, v in dataset.items() if k not in {
        "image_paths_root", 
        "sampling_mask_paths_root", 
        "points3D_xyz", 
        "points3D_rgb", 
        "metadata"}})
    return cast(TDataset, _dataset)


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Union[np.ndarray, List[np.ndarray]],
                sampling_mask_paths: Optional[Sequence[str]] = ...,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> Dataset:
    ...


@overload
def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = ...,
                images: Literal[None] = None,
                sampling_mask_paths: Optional[Sequence[str]] = ...,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = ...,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = ...,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = ...,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> UnloadedDataset:
    ...


def new_dataset(*,
                cameras: Cameras,
                image_paths: Sequence[str],
                image_paths_root: Optional[str] = None,
                images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W, 3]
                sampling_mask_paths: Optional[Sequence[str]] = None,
                sampling_mask_paths_root: Optional[str] = None,
                sampling_masks: Optional[Union[np.ndarray, List[np.ndarray]]] = None,  # [N][H, W]
                points3D_xyz: Optional[np.ndarray] = None,  # [M, 3]
                points3D_rgb: Optional[np.ndarray] = None,  # [M, 3]
                images_points3D_indices: Optional[Sequence[np.ndarray]] = None,  # [N][<M]
                metadata: Dict) -> Union[UnloadedDataset, Dataset]:
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    if sampling_mask_paths_root is None and sampling_mask_paths is not None:
        sampling_mask_paths_root = os.path.commonpath(sampling_mask_paths)
    if image_paths_root is None:
        image_paths_root = os.path.commonpath(image_paths)
    return UnloadedDataset(
        cameras=cameras,
        image_paths=list(image_paths),
        sampling_mask_paths=list(sampling_mask_paths) if sampling_mask_paths is not None else None,
        sampling_mask_paths_root=sampling_mask_paths_root,
        image_paths_root=image_paths_root,
        images=images,
        sampling_masks=sampling_masks,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        images_points3D_indices=list(images_points3D_indices) if images_points3D_indices is not None else None,
        metadata=metadata
    )


def get_transform_and_scale(transform):
    assert len(transform.shape) == 2, "Transform should be a 4x4 or a 3x4 matrix."
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0], rtol=1e-3, atol=0)
    scale = float(np.mean(scale).item())
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def apply_transform(transform, poses):
    transform, scale = get_transform_and_scale(transform)
    poses = unpad_poses(transform @ pad_poses(poses))
    poses[..., :3, 3] *= scale
    return poses


def invert_transform(transform, has_scale=False):
    scale = None
    if has_scale:
        transform, scale = get_transform_and_scale(transform)
    else:
        transform = transform.copy()
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    transform[..., :3, :] = np.concatenate([R.T, -np.matmul(R.T, t[..., None])], axis=-1)
    if scale is not None:
        transform[..., :3, :3] *= 1/scale
    return transform

def _padded_stack(tensors: Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]) -> np.ndarray:
    if not isinstance(tensors, (tuple, list)):
        return tensors
    max_shape = tuple(max(s) for s in zip(*[x.shape for x in tensors]))
    out_tensors = []
    for x in tensors:
        pads = [(0, m - s) for s, m in zip(x.shape, max_shape)]
        out_tensors.append(np.pad(x, pads))
    return np.stack(out_tensors, 0)


def _parse_colmap_camera_params(camera: Camera) -> Tuple[np.ndarray, int, np.ndarray, Tuple[int, int]]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    out = OrderedDict()  # Default in Python 3.7+
    camera_params = camera.params
    camera_model: CameraModel
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        camera_model = "pinhole"
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        camera_model = "pinhole"
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        camera_model = "opencv"
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        camera_model = "opencv"
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = "opencv"
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = "opencv_fisheye"
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        camera_model = "opencv_fisheye"
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = "opencv_fisheye"
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    image_width: int = camera.width
    image_height: int = camera.height
    intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32)
    distortion_params = np.array([out.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4")], dtype=np.float32)
    return intrinsics, camera_model_to_int(camera_model), distortion_params, (image_width, image_height)


def load_colmap_dataset(path: Union[Path, str],
        images_path: Optional[Union[str, Path]] = None, 
        split: Optional[str] = None, 
        test_indices: Optional[Indices] = None,
        features: Optional[FrozenSet[DatasetFeature]] = None,
        colmap_path: Optional[Union[str, Path]] = None):
    path = Path(path)
    colmap_path = Path(colmap_path) if colmap_path is not None else None
    if features is None:
        features = typing.cast(FrozenSet[DatasetFeature], {})
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}
    # Load COLMAP dataset
    if colmap_path is None:
        colmap_path = Path("sparse") / "0"
        if not (path / colmap_path).exists():
            colmap_path = Path("sparse")
    colmap_path = path / colmap_path
    if images_path is None:
        images_path = Path("images")
    images_path = path / images_path
    if not colmap_path.exists():
        raise DatasetNotFoundError("Missing 'sparse/0' folder in COLMAP dataset")
    if not (colmap_path / "cameras.bin").exists() and not (colmap_path / "cameras.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")
    if not images_path.exists():
        raise DatasetNotFoundError("Missing 'images' folder in COLMAP dataset")

    if (colmap_path / "cameras.bin").exists():
        colmap_cameras = read_cameras_binary(colmap_path / "cameras.bin")
    elif (colmap_path / "cameras.txt").exists():
        colmap_cameras = read_cameras_text(colmap_path / "cameras.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")

    if not (colmap_path / "images.bin").exists() and not (colmap_path / "images.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")
    if (colmap_path / "images.bin").exists():
        images = read_images_binary(colmap_path / "images.bin")
    elif (colmap_path / "images.txt").exists():
        images = read_images_text(colmap_path / "images.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")

    points3D: Optional[Dict[int, Point3D]] = None
    if load_points:
        if not (colmap_path / "points3D.bin").exists() and not (colmap_path / "points3D.txt").exists():
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")
        if (colmap_path / "points3D.bin").exists():
            points3D = read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            points3D = read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")

    # Convert to tensors
    camera_intrinsics = []
    camera_poses = []
    camera_models = []
    camera_distortion_params = []
    image_paths: List[str] = []
    image_names = []
    camera_sizes = []

    image: Image
    i = 0
    c2w: np.ndarray
    for image in images.values():
        camera: Camera = colmap_cameras[image.camera_id]
        intrinsics, camera_model, distortion_params, (w, h) = _parse_colmap_camera_params(camera)
        camera_sizes.append(np.array((w, h), dtype=np.int32))
        camera_intrinsics.append(intrinsics)
        camera_models.append(camera_model)
        camera_distortion_params.append(distortion_params)
        image_names.append(image.name)
        image_paths.append(str(images_path / image.name))

        rotation = qvec2rotmat(image.qvec).astype(np.float64)

        translation = image.tvec.reshape(3, 1).astype(np.float64)
                
        # flip_y is a 3×3 matrix that negates the y-axis
        flip = np.array([
            [1,  0,  0],   # X stays the same
            [0, -1,  0],   # Flip Y
            [0,  0, -1],   # Flip Z
        ], dtype=np.float32)

        # Apply the flip on rotation and translation.
        # Multiply from the left on rotation. For translation, you can multiply directly.
        rotation_flipped = flip.T @ rotation
        translation_flipped = flip.T @ translation

        # If your original w2c is [R | -t], then it becomes:
        w2c = np.concatenate([rotation_flipped, translation_flipped], axis=1)
        #w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=w2c.dtype)], 0)
        c2w = np.linalg.inv(w2c)

        camera_poses.append(c2w[0:3, :])
        i += 1

    # Estimate nears fars
    near = 0.01
    far = np.stack([x[:3, -1] for x in camera_poses], 0)
    far = float(np.percentile(np.linalg.norm(far - np.mean(far, keepdims=True, axis=0), axis=-1), 90, axis=0))
    nears_fars = np.array([[near, far]] * len(camera_poses), dtype=np.float32)

    # Load points
    points3D_xyz = None
    points3D_rgb = None
    if load_points:
        assert points3D is not None, "3D points have not been loaded"
        points3D_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points3D_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

    # camera_ids=torch.tensor(camera_ids, dtype=torch.int32),
    all_cameras = new_cameras(
        poses=np.stack(camera_poses, 0).astype(np.float32),
        intrinsics=np.stack(camera_intrinsics, 0).astype(np.float32),
        camera_models=np.array(camera_models, dtype=np.int32),
        distortion_parameters=_padded_stack(camera_distortion_params).astype(np.float32),
        image_sizes=np.stack(camera_sizes, 0).astype(np.int32),
        nears_fars=nears_fars.astype(np.float32),
    )
    indices = None
    train_indices = np.arange(len(image_paths))
    if split is not None:
        if test_indices is None and ((path / "train_list.txt").exists() or (path / "test_list.txt").exists()):
            logging.info(f"colmap dataloader is loading split data from {path / f'{split}_list.txt'}")
            train_indices = None
            for split in ("train", split):
                split_image_names = set((path / f"{split}_list.txt").read_text().splitlines())
                indices = np.array([name in split_image_names for i, name in enumerate(image_names)], dtype=bool)
                if indices.sum() == 0:
                    raise DatasetNotFoundError(f"no images found for split {split} in {path / f'{split}_list.txt'}")
                if indices.sum() < len(split_image_names):
                    logging.warning(f"only {indices.sum()} images found for split {split} in {path / f'{split}_list.txt'}")
                if split == "train":
                    train_indices = indices
            assert train_indices is not None
        else:
            if test_indices is None:
                test_indices = Indices.every_iters(8)
            dataset_len = len(image_paths)
            test_indices.total = dataset_len
            test_indices_array: np.ndarray = np.array([i in test_indices for i in range(dataset_len)], dtype=bool)
            train_indices = np.logical_not(test_indices_array)
            indices = train_indices if split == "train" else test_indices_array

    viewer_transform, viewer_pose = get_default_viewer_transform(all_cameras[train_indices].poses, None)
    dataset = new_dataset(
        cameras=all_cameras,
        image_paths=image_paths,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        sampling_mask_paths=None,
        image_paths_root=str(images_path),
        metadata={
            "name": None,
            "color_space": "srgb",
            "viewer_transform": viewer_transform,
            "viewer_initial_pose": viewer_pose,
        })
    if indices is not None:
        dataset = dataset_index_select(dataset, indices)

    return dataset
