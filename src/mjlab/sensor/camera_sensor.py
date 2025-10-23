from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mujoco_warp._src.render_context import RenderContext

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.sim.sim_data import TorchArray

CameraDataType = Literal["rgb", "depth"]


@wp.kernel
def unpack_rgb_kernel(
  packed: wp.array3d(dtype=wp.uint32),  # type: ignore (nworld, ncam, height*width)
  width: int,
  height: int,
  rgb: wp.array4d(dtype=wp.uint8),  # type: ignore (nworld, ncam, height*width, 3)
):
  """Unpack packed uint32 RGB into uint8 buffer for all cameras."""
  nworld_idx, cam_idx, pixel_idx = wp.tid()  # type: ignore

  if pixel_idx < width * height:
    packed_val = packed[nworld_idx, cam_idx, pixel_idx]

    r = wp.uint8(packed_val & wp.uint32(0xFF))
    g = wp.uint8((packed_val >> wp.uint32(8)) & wp.uint32(0xFF))
    b = wp.uint8((packed_val >> wp.uint32(16)) & wp.uint32(0xFF))

    rgb[nworld_idx, cam_idx, pixel_idx, 0] = r
    rgb[nworld_idx, cam_idx, pixel_idx, 1] = g
    rgb[nworld_idx, cam_idx, pixel_idx, 2] = b


@dataclass
class CameraSensorCfg(SensorCfg):
  """Configuration for a camera sensor."""

  width: int
  height: int
  type: tuple[CameraDataType, ...] = ("rgb",)
  use_textures: bool = True
  use_shadows: bool = True
  fov_rad: float = math.radians(120.0)
  enabled_geom_groups: tuple[int, ...] = (0, 1, 2)

  def build(self) -> CameraSensor:
    return CameraSensor(self)


@dataclass
class CameraSensorData:
  """Data structure for camera sensor data."""

  output: dict[str, dict[str, torch.Tensor | TorchArray]]
  """Structure: {camera_name: {"rgb": tensor, "depth": tensor}}"""


class CameraSensor(Sensor[CameraSensorData]):
  """Camera sensor implementation."""

  def __init__(self, cfg: CameraSensorCfg) -> None:
    self.cfg = cfg

    self._model: mjwarp.Model | None = None
    self._data: mjwarp.Data | None = None
    self._device: str | None = None
    self._ctx: RenderContext | None = None
    self._rgb_unpacked: wp.array4d | None = None
    self._cam_names: dict[int, str] = {}
    self._ncam: int = 0

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    # TODO(kevin): Add desired camera.
    pass

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    self._model = model
    self._data = data
    self._device = device

    self._ctx = mjwarp.create_render_context(
      mjm=mj_model,
      m=model.struct,  # type: ignore
      d=data.struct,  # type: ignore
      nworld=data.nworld,
      width=self.cfg.width,
      height=self.cfg.height,
      fov_rad=self.cfg.fov_rad,
      use_textures=self.cfg.use_textures,
      use_shadows=self.cfg.use_shadows,
      render_rgb="rgb" in self.cfg.type,
      render_depth="depth" in self.cfg.type,
      enabled_geom_groups=list(self.cfg.enabled_geom_groups),
    )

    self._ncam = mj_model.ncam
    assert self._ncam > 0, "No cameras found in the MuJoCo model."
    self._cam_names = {i: mj_model.cam(i).name for i in range(self._ncam)}

    wp_device = device if device.startswith("cuda") else "cpu"
    if "rgb" in self.cfg.type:
      self._rgb_unpacked = wp.array4d(
        shape=(data.nworld, self._ncam, self.cfg.height * self.cfg.width, 3),
        dtype=wp.uint8,
        device=wp_device,
      )

  @property
  def data(self) -> CameraSensorData:
    assert self._model is not None
    assert self._data is not None
    assert self._device is not None
    assert self._ctx is not None
    assert self._cam_names is not None

    mjwarp.render(self._model, self._data, self._ctx)

    if "rgb" in self.cfg.type:
      assert self._rgb_unpacked is not None
      wp.launch(
        unpack_rgb_kernel,
        dim=(self._data.nworld, self._ncam, self.cfg.height * self.cfg.width),
        inputs=[self._ctx.pixels, self.cfg.width, self.cfg.height],
        outputs=[self._rgb_unpacked],
        device=self._rgb_unpacked.device,
      )

    output: dict[str, dict[str, torch.Tensor | TorchArray]] = {}
    for cam_idx in range(self._ncam):
      cam_name = self._cam_names.get(cam_idx, f"camera_{cam_idx}")
      cam_output: dict[str, torch.Tensor | TorchArray] = {}

      if "rgb" in self.cfg.type:
        assert self._rgb_unpacked is not None
        # Convert to torch first, then reshape (warp doesn't support reshaping slices)
        rgb_cam_flat = wp.to_torch(self._rgb_unpacked[:, cam_idx, :, :])
        rgb_cam = rgb_cam_flat.reshape(
          self._data.nworld, self.cfg.height, self.cfg.width, 3
        )
        cam_output["rgb"] = rgb_cam

      if "depth" in self.cfg.type:
        # Convert to torch first, then reshape
        depth_cam_flat = wp.to_torch(self._ctx.depth[:, cam_idx, :])
        depth_cam = depth_cam_flat.reshape(
          self._data.nworld, self.cfg.height, self.cfg.width, 1
        )
        cam_output["depth"] = depth_cam

      output[cam_name] = cam_output

    return CameraSensorData(output=output)
