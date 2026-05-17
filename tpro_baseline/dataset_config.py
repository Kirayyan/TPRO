# Grid settings aligned with MST-OATD preprocess (porto / cd).

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
  name: str
  lat_grid_num: int
  lng_grid_num: int
  lon_blocks: int = 10
  lat_blocks: int = 20
  top_k: int = 5
  grid_only: bool = False  # if True, TPRO runs on grid cells (no road map)


PORTO = DatasetConfig(
    name="porto",
    lat_grid_num=51,
    lng_grid_num=119,
    grid_only=False,
)

CD = DatasetConfig(
    name="cd",
    lat_grid_num=167,
    lng_grid_num=154,
    grid_only=True,
)


def get_config(dataset: str) -> DatasetConfig:
  d = dataset.lower().strip()
  if d in ("cd", "chengdu"):
    return CD
  if d in ("porto", "pt"):
    return PORTO
  raise ValueError(f"unknown dataset {dataset!r}, use porto or cd")


def grid_id_to_ij(grid_id: int, lng_grid_num: int) -> tuple[int, int]:
  gi = int(grid_id) // lng_grid_num
  gj = int(grid_id) % lng_grid_num
  return gi, gj


def ij_to_group(gi: int, gj: int, cfg: DatasetConfig) -> int:
  row = min(int(gi / (cfg.lat_grid_num / cfg.lat_blocks)), cfg.lat_blocks - 1)
  col = min(int(gj / (cfg.lng_grid_num / cfg.lon_blocks)), cfg.lon_blocks - 1)
  return row * cfg.lon_blocks + col
