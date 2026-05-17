# TPRO on grid-cell trajectories (no road network). For MST-OATD porto / cd .npy data.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .dataset_config import DatasetConfig, grid_id_to_ij, ij_to_group
from .tpro import RouteRecord, _route_popularity_key, normalized_edit_distance


def traj_to_grid_seq(traj: Sequence) -> List[int]:
  return [int(p[0]) for p in traj]


def trajs_to_grid_seqs(trajs: Sequence) -> List[List[int]]:
  return [traj_to_grid_seq(t) for t in trajs]


class GridTPRO:
  """TPRO popular-route model on grid ID sequences (MST-OATD format)."""

  def __init__(self, cfg: DatasetConfig):
    self.cfg = cfg
    self.max_cell = cfg.lat_grid_num * cfg.lng_grid_num + 1
    self.top_k_data: Dict[int, Dict[int, List[RouteRecord]]] = {}

  def _group_from_cell(self, grid_id: int) -> int:
    gi, gj = grid_id_to_ij(grid_id, self.cfg.lng_grid_num)
    return ij_to_group(gi, gj, self.cfg)

  def fit(self, grid_routes: Sequence[Sequence[int]]) -> None:
    cfg = self.cfg
    group_data: Dict[int, Dict[int, List[RouteRecord]]] = {}
    route_id = 0

    for seq in grid_routes:
      if len(seq) < 2:
        continue
      start_g = self._group_from_cell(seq[0])
      end_g = self._group_from_cell(seq[-1])
      rec = RouteRecord(
          edges=list(seq),
          points=list(seq),
          route_id=route_id,
          route_length=float(len(seq)),
      )
      route_id += 1
      group_data.setdefault(start_g, {}).setdefault(end_g, []).append(rec)

    self.top_k_data = {}
    for start_gid, end_map in group_data.items():
      self.top_k_data.setdefault(start_gid, {})
      for end_gid, group in end_map.items():
        times_count = [0] * self.max_cell
        count_last_change = [-1] * self.max_cell
        change_time = 0
        for rec in group:
          change_time += 1
          for cell in rec.points:
            if count_last_change[cell] != change_time:
              count_last_change[cell] = change_time
              times_count[cell] += 1

        for rec in group:
          change_time += 1
          pop = []
          for cell in rec.points:
            if count_last_change[cell] != change_time:
              count_last_change[cell] = change_time
              pop.append(times_count[cell])
          pop.sort()
          rec.popularity_data = pop
          rec.popularity_sum = sum(pop)

        group.sort(key=_route_popularity_key)
        seen_lengths: set = set()
        top: List[RouteRecord] = []
        for rec in group:
          if len(top) >= cfg.top_k:
            break
          if rec.route_length in seen_lengths:
            continue
          seen_lengths.add(rec.route_length)
          top.append(rec)
        self.top_k_data[start_gid][end_gid] = top

  def anomaly_scores(self, grid_routes: Sequence[Sequence[int]]) -> List[float]:
    scores: List[float] = []
    for seq in grid_routes:
      if len(seq) < 2:
        scores.append(1.0)
        continue
      start_g = self._group_from_cell(seq[0])
      end_g = self._group_from_cell(seq[-1])
      bucket = self.top_k_data.get(start_g, {}).get(end_g, [])
      up = 0.0
      down = 0.0
      for rec in bucket[: self.cfg.top_k]:
        down += rec.popularity_sum
        up += rec.popularity_sum * normalized_edit_distance(seq, rec.points)
      scores.append(1.0 if down == 0 else up / down)
    return scores
