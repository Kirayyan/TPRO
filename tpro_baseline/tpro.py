"""TPRO: time-dependent popular route based trajectory outlier detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .map_graph import RoadGraph


@dataclass
class RouteRecord:
  edges: List[int]
  points: List[int]
  popularity_data: List[int] = field(default_factory=list)
  popularity_sum: int = 0
  route_id: int = 0
  route_length: float = 0.0


def _route_popularity_key(rec: RouteRecord) -> Tuple:
  """Match C++ TPROData::operator< sort order (descending popularity)."""
  return tuple(-v for v in rec.popularity_data) + (-len(rec.popularity_data),)


def normalized_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
  n, m = len(a), len(b)
  if n == 0 and m == 0:
    return 0.0
  dp = [[0] * (m + 1) for _ in range(n + 1)]
  for i in range(n + 1):
    dp[i][0] = i
  for j in range(m + 1):
    dp[0][j] = j
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      cost = 0 if a[i - 1] == b[j - 1] else 1
      dp[i][j] = min(
          dp[i - 1][j - 1] + cost,
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
      )
  return dp[n][m] / max(n, m)


class TPRO:
  """Python port of the original TPRO C++ implementation."""

  def __init__(
      self,
      graph: RoadGraph,
      min_lat: float,
      max_lat: float,
      min_lon: float,
      max_lon: float,
      lon_blocks: int = 10,
      lat_blocks: int = 20,
      top_k: int = 5,
  ):
    self.graph = graph
    self.min_lat = min_lat
    self.max_lat = max_lat
    self.min_lon = min_lon
    self.max_lon = max_lon
    self.lon_blocks = lon_blocks
    self.lat_blocks = lat_blocks
    self.top_k = top_k
    self.d_lat = (max_lat - min_lat) / lat_blocks
    self.d_lon = (max_lon - min_lon) / lon_blocks
    self.group_data: Dict[int, Dict[int, List[RouteRecord]]] = {}
    self.top_k_data: Dict[int, Dict[int, List[RouteRecord]]] = {}

  def get_group_id(self, point_id: int) -> int:
    lat = self.graph.nodes_lat[point_id]
    lon = self.graph.nodes_lon[point_id]
    row = int((lat - self.min_lat) / self.d_lat)
    col = int((lon - self.min_lon) / self.d_lon)
    return row * self.lon_blocks + col

  def _edges_to_points(self, edges: Sequence[int]) -> Tuple[List[int], float]:
    if not edges:
      return [], 0.0
    points = []
    first = self.graph.edges[edges[0]]
    if first is None:
      return [], 0.0
    points.append(first.start_node)
    length = 0.0
    for eid in edges:
      edge = self.graph.edges[eid]
      if edge is None:
        continue
      points.append(edge.end_node)
      length += edge.length_m
    return points, length

  def fit(self, routes: Sequence[Sequence[int]]) -> None:
    self.group_data = {}
    route_id = 0
    for edges in routes:
      if not edges:
        continue
      points, length = self._edges_to_points(edges)
      if not points:
        continue
      start_gid = self.get_group_id(points[0])
      end_gid = self.get_group_id(points[-1])
      rec = RouteRecord(list(edges), points, route_id=route_id, route_length=length)
      route_id += 1
      self.group_data.setdefault(start_gid, {}).setdefault(end_gid, []).append(rec)

    self.top_k_data = {}
    for start_gid, end_map in self.group_data.items():
      self.top_k_data.setdefault(start_gid, {})
      for end_gid, group in end_map.items():
        times_count = [0] * len(self.graph.nodes_lat)
        count_last_change = [-1] * len(self.graph.nodes_lat)
        change_time = 0
        for rec in group:
          change_time += 1
          for node in rec.points:
            if count_last_change[node] != change_time:
              count_last_change[node] = change_time
              times_count[node] += 1

        for rec in group:
          change_time += 1
          pop = []
          for node in rec.points:
            if count_last_change[node] != change_time:
              count_last_change[node] = change_time
              pop.append(times_count[node])
          pop.sort()
          rec.popularity_data = pop
          rec.popularity_sum = sum(pop)

        group.sort(key=_route_popularity_key)
        seen_lengths = set()
        top: List[RouteRecord] = []
        for rec in group:
          if len(top) >= self.top_k:
            break
          if rec.route_length in seen_lengths:
            continue
          seen_lengths.add(rec.route_length)
          top.append(rec)
        self.top_k_data[start_gid][end_gid] = top

  def anomaly_scores(self, routes: Sequence[Sequence[int]]) -> List[float]:
    scores: List[float] = []
    for edges in routes:
      if not edges:
        scores.append(1.0)
        continue
      points, _ = self._edges_to_points(edges)
      if not points:
        scores.append(1.0)
        continue
      start_gid = self.get_group_id(points[0])
      end_gid = self.get_group_id(points[-1])
      bucket = self.top_k_data.get(start_gid, {}).get(end_gid, [])
      up = 0.0
      down = 0.0
      for rec in bucket[: self.top_k]:
        down += rec.popularity_sum
        delta = normalized_edit_distance(points, rec.points)
        up += rec.popularity_sum * delta
      scores.append(1.0 if down == 0 else up / down)
    return scores
