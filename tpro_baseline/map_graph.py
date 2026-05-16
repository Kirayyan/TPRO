"""Road network loader and map-matching utilities for Porto (TPRO map files)."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class Edge:
    edge_id: int
    start_node: int
    end_node: int
    length_m: float


class RoadGraph:
  """Minimal road graph used by TPRO (nodes + directed edges + Dijkstra)."""

  def __init__(self, map_dir: str | Path, area: Optional[Tuple[float, float, float, float]] = None):
    map_dir = Path(map_dir)
    self.nodes_lat: List[float] = []
    self.nodes_lon: List[float] = []
    self.edges: List[Optional[Edge]] = []
    self.adj: List[List[Tuple[int, int]]] = []  # node -> [(neighbor, edge_id), ...]
    self._load(map_dir, area)

  def _load(self, map_dir: Path, area: Optional[Tuple[float, float, float, float]]) -> None:
    node_path = map_dir / "nodeOSM.txt"
    edge_path = map_dir / "edgeOSM.txt"

    with node_path.open() as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
          continue
        nid = int(parts[0])
        lat, lon = float(parts[1]), float(parts[2])
        while len(self.nodes_lat) <= nid:
          self.nodes_lat.append(0.0)
          self.nodes_lon.append(0.0)
        self.nodes_lat[nid] = lat
        self.nodes_lon[nid] = lon

    n_nodes = len(self.nodes_lat)
    self.adj = [[] for _ in range(n_nodes)]

    min_lat = max_lat = min_lon = max_lon = None
    if area is not None:
      min_lat, max_lat, min_lon, max_lon = area

    def in_area(lat: float, lon: float, geo: bool) -> bool:
      if area is None:
        return True
      if geo:
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
      return min_lat < lat < max_lat and min_lon < lon < max_lon

    with edge_path.open() as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
          continue
        edge_id = int(parts[0])
        start_node = int(parts[1])
        end_node = int(parts[2])
        figure_count = int(parts[3])
        coords = [float(x) for x in parts[4 : 4 + figure_count * 2]]
        length_m = 0.0
        for i in range(figure_count - 1):
          lat1, lon1 = coords[2 * i], coords[2 * i + 1]
          lat2, lon2 = coords[2 * (i + 1)], coords[2 * (i + 1) + 1]
          length_m += haversine_m(lat1, lon1, lat2, lon2)

        while len(self.edges) <= edge_id:
          self.edges.append(None)

        s_lat, s_lon = self.nodes_lat[start_node], self.nodes_lon[start_node]
        e_lat, e_lon = self.nodes_lat[end_node], self.nodes_lon[end_node]
        if not in_area(s_lat, s_lon, False) and not in_area(e_lat, e_lon, False):
          self.edges[edge_id] = None
          continue

        edge = Edge(edge_id, start_node, end_node, length_m)
        self.edges[edge_id] = edge
        self.adj[start_node].append((end_node, edge_id))

  def nearest_node(self, lat: float, lon: float) -> int:
    best_id = 0
    best_dist = float("inf")
    for nid, (nlat, nlon) in enumerate(zip(self.nodes_lat, self.nodes_lon)):
      d = haversine_m(lat, lon, nlat, nlon)
      if d < best_dist:
        best_dist = d
        best_id = nid
    return best_id

  def shortest_path_edges(self, src: int, dst: int) -> List[int]:
    if src == dst:
      return []
    dist = [float("inf")] * len(self.nodes_lat)
    prev_edge = [-1] * len(self.nodes_lat)
    dist[src] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, src)]
    while heap:
      d, u = heapq.heappop(heap)
      if d > dist[u]:
        continue
      if u == dst:
        break
      for v, eid in self.adj[u]:
        edge = self.edges[eid]
        if edge is None:
          continue
        nd = d + edge.length_m
        if nd < dist[v]:
          dist[v] = nd
          prev_edge[v] = eid
          heapq.heappush(heap, (nd, v))

    if dist[dst] == float("inf"):
      return []

    path: List[int] = []
    cur = dst
    while cur != src:
      eid = prev_edge[cur]
      if eid < 0:
        return []
      path.append(eid)
      edge = self.edges[eid]
      if edge is None:
        return []
      cur = edge.start_node
    path.reverse()
    return path

  def match_latlon_sequence_to_route(self, lats: Sequence[float], lons: Sequence[float]) -> List[int]:
    """Snap GPS points to nodes and stitch shortest-path edge sequences."""
    if len(lats) == 0:
      return []
    nodes = [self.nearest_node(lat, lon) for lat, lon in zip(lats, lons)]
    route: List[int] = []
    for i in range(len(nodes) - 1):
      segment = self.shortest_path_edges(nodes[i], nodes[i + 1])
      if not segment:
        continue
      if route and segment[0] == route[-1]:
        route.extend(segment[1:])
      else:
        route.extend(segment)
    if not route and len(nodes) >= 2:
      route = self.shortest_path_edges(nodes[0], nodes[-1])
    return route


# MST-OATD Porto grid settings (preprocess/preprocess_porto.py)
PORTO_MST_BOUNDARY = {
    "min_lat": 41.140092,
    "max_lat": 41.185969,
    "min_lng": -8.690261,
    "max_lng": -8.549155,
}
PORTO_MST_MAP_SIZE = (51, 119)
PORTO_MST_GRID_SIZE_KM = 0.1


def _grid_sizes(boundary: Dict[str, float], grid_size_km: float) -> Tuple[float, float, int, int]:
  lat_dist_km = haversine_m(boundary["min_lat"], boundary["min_lng"], boundary["max_lat"], boundary["min_lng"]) / 1000.0
  lng_dist_km = haversine_m(boundary["min_lat"], boundary["min_lng"], boundary["min_lat"], boundary["max_lng"]) / 1000.0
  lat_size = (boundary["max_lat"] - boundary["min_lat"]) / lat_dist_km * grid_size_km
  lng_size = (boundary["max_lng"] - boundary["min_lng"]) / lng_dist_km * grid_size_km
  lat_grid_num = int(lat_dist_km / grid_size_km) + 1
  lng_grid_num = int(lng_dist_km / grid_size_km) + 1
  return lat_size, lng_size, lat_grid_num, lng_grid_num


def grid_id_to_latlon(
    grid_id: int,
    boundary: Dict[str, float] = PORTO_MST_BOUNDARY,
    lat_grid_num: int = PORTO_MST_MAP_SIZE[0],
    lng_grid_num: int = PORTO_MST_MAP_SIZE[1],
) -> Tuple[float, float]:
  lat_size, lng_size, lat_gn, lng_gn = _grid_sizes(boundary, PORTO_MST_GRID_SIZE_KM)
  if lat_grid_num != lat_gn or lng_grid_num != lng_gn:
    lat_grid_num, lng_grid_num = lat_gn, lng_gn
  gi = int(grid_id // lng_grid_num)
  gj = int(grid_id % lng_grid_num)
  lat = boundary["min_lat"] + (gi + 0.5) * lat_size
  lng = boundary["min_lng"] + (gj + 0.5) * lng_size
  return lat, lng


def mst_traj_to_latlon(traj: Sequence[Sequence]) -> Tuple[List[float], List[float]]:
  lats, lons = [], []
  for point in traj:
    grid_id = int(point[0])
    lat, lng = grid_id_to_latlon(grid_id)
    lats.append(lat)
    lons.append(lng)
  return lats, lons


def mst_traj_to_edge_route(graph: RoadGraph, traj: Sequence[Sequence]) -> List[int]:
  lats, lons = mst_traj_to_latlon(traj)
  return graph.match_latlon_sequence_to_route(lats, lons)


def load_edge_routes_csv(path: str | Path) -> List[List[int]]:
  routes = []
  with Path(path).open() as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = [x.strip() for x in line.replace(" ", "").split(",") if x.strip()]
      routes.append([int(x) for x in parts])
  return routes
