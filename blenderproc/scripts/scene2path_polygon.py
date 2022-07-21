from scipy.interpolate import interp1d
from dutils import rot_mat, trans_mat, dot, hmg
from trimesh.proximity import ProximityQuery
from scipy import interpolate
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.feature_extraction.image import grid_to_graph
from scipy.spatial import distance_matrix
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from numba import jit, njit
import numba
import trimesh
import shapely
from dvis import dvis
from pathlib import Path
import networkx as nx
import pickle
import hashlib

### functions for polygon point intersection

def in_hull(p, hull):
    return hull.find_simplex(p) >= 0


def points2convex_hull(points):
    return ConvexHull(points)


def point_in_hull(point, hull, tolerance=1e-12):
    return np.stack((np.dot(eq[:-1], point.T) + eq[-1] <= tolerance) for eq in hull.equations).all(0)


def trs2f_vec(trs):
    # get nx6 forward vector as vec_origin, vec_target
    return np.concatenate([trs[:, :3, 3], trs[:, :3, 3] + trs[:, :3, 2]], 1)


# Taken from
# https://github.com/sasamil/PointInPolygon_Py
@jit(nopython=True)
def is_inside_sm(polygon, point):
    # the representation of a point will be a tuple (x,y)
    # the representation of a polygon wil be a list of points [(x1,y1), (x2,y2), (x3,y3), ... ]
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy*(polygon[jj][0] - polygon[ii][0]) / \
                    (dy-dy2) + polygon[ii][0]

                # if line is left from the point - the ray moving towards left, will intersect it
                if point[0] > F:
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (point[0] == polygon[jj][0] or (dy == 0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0]) <= 0)):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D


def _room_floor2polygons(room_floor_fn: str):
    room_floor = trimesh.load(room_floor_fn)

    room_outline = room_floor.outline()

    polygons = []
    for entity in room_outline.entities:
        polygon = np.array(room_outline.vertices[entity.points])[:, :2]
        polygon = np.concatenate([polygon, polygon[:1]])
        polygons.append(polygon)
    return polygons


def room_floor2polygons(room_floor_fn: str):
    room_floor = trimesh.load(room_floor_fn)

    room_outline = room_floor.outline()

    polygons = []
    for entity in room_outline.entities:
        polygon = np.array(room_outline.vertices[entity.points])[:, :2]
        polygon = np.concatenate([polygon, polygon[:1]])
        polygons.append(polygon)
    return polygons


def get_scene_layout(vfront_root):
    scene_layout = {}
    for room_floor_fn in Path(vfront_root, "floor").iterdir():
        room_name = room_floor_fn.stem
        if "OtherRoom" in room_name:
            # TODO Deal with OtherRoom at generation
            continue
        room_polygons = room_floor2polygons(room_floor_fn)
        scene_layout[room_name] = room_polygons
    # keep sorted so name-> idx = room_id
    scene_layout = {room_name: scene_layout[room_name]
                    for room_name in sorted(scene_layout.keys())}
    return scene_layout

def get_room_id_mask(query_pts, scene_layout: dict):
    # exhaustively query all rooms
    room_id_mask = np.full(query_pts.shape[0], -1)
    # room_idx = room_id because already sorted
    for room_id, (room_name, room_polygons) in enumerate(scene_layout.items()):
        single_room_mask = np.zeros(query_pts.shape[0], dtype=bool)
        for room_polygon in room_polygons:
            single_room_mask |= is_inside_sm_parallel(query_pts, room_polygon)

        room_id_mask[single_room_mask] = room_id

    return room_id_mask

def get_layout_graph(scene_layout: dict, adj_th=0.05):
    shapely_layout = []
    for room_idx, (room_name, room_polygon) in enumerate(scene_layout.items()):
        shapely_layout.append(shapely.geometry.Polygon(room_polygon))

    num_rooms = len(shapely_layout)
    distance_mat = np.full((num_rooms, num_rooms), 100, dtype=np.float32)

    def polygon_point_dist(polygon, point):
        return polygon.exterior.distance(shapely.geometry.Point(point))
    for room_idx_a, shapely_poly in enumerate(shapely_layout):
        for room_idx_b in range(num_rooms):
            min_dist = 100
            for point in np.array(shapely_poly.exterior.coords):
                min_dist = min(polygon_point_dist(
                    shapely_layout[room_idx_b], point), min_dist)
            distance_mat[room_idx_a, room_idx_b] = min_dist
    # non-symmetric distance
    # TODO: check if it's really correct to test d(points,poly) alone
    distance_mat = np.min(np.stack([distance_mat, distance_mat.T]), 0)
    np.fill_diagonal(distance_mat, 100)
    # create directed scene graph of scene layout using adj_th
    scene_graph = nx.from_numpy_matrix(distance_mat < adj_th)
    # get shortest path connecting all rooms (use undirected graph)
    shortest_path = nx.approximation.traveling_salesman_problem(
        scene_graph, cycle=False)

    travel_room_path = list(np.array(list(scene_layout))[shortest_path])
    return travel_room_path