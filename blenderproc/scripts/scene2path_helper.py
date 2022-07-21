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
from dvis import dvis
from pathlib import Path
import networkx as nx
import pickle
import hashlib


def pts_inside_floor_mesh(floor_mesh, pts, th=0.02):
    pts_2d = np.concatenate([pts[:, :2], np.zeros((len(pts), 1))], 1)
    unique_pts_2d, indices, map_inv = np.unique(
        pts_2d, return_index=True, return_inverse=True, axis=0)
    unique_mask = np.abs(ProximityQuery(
        floor_mesh).signed_distance(unique_pts_2d)) < th
    mask = unique_mask[map_inv]
    return mask


def get_room_id_mask(floor_meshes, query_pts, th=0.02):
    # exhaustively query all rooms
    room_id_mask = np.full(query_pts.shape[0], -1)
    # room_idx = room_id because already sorted
    for room_id, (room_name, floor_mesh) in enumerate(floor_meshes.items()):
        single_room_mask = pts_inside_floor_mesh(
            floor_mesh, query_pts, th=0.02)
        room_id_mask[single_room_mask] = room_id
    return room_id_mask

