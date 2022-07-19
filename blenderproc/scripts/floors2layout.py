from scipy.interpolate import interp1d
from dutils import rot_mat, trans_mat
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
import shapely
import networkx as nx
from dutils import dot, hmg
import pickle

import hashlib


def int_hash(x):
    return int(hashlib.sha1(x.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


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


def _get_scene_layout(vfront_root):
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


def get_floor_meshes(vfront_root):
    floor_meshes = {}
    for room_floor_fn in Path(vfront_root, "floor").iterdir():
        room_name = room_floor_fn.stem
        if "OtherRoom" in room_name:
            # TODO Deal with OtherRoom at generation
            continue
        room_floor = trimesh.load(room_floor_fn)
        floor_meshes[room_name] = room_floor
    return floor_meshes


def _get_room_id_mask(query_pts, scene_layout: dict):
    # exhaustively query all rooms
    room_id_mask = np.full(query_pts.shape[0], -1)
    # room_idx = room_id because already sorted
    for room_id, (room_name, room_polygons) in enumerate(scene_layout.items()):
        single_room_mask = np.zeros(query_pts.shape[0], dtype=bool)
        for room_polygon in room_polygons:
            single_room_mask |= is_inside_sm_parallel(query_pts, room_polygon)

        room_id_mask[single_room_mask] = room_id

    return room_id_mask


def _get_layout_graph(scene_layout: dict, adj_th=0.05):
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


def slerp_forward(R1, R2, times):
    f1, f2 = R1[:3, 2], R2[:3, 2]
    f1, f2 = f1/np.linalg.norm(f1), f2/np.linalg.norm(f2)
    n = np.cross(f1, f2)
    n /= np.linalg.norm(n)
    delta_angle = np.arccos(np.dot(f1, f2))
    inter_R = []
    for time in times:
        inter_R.append(rot_mat(n, time*delta_angle) @ R1)
    return inter_R


def equal_slerp(T1, T2, axis=2, max_angle=0.05, max_trans=1):
    eps = 1e-6
    pos1, pos2 = T1[:3, 3], T2[:3, 3]
    vec1, vec2 = T1[:3, axis], T2[:3, axis]
    vec1, vec2 = vec1/(np.linalg.norm(vec1)+eps), vec2 / \
        (np.linalg.norm(vec2)+eps)
    n = np.cross(vec1, vec2)
    n /= np.linalg.norm(n)+eps
    delta_angle = np.arccos(np.dot(vec1, vec2))
    if delta_angle < max_angle:
        # small change, prevent using invalid n
        inter_trs = T1[None, :]
    else:
        inter_R = []
        n_steps = int(delta_angle/max_angle)+1
        if max_trans is not None:
            n_steps = max(n_steps, int(np.linalg.norm(pos2-pos1)/max_angle))
        for idx in range(n_steps):
            inter_R.append(rot_mat(n, idx*max_angle) @ T1[:3, :3])
        inter_R = np.stack(inter_R)

        lerp = interp1d([0, 1], np.stack([pos1, pos2]), axis=0)
        inter_trans = lerp(np.linspace(0, 1, n_steps, endpoint=False))

        inter_trs = np.tile(np.eye(4)[None, ], (n_steps, 1, 1))
        inter_trs[:, :3, :3] = inter_R
        inter_trs[:, :3, 3] = inter_trans
    assert np.isnan(inter_trs).sum() == 0

    return inter_trs


def path_following_traj2d(inter_pts, max_rot_deg=10.0, max_f_deg=10.0, max_trans=1, temp=0.2, 
    pitch_deg=20, yaw_deg=180, pitch_len=100, yaw_len=60, eps=1e-6, end_trs=None, cycle=True, start_trs=None):

    num_samples = len(inter_pts)
    up_vec = np.array([0, 0, 1])
    # compute 2d forward direction
    if cycle:
        # (last point attached, so take gradient of 1 and -1)
        f0 = np.concatenate([inter_pts[1:] - inter_pts[:-1],
                            (inter_pts[1]-inter_pts[-1])[None, :]])
    else:
        f0 = np.concatenate([inter_pts[1:] - inter_pts[:-1],
                            (inter_pts[-1]-inter_pts[-2])[None, :]])

    f0 /= np.linalg.norm(f0, axis=1)[:, None] + eps

    proj_f0 = np.stack([f0[:, 0], f0[:, 1], np.zeros(len(f0))], 1)
    f0_len = np.linalg.norm(proj_f0, axis=1)

    f1 = [f0[0]]
    for t in range(1, len(f0)):
        f_next = f1[-1] + min(f0_len[t] * temp, 1) * (f0[t]-f1[-1])
        f_next /= np.linalg.norm(f_next) + eps
        f1.append(f_next)
    f1 = np.stack(f1)

    f_ang_max = max_f_deg * np.pi/180
    f2 = np.copy(f1)
    for i in range(f1.shape[0]):
        sgn = np.sign(np.dot(f1[i], up_vec))
        ang0 = np.arccos(np.dot(f1[i], sgn*up_vec))
        # angle between f1 and up
        if np.pi/2 - np.abs(ang0) > f_ang_max:
            # the angle is too small
            # cam looks too much up or down
            delta_angle = sgn * (f_ang_max - (np.pi/2 - np.abs(ang0)))
            # -> get f1, vector that has same ortho. direction to up_vec but less yaw
            n = np.cross(f1[i], up_vec)
            n /= np.linalg.norm(n)+eps
            # rotate around n so f1 = rot_a * f1, with
            # np.abs(np.dot(f1,up_vec)) = f_dot_max
            f2[i] = dot(rot_mat(n, delta_angle), f1[i])
    # with new forward vectors, update left and up

    l = np.cross(f2, up_vec)
    l /= np.linalg.norm(l, axis=1)[:, None] + eps
    u = np.cross(f2, l)
    u *= np.sign(u[:, 2])[:, None]
    sample_rots = np.stack([l, u, f2], -1)

    sample_tranfs = np.tile(np.eye(4)[None, ...], (num_samples, 1, 1))
    sample_tranfs[:, :3, 3] = inter_pts
    sample_tranfs[:, :3, :3] = sample_rots
    if end_trs is not None:
        sample_tranfs = np.concatenate([sample_tranfs, end_trs[None, :]])
    if start_trs is not None:
        sample_tranfs = np.concatenate([start_trs[None, :], sample_tranfs])

    max_rot_angle = max_rot_deg * np.pi/180  # maximum about of rotation per step
    interp_trs = [np.copy(x) for x in sample_tranfs]

    if pitch_len is not None:
        interp_trs = np.stack(interp_trs)
        # add tiktok movement on top
        t = np.arange(len(interp_trs))
        pitch_angles = pitch_deg * np.pi/180 * np.sin(t/pitch_len*2*np.pi)
        yaw_angles = yaw_deg * np.pi/180 * np.sin(t/yaw_len*2*np.pi)

        pitch_rot = np.stack([rot_mat([1, 0, 0], pa) for pa in pitch_angles])
        yaw_rot = np.stack([rot_mat([0, 0, 1], ya) for ya in yaw_angles])

        tiktok_rots = (yaw_rot @ pitch_rot) # [:, :, [0, 2, 1]]
        interp_trs2 = np.copy(interp_trs)
        interp_trs2[:,:3,:3] = tiktok_rots @ interp_trs2[:,:3,:3]

        interp_trs = [np.copy(x) for x in interp_trs2]
    # run interpolation for each axis
    # -> this enforces that changes for all rotation axes are small
    if cycle:
        bias = 0
    else:
        bias = 1
    for axis in [2, 1, 0]:
        new_interp_trs = []
        for i in range(len(interp_trs)-bias):
            int_tranfs = equal_slerp(interp_trs[i], interp_trs[(
                i+1) % len(interp_trs)], axis, max_rot_angle, max_trans)
            new_interp_trs += [x for x in int_tranfs]
        interp_trs = new_interp_trs
    interp_trs = np.stack(interp_trs)

    return interp_trs


def path_following_traj(inter_pts, max_f_deg=30.0, max_rot_deg=10.0, eps=1e-6, end_trs=None, cycle=True):
    num_samples = len(inter_pts)

    up_vec = np.array([0, 0, 1])
    # bound the angle of forward and up vector by a threshold f_ang_max
    f_ang_max = max_f_deg * np.pi/180
    # (allow up_vec to tilt a bit)
    if cycle:
        # (last point attached, so take gradient of 1 and -1)
        f0 = np.concatenate([inter_pts[1:] - inter_pts[:-1],
                            (inter_pts[1]-inter_pts[-1])[None, :]])
    else:
        f0 = np.concatenate([inter_pts[1:] - inter_pts[:-1],
                            (inter_pts[-1]-inter_pts[-2])[None, :]])

    f0 /= np.linalg.norm(f0, axis=1)[:, None] + eps

    f1 = np.copy(f0)
    for i in range(f0.shape[0]):
        sgn = np.sign(np.dot(f0[i], up_vec))
        ang0 = np.arccos(np.dot(f0[i], sgn*up_vec))
        # angle between f0 and up
        if np.pi/2 - np.abs(ang0) > f_ang_max:
            # the angle is too small
            # cam looks too much up or down
            delta_angle = sgn * (f_ang_max - (np.pi/2 - np.abs(ang0)))
            # -> get f1, vector that has same ortho. direction to up_vec but less yaw
            n = np.cross(f0[i], up_vec)
            n /= np.linalg.norm(n)+eps
            # rotate around n so f1 = rot_a * f0, with
            # np.abs(np.dot(f1,up_vec)) = f_dot_max
            f1[i] = dot(rot_mat(n, delta_angle), f0[i])
    # with new forward vectors, update left and up
    l = np.cross(f1, up_vec)
    l /= np.linalg.norm(l, axis=1)[:, None] + eps
    u = np.cross(f1, l)
    u *= np.sign(u[:, 2])[:, None]
    sample_rots = np.stack([l, u, f1], -1)

    sample_tranfs = np.tile(np.eye(4)[None, ...], (num_samples, 1, 1))
    sample_tranfs[:, :3, 3] = inter_pts
    sample_tranfs[:, :3, :3] = sample_rots
    if end_trs is not None:
        sample_tranfs = np.concatenate([sample_tranfs, end_trs[None, :]])

    max_rot_angle = max_rot_deg * np.pi/180  # maximum about of rotation per step
    interp_trs = [np.copy(x) for x in sample_tranfs]
    # run interpolation for each axis
    # -> this enforces that changes for all rotation axes are small
    if cycle:
        bias = 0
    else:
        bias = 1
    for axis in [0,1,2]:
        new_interp_trs = []
        for i in range(len(interp_trs)-bias):
            int_tranfs = equal_slerp(interp_trs[i], interp_trs[(
                i+1) % len(interp_trs)], axis, max_rot_angle)
            new_interp_trs += [x for x in int_tranfs]
        interp_trs = new_interp_trs
    interp_trs = np.stack(interp_trs)
    return interp_trs


def tiktok_traj(inter_pts, pitch_deg=30, yaw_deg=180, pitch_len=100, yaw_len=60, max_f_deg=30.0, max_rot_deg=10.0, eps=1e-6, end_trs=None, start_trs=None, cycle=True):
    # pitch speed
    num_samples = len(inter_pts)

    t = np.arange(num_samples)
    down_look_deg = 5
    pitch_angles = pitch_deg * np.pi/180 * np.sin(t/pitch_len*2*np.pi) - down_look_deg * np.pi/180
    yaw_angles = yaw_deg * np.pi/180 * np.sin(t/yaw_len*2*np.pi)
    yaw_angles = 2*yaw_deg * np.pi/180 * ((t/yaw_len) % 1)  # np.sin(t/yaw_len*2*np.pi)

    pitch_rot = np.stack([rot_mat([3, 0, 0], pa) for pa in pitch_angles])
    yaw_rot = np.stack([rot_mat([0, 0, 1], ya) for ya in yaw_angles])

    sample_rots = (yaw_rot @ pitch_rot)[:, :, [0, 2, 1]]

    sample_tranfs = np.tile(np.eye(4)[None, ...], (num_samples, 1, 1))
    sample_tranfs[:, :3, 3] = inter_pts
    sample_tranfs[:, :3, :3] = sample_rots
    if end_trs is not None:
        sample_tranfs = np.concatenate([sample_tranfs, end_trs[None, :]])
    if start_trs is not None:
        sample_tranfs = np.concatenate([start_trs[None, :], sample_tranfs])

    if max_rot_deg is not None:
        max_rot_angle = max_rot_deg * np.pi/180  # maximum about of rotation per step
        interp_trs = [np.copy(x) for x in sample_tranfs]
        # run interpolation for each axis
        # -> this enforces that changes for all rotation axes are small
        if cycle:
            bias = 0
        else:
            bias = 1
        for axis in [0,1,2]:
            new_interp_trs = []
            for i in range(len(interp_trs)-bias):
                int_tranfs = equal_slerp(interp_trs[i], interp_trs[(
                    i+1) % len(interp_trs)], axis, max_rot_angle)
                new_interp_trs += [x for x in int_tranfs]
            interp_trs = new_interp_trs
        interp_trs = np.stack(interp_trs)
    else:
        interp_trs = sample_tranfs
    return interp_trs


def get_complete_traj(vfront_root: str,  vox_size=0.15, min_height_v=2, max_height_v=13, dilation_size_v=1, cam_min_dist_v=4, traj_mode='follow', conn_traj_mode=None):
    if conn_traj_mode is None:
        conn_traj_mode = traj_mode
    np.random.seed(int_hash(vfront_root))
    scene_mesh_fn = Path(vfront_root, "mesh", "mesh.obj")
    scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')
    # NOTE: Trimesh does some weird transformation, this seems to fix it
    scene_mesh.vertices = np.column_stack(
        (scene_mesh.vertices[:, 0], -scene_mesh.vertices[:, 2], scene_mesh.vertices[:, 1]))
    # for visualization
    sl_scene_mesh = scene_mesh.slice_plane(
        np.array([0, 0, 1.8]), np.array([0, 0, -1]))

    tm_vox = scene_mesh.voxelized(vox_size)
    vox2scene = np.array(tm_vox.transform)
    # sliced room based on z min height
    sl_vox2scene = np.copy(vox2scene) @ trans_mat([0, 0, min_height_v])
    floor_meshes = get_floor_meshes(vfront_root)
    occ_vox = np.zeros(tm_vox.shape, dtype=bool)
    occ_vox[tuple(np.array(tm_vox.sparse_indices).T)] = True
    # dil_struct = generate_binary_structure(dilation_size_v, dilation_size_v)
    dil_occ_vox = binary_dilation(occ_vox, iterations=dilation_size_v)

    # for distance computation get room ids of occupied
    occ_vox_inds = np.stack(np.nonzero(occ_vox), 1)
    occ_pts_scene = dot(vox2scene, occ_vox_inds)
    occ_room_id_mask = get_room_id_mask(floor_meshes, occ_pts_scene)
    occ_room_id_vox = np.full(occ_vox.shape, -1)
    occ_room_id_vox[occ_vox] = occ_room_id_mask

    # get room assignments for each free space vox
    free_vox = ~dil_occ_vox
    free_vox_inds = np.stack(np.nonzero(free_vox), 1)
    free_pts_scene = dot(vox2scene, free_vox_inds)
    free_room_id_mask = get_room_id_mask(floor_meshes, free_pts_scene)
    free_room_id_vox = np.full(free_vox.shape, -1)
    free_room_id_vox[free_vox] = free_room_id_mask

    # limit the height and only work with the the slice
    sl_free_vox = free_vox[..., min_height_v:max_height_v+1]
    sl_free_room_id_vox = free_room_id_vox[..., min_height_v:max_height_v+1]

    sl_occ_vox = occ_vox[..., min_height_v:max_height_v+1]
    sl_occ_room_id_vox = occ_room_id_vox[..., min_height_v:max_height_v+1]

    # calculate the reachable points inside room as the largest connected component
    # per room, store ids
    sl_free_room_id_vox_cc = np.full(sl_free_room_id_vox.shape, -1)
    sl_room_ids = [x for x in np.unique(sl_free_room_id_vox) if x >= 0]
    # generate per-room graphs to get cc
    for room_id in sl_room_ids:
        room_vox_mask = sl_free_room_id_vox == room_id
        room_vox_inds = np.stack(np.nonzero(room_vox_mask), 1)
        room_sparse_adj = grid_to_graph(
            sl_free_vox.shape[0], sl_free_vox.shape[1], sl_free_vox.shape[2], mask=room_vox_mask)
        room_G = nx.from_scipy_sparse_array(room_sparse_adj)
        # compute the biggest connected component
        conn_comps = list(nx.algorithms.connected_components(room_G))
        largest_cc = max(conn_comps, key=len)
        room_vox_cc_inds = room_vox_inds[list(largest_cc)]
        sl_free_room_id_vox_cc[tuple(room_vox_cc_inds.T)] = room_id
        # dvis(room_vox_cc_inds,c=int(room_id), name=f"cc_room/{int(room_id)}")

    # compute a path connecting all room
    # 1. convert sliced free space into a graph
    sl_free_vox_inds = np.stack(np.nonzero(sl_free_vox), 1)
    sl_free_adj = grid_to_graph(
        sl_free_vox.shape[0], sl_free_vox.shape[1], sl_free_vox.shape[2], mask=sl_free_vox)
    sl_free_G = nx.from_scipy_sparse_array(sl_free_adj)
    # 2. pick points from the cc of all rooms
    sl_room_sample_inds = {}
    for room_id in sl_room_ids:
        sl_room_sample_inds[room_id] = np.stack(
            np.nonzero(sl_free_room_id_vox_cc == room_id), 1)[0]
    # 3. compute shortest paths between any two rooms, check if they do not touch another room
    #   -> rooms are connected
    room_adj_mat = np.zeros((len(sl_room_ids), len(sl_room_ids)), dtype=bool)
    for idx_a, room_id_a in enumerate(sl_room_ids):
        for idx_b, room_id_b in enumerate(sl_room_ids[idx_a+1:]):
            # get source and target as graph indices
            sample_ind_a = sl_room_sample_inds[room_id_a]
            sample_ind_b = sl_room_sample_inds[room_id_b]
            source = int(
                np.where(np.all(sl_free_vox_inds == sample_ind_a, 1))[0])
            target = int(
                np.where(np.all(sl_free_vox_inds == sample_ind_b, 1))[0])
            a_path = nx.astar_path(sl_free_G, source=source, target=target)
            room_ids_visited = sl_free_room_id_vox_cc[tuple(
                sl_free_vox_inds[a_path].T)]
            directly_connected = np.all(
                np.isin(room_ids_visited, [room_id_a, room_id_b, -1]))
            room_adj_mat[room_id_a, room_id_b] = directly_connected

    # [dvis(np.stack([sl_room_sample_inds[x[0]],sl_room_sample_inds[x[1]]],0),'line',c=3,vs=5,name="connected/0") for x in np.stack(np.nonzero(room_adj_mat),1)]
    # generate an order of visiting all rooms
    room_conn_G = nx.from_numpy_array(room_adj_mat)
    tsp_sol = nx.approximation.traveling_salesman_problem(
        room_conn_G, cycle=False)
    # remove duplicates, maintain order -> even if intermediate rooms need to be visited, ignore them for a-star
    # for debugging: a-star through all
    full_path_inds = []
    unique_tsp_sol = list(dict.fromkeys(tsp_sol))
    for idx in range(len(unique_tsp_sol)-1):
        room_id_a, room_id_b = unique_tsp_sol[idx], unique_tsp_sol[idx+1]
        sample_ind_a = sl_room_sample_inds[room_id_a]
        sample_ind_b = sl_room_sample_inds[room_id_b]
        source = int(np.where(np.all(sl_free_vox_inds == sample_ind_a, 1))[0])
        target = int(np.where(np.all(sl_free_vox_inds == sample_ind_b, 1))[0])
        a_path = nx.astar_path(sl_free_G, source=source, target=target)
        path_inds = sl_free_vox_inds[a_path]
        full_path_inds.append(path_inds)
        # dvis(dot(sl_vox2scene, path_inds),vs=vox_size,c=idx,name=f"conn_path/{idx}")
    full_path_inds = np.concatenate(full_path_inds)

    # sample locations inside room
    # based on distance to anything
    # relative to size of room
    min_samples = 30
    max_samples = 500
    sample_devisor = 40
    smoothness = 200
    sample_devisor = 30
    smoothness = 150
    room_trajectories = dict()
    for sl_room_id in sl_room_ids:
        sl_room_occ_mask = sl_occ_room_id_vox == sl_room_id
        sl_room_free_mask = sl_free_room_id_vox_cc == sl_room_id
        sl_room_occ_inds = np.stack(np.nonzero(sl_room_occ_mask), 1)
        sl_room_free_inds = np.stack(np.nonzero(sl_room_free_mask), 1)
        free2occ_dist_mat = distance_matrix(
            sl_room_free_inds, sl_room_occ_inds)
        free_dists = np.min(free2occ_dist_mat, 1)
        valid_room_inds = sl_room_free_inds[free_dists >= cam_min_dist_v]
        valid_dists = free_dists[free_dists >= cam_min_dist_v]

        # num of samples based on valid region size
        num_valid_samples = valid_room_inds.shape[0]
        num_cam_samples = 2*max(min_samples, min(
            int(num_valid_samples/sample_devisor), max_samples))
        cam_key_indices = np.random.choice(np.arange(
            num_valid_samples), num_cam_samples, p=valid_dists/np.sum(valid_dists))
        cam_key_inds = valid_room_inds[cam_key_indices]

        # compute path between all the cam key points
        num_samples = 3*num_cam_samples
        dist_mat = distance_matrix(cam_key_inds, cam_key_inds)
        cam_key_G = nx.from_numpy_array(dist_mat)
        tsp_sol = nx.approximation.traveling_salesman_problem(cam_key_G)
        # create a smooth trajectory between ordered points
        tck, u = interpolate.splprep(
            tuple(cam_key_inds[tsp_sol].T), s=smoothness)
        sample_u = np.linspace(0, 1, num_samples, endpoint=True)
        inter_pts = np.stack(interpolate.splev(sample_u, tck), 1)
        # ensure that first point is valid by concantenating first key point
        inter_pts = np.concatenate(
            [(cam_key_inds[tsp_sol][0]+np.random.rand(3))[None, :], inter_pts], 0)
        # clip according to scene bounds
        inter_pts = np.clip(inter_pts, np.zeros(
            3), np.array(sl_occ_vox.shape)-1)

        # after smoothing, points can lay in invalid regions
        # compute astar path for all invalid segments
        invalid_sample = ~sl_room_free_mask[tuple(inter_pts.astype(int).T)]
        if invalid_sample.sum() > 0:
            # there are segments that hit invalid regions
            new_inter_pts = []
            idx = 0
            in_invalid_segm = False
            for idx in range(len(inter_pts)):
                if not in_invalid_segm:
                    if invalid_sample[idx]:
                        # invalid segment starts
                        invalid_seg_start = idx
                        in_invalid_segm = True
                    else:
                        # just append the valid inter_pt
                        new_inter_pts.append(inter_pts[idx])
                else:
                    if not invalid_sample[idx]:
                        # invalid segment ends
                        invalid_seg_end = idx
                        # compute astar path
                        seg_start_ind = inter_pts[invalid_seg_start -
                                                  1].astype(int)
                        seg_end_ind = inter_pts[invalid_seg_end].astype(int)
                        source = int(
                            np.where(np.all(sl_free_vox_inds == seg_start_ind, 1))[0])
                        target = int(
                            np.where(np.all(sl_free_vox_inds == seg_end_ind, 1))[0])
                        a_path = nx.astar_path(
                            sl_free_G, source=source, target=target)
                        path_inds = sl_free_vox_inds[a_path]
                        # append new astar path
                        new_inter_pts += [x +
                                          np.random.rand(3) for x in path_inds]
                        in_invalid_segm = False

            new_inter_pts = np.stack(new_inter_pts)
            inter_pts = new_inter_pts
        if traj_mode == "follow":
            room_trajectories[int(sl_room_id)] = path_following_traj(
                inter_pts, max_f_deg=30, max_rot_deg=10)
        if traj_mode == "follow_2d":
            room_trajectories[int(sl_room_id)] = path_following_traj2d(
                inter_pts, max_rot_deg=(1-min(0.6,num_samples/max_samples))*20, max_trans=1, temp=0.2,
                pitch_deg=20, yaw_deg=180, pitch_len=100, yaw_len=120*(1-min(0.6,num_samples/max_samples))
                # pitch_deg=0, yaw_deg=0, pitch_len=100, yaw_len=120*(1-min(0.6,num_samples/max_samples))
                )
        elif traj_mode == "tiktok":
            room_trajectories[int(sl_room_id)] = tiktok_traj(
                inter_pts, pitch_deg=20, yaw_deg=180, pitch_len=100, yaw_len=70, max_rot_deg=None)

        if False:
            # debugging code
            # scaled back to orginal mesh
            dvis(sl_scene_mesh)
            interp_trs = room_trajectories[int(sl_room_id)]
            room_trs = sl_vox2scene @ interp_trs
            room_cam_pos = room_trs[:, :3, 3]

            dvis(trs2f_vec(room_trs), "vec", c=int(sl_room_id), vs=1)

            dvis(room_cam_pos, "line", c=int(sl_room_id), vs=3)
            dvis(inter_pts, "line", c=6, vs=3)
            dvis(pts, vs=0.03, c=3)
            dvis(pts[tsp_sol], "line", c=2)
            [dvis(room_trs[i], t=i, vs=0.5,
                  name=f"trs/{i}") for i in range(100)]

            dvis({"trs": room_trs[0], "fov": 90}, 'cam', name='cam')
            [dvis(room_trs[t], "obj_kf", name="cam", t=t)
             for t in range(0, 200)]
    if False:
        for sl_room_id, interp_trs in room_trajectories.items():
            room_trs = sl_vox2scene @ interp_trs
            room_cam_pos = room_trs[:, :3, 3]
            dvis(room_cam_pos, "line", c=sl_room_id,
                 vs=3, name=f"room_traj/{sl_room_id}")
    # following given room order, find closest points by l2 connecting traj
    reord_room_trajectories = {}
    connecting_trajectories = {}
    # reorder the room scanning cycles
    for idx in range(len(unique_tsp_sol)):
        room_id_a, room_id_b = unique_tsp_sol[idx], unique_tsp_sol[(idx+1)%len(unique_tsp_sol)]
        room_trj_a = room_trajectories[room_id_a]
        room_trj_b = room_trajectories[room_id_b]
        # compute l2 distance between points
        trj_dist = distance_matrix(room_trj_a[:, :3, 3], room_trj_b[:, :3, 3])
        shorted_conn_idx = np.unravel_index(
            np.argmin(trj_dist), trj_dist.shape)

        closest_point_a = room_trj_a[shorted_conn_idx[0], :3, 3].astype(int)
        closest_point_b = room_trj_b[shorted_conn_idx[1], :3, 3].astype(int)

        # start at connecting position at first visit of room
        if room_id_a not in reord_room_trajectories:
            reord_trj = np.concatenate(
                [room_trj_a[shorted_conn_idx[0]:], room_trj_a[:shorted_conn_idx[0]]])
            reord_room_trajectories[room_id_a] = reord_trj
        if room_id_b not in reord_room_trajectories:
            reord_trj = np.concatenate(
                [room_trj_b[shorted_conn_idx[1]:], room_trj_b[:shorted_conn_idx[1]]])
            reord_room_trajectories[room_id_b] = reord_trj

        # already re-ordered the trajectory
        # take end of this as starting point for connecting traj
        closest_point_a = reord_room_trajectories[room_id_a][-1,
                                                             :3, 3].astype(int)

        source = int(
            np.where(np.all(sl_free_vox_inds == closest_point_a, 1))[0])
        target = int(
            np.where(np.all(sl_free_vox_inds == closest_point_b, 1))[0])
        a_path = nx.astar_path(sl_free_G, source=source, target=target)
        path_inds = sl_free_vox_inds[a_path]

        # attach end point of traj in room a to the connecting path
        # let interpolation handle the smooth transition
        # path_inds = np.concatenate([reord_room_trajectories[room_id_a][-1,:3,3][None,:],path_inds],0)
        num_samples = 3*len(path_inds)
        tck, u = interpolate.splprep(tuple(path_inds.T), s=10)
        sample_u = np.linspace(0, 1, num_samples, endpoint=True)
        inter_pts = np.stack(interpolate.splev(sample_u, tck), 1)
        # dvis(dot(sl_vox2scene, inter_pts),vs=vox_size)
        if conn_traj_mode == "follow":
            conn_trs = path_following_traj(inter_pts, max_f_deg=30, max_rot_deg=10, cycle=False,
                                           end_trs=reord_room_trajectories[room_id_b][0], start_trs=reord_room_trajectories[room_id_a][-1])
        if conn_traj_mode == "follow_2d":
            conn_trs = path_following_traj2d(inter_pts, max_rot_deg=10, max_trans=1, temp=0.2, cycle=False,
                                            pitch_deg=0, yaw_deg=0, pitch_len=100, yaw_len=200,
                                             end_trs=reord_room_trajectories[room_id_b][0], start_trs=reord_room_trajectories[room_id_a][-1])
        elif conn_traj_mode == "tiktok":
            conn_trs = tiktok_traj(inter_pts, pitch_deg=0, yaw_deg=0, pitch_len=10, yaw_len=10, cycle=False,
                                   end_trs=reord_room_trajectories[room_id_b][0], start_trs=reord_room_trajectories[room_id_a][-1])
        connecting_trajectories[idx] = conn_trs
        # [dvis(sl_vox2scene @ conn_trs[i], name=f"asb/{i}") for i in range(50)]

    # for visualizing
    if False:
        for idx in range(len(unique_tsp_sol)-1):
            room_id_a, room_id_b = unique_tsp_sol[idx], unique_tsp_sol[idx+1]
            room_trj = reord_room_trajectories[room_id_a]
            conn_trj = connecting_trajectories[idx]
            dvis(dot(sl_vox2scene, room_trj[:, :3, 3]), "line",
                 vs=3, c=room_id_a, name=f"room_trj/{room_id_a}", t=idx)
            dvis(dot(sl_vox2scene, conn_trj[:, :3, 3]), "line",
                 vs=3, c=idx, name=f"conn_trj/{idx}", t=idx)

    # fuse all trajectories to a complete one + meta data
    # meta data: room_id_a []
    compl_trajectory = []
    compl_meta = []
    for idx in range(len(unique_tsp_sol)):
        room_id_a, room_id_b = unique_tsp_sol[idx], unique_tsp_sol[(idx+1)%len(unique_tsp_sol)]
        room_traj = reord_room_trajectories[room_id_a]
        conn_traj = connecting_trajectories[idx]
        compl_trajectory.append(room_traj)
        compl_trajectory.append(conn_traj)

        meta_room_id_a = np.array(
            (len(room_traj)+len(conn_traj)) * [room_id_a])  # room traj
        # identifies conn traj if >=0
        meta_room_id_b = np.array(
            len(room_traj)*[-1] + len(conn_traj)*[room_id_b])
        meta_room_id = np.concatenate([sl_free_room_id_vox[tuple(room_traj[:, :3, 3].astype(int).T)],
                                       sl_free_room_id_vox[tuple(conn_traj[:, :3, 3].astype(int).T)]])
        meta = np.stack([meta_room_id_a, meta_room_id_b, meta_room_id], 1)
        # lookup room ids at each position
        compl_meta.append(meta)

    compl_trajectory = np.concatenate(compl_trajectory)
    # convert back into scene space
    # preserve rotation (no scaling on this)
    compl_trajectory[:,:3,3] = dot(sl_vox2scene, compl_trajectory[:,:3,3])
    # compl_trajectory = sl_vox2scene @ compl_trajectory
    compl_meta = np.concatenate(compl_meta)

    return compl_trajectory, compl_meta


if __name__ == "__main__":
    # testing inside polygon functionality
    # scene_layout = get_scene_layout("/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1")
    # testing voxelization
    scene_name = "/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1"
    # scene_name ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"
    traj_mode = 'tiktok'
    suffix = "direct"
    suffix = "tiktok"
    compl_trajectory, compl_meta = get_complete_traj(
        scene_name, vox_size=0.15, min_height_v=2, max_height_v=13, traj_mode=traj_mode, conn_traj_mode="follow_2d")
    dvis(compl_trajectory[:, :3, 3], 'line', vs=2)
    dvis(trs2f_vec(compl_trajectory), "vec", c=-1, vs=3)
    pickle.dump(compl_trajectory, open(
        f"{scene_name}/compl_trajectory_2d_{suffix}.pkl", 'wb'))
    pickle.dump(compl_meta, open(f"{scene_name}/compl_meta_2d_{suffix}.pkl", 'wb'))
