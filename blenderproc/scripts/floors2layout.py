from concurrent.futures import process
from enum import unique
from math import dist
from turtle import distance, shape
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from numba import jit, njit
import numba
import trimesh
from dvis import dvis
from pathlib import Path
from glob import glob
import shapely
import networkx as nx
from dutils import dot

def in_hull(p, hull):
    return hull.find_simplex(p)>=0

def points2convex_hull(points):
    return  ConvexHull(points)

def point_in_hull(point, hull, tolerance=1e-12):
    return np.stack((np.dot(eq[:-1], point.T) + eq[-1] <= tolerance) for eq in hull.equations).all(0)

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

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D  


def _room_floor2polygons(room_floor_fn: str):
    room_floor = trimesh.load(room_floor_fn)

    room_outline = room_floor.outline()

    polygons = []
    for entity in room_outline.entities:
        polygon = np.array(room_outline.vertices[entity.points ])[:,:2]
        polygon = np.concatenate([polygon, polygon[:1]])
        polygons.append(polygon)
    return polygons

def _get_scene_layout(vfront_root):
    scene_layout = {}
    for room_floor_fn in Path(vfront_root,"floor").iterdir():
        room_name = room_floor_fn.stem 
        if "OtherRoom" in room_name:
            # TODO Deal with OtherRoom at generation
            continue
        room_polygons = room_floor2polygons(room_floor_fn)
        scene_layout[room_name] = room_polygons
    # keep sorted so name-> idx = room_id
    scene_layout = {room_name: scene_layout[room_name] for room_name in sorted(scene_layout.keys())}
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
    room_id_mask = np.full(query_pts.shape[0],-1)
    # room_idx = room_id because already sorted
    for room_id, (room_name, room_polygons) in enumerate(scene_layout.items()):
        single_room_mask = np.zeros(query_pts.shape[0],dtype=bool)
        for room_polygon in room_polygons:
            single_room_mask |= is_inside_sm_parallel(query_pts, room_polygon)
        
        room_id_mask[single_room_mask] = room_id

    return room_id_mask

def _get_layout_graph(scene_layout: dict, adj_th=0.05):
    shapely_layout = []
    for room_idx, (room_name, room_polygon) in enumerate(scene_layout.items()):
        shapely_layout.append(shapely.geometry.Polygon(room_polygon))

    num_rooms = len(shapely_layout)
    distance_mat = np.full((num_rooms,num_rooms),100,dtype=np.float32)
    def polygon_point_dist(polygon, point):
        return polygon.exterior.distance(shapely.geometry.Point(point))
    for room_idx_a, shapely_poly in enumerate(shapely_layout):
        for room_idx_b in range(num_rooms):
            min_dist = 100
            for point in np.array(shapely_poly.exterior.coords):
                min_dist = min(polygon_point_dist(shapely_layout[room_idx_b],point),min_dist)
            distance_mat[room_idx_a,room_idx_b] = min_dist
    # non-symmetric distance
    # TODO: check if it's really correct to test d(points,poly) alone
    distance_mat = np.min(np.stack([distance_mat,distance_mat.T]),0)
    np.fill_diagonal(distance_mat,100)
    # create directed scene graph of scene layout using adj_th 
    scene_graph = nx.from_numpy_matrix(distance_mat<adj_th)
    # get shortest path connecting all rooms (use undirected graph)
    shortest_path = nx.approximation.traveling_salesman_problem(scene_graph, cycle=False)

    travel_room_path = list(np.array(list(scene_layout))[shortest_path])
    return travel_room_path


from scipy.spatial import distance_matrix
from sklearn.feature_extraction.image import grid_to_graph
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy import interpolate
from trimesh.proximity import ProximityQuery

def pts_inside_floor_mesh(floor_mesh, pts, th=0.02):
    pts_2d = np.concatenate([pts[:,:2],np.zeros((len(pts),1))],1)
    unique_pts_2d, indices, map_inv = np.unique(pts_2d,return_index=True, return_inverse=True,axis=0)
    unique_mask = np.abs(ProximityQuery(floor_mesh).signed_distance(unique_pts_2d)) < th
    mask = unique_mask[map_inv]
    return mask

def get_room_id_mask(floor_meshes, query_pts, th=0.02):
    # exhaustively query all rooms 
    room_id_mask = np.full(query_pts.shape[0],-1)
    # room_idx = room_id because already sorted
    for room_id, (room_name, floor_mesh) in enumerate(floor_meshes.items()):
        single_room_mask = pts_inside_floor_mesh(floor_mesh, query_pts, th=0.02)
        room_id_mask[single_room_mask] = room_id
    return room_id_mask


def get_all(vfront_root: str,  vox_size=0.15, min_height_v = 2, max_height_v = 13, dilation_size_v=1):
    scene_mesh_fn = Path(vfront_root,"mesh","mesh.obj")
    scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')
    # NOTE: Trimesh does some weird transformation, this seems to fix it
    scene_mesh.vertices = np.column_stack((scene_mesh.vertices[:,0], -scene_mesh.vertices[:,2], scene_mesh.vertices[:,1]))
    tm_vox = scene_mesh.voxelized(vox_size)
    occ_vox_grid = np.zeros(tm_vox.shape,dtype=bool)
    occ_vox_grid[tuple(np.array(tm_vox.sparse_indices).T)]=True
    # dil_struct = generate_binary_structure(dilation_size_v, dilation_size_v)
    dil_occ_vox_grid = binary_dilation(occ_vox_grid, iterations=dilation_size_v)

    # for distance computation get room ids of occupied
    occ_vox_inds = np.stack(np.nonzero(occ_vox_grid),1)
    occ_pts_scene = dot(tm_vox.transform, occ_vox_inds) 
    occ_room_id_mask = get_room_id_mask(floor_meshes, occ_pts_scene)
    occ_room_id_vox = np.full(occ_vox_grid.shape,-1)
    occ_room_id_vox[occ_vox_grid] = occ_room_id_mask

    # get room assignments for each free space vox
    free_vox = ~dil_occ_vox_grid
    free_vox_inds = np.stack(np.nonzero(free_vox),1)
    free_pts_scene =dot(tm_vox.transform, free_vox_inds) 
    floor_meshes = get_floor_meshes(vfront_root)
    free_room_id_mask = get_room_id_mask(floor_meshes, free_pts_scene)
    free_room_id_vox = np.full(free_vox.shape,-1)
    free_room_id_vox[free_vox] = free_room_id_mask

    # limit the height and only work with the the slice
    sl_free_vox = free_vox[...,min_height_v:max_height_v+1]
    sl_free_room_id_vox = free_room_id_vox[...,min_height_v:max_height_v+1]


    # calculate the reachable points inside room as the largest connected component
    # per room, store ids 
    sl_free_room_id_vox_cc = np.full(sl_free_room_id_vox.shape,-1) 
    sl_room_ids = [x for x in np.unique(sl_free_room_id_vox) if x>=0]
    # generate per-room graphs to get cc
    for room_id in sl_room_ids:
        room_vox_mask = sl_free_room_id_vox==room_id
        room_vox_inds = np.stack(np.nonzero(room_vox_mask),1)
        room_sparse_adj = grid_to_graph(sl_free_vox.shape[0], sl_free_vox.shape[1], sl_free_vox.shape[2], mask=room_vox_mask)
        room_G = nx.from_scipy_sparse_array(room_sparse_adj)
        # compute the biggest connected component
        conn_comps = list(nx.algorithms.connected_components(room_G))
        largest_cc = max(conn_comps, key=len)
        room_vox_cc_inds= room_vox_inds[list(largest_cc)]
        sl_free_room_id_vox_cc[tuple(room_vox_cc_inds.T)] = room_id
        # dvis(room_vox_cc_inds,c=int(room_id), name=f"cc_room/{int(room_id)}")

    # compute a path connecting all room
    # 1. convert sliced free space into a graph
    sl_free_vox_inds = np.stack(np.nonzero(sl_free_vox),1)
    sl_free_adj = grid_to_graph(sl_free_vox.shape[0], sl_free_vox.shape[1], sl_free_vox.shape[2], mask=sl_free_vox)
    sl_free_G = nx.from_scipy_sparse_array(sl_free_adj)
    # 2. pick points from the cc of all rooms 
    sl_room_sample_inds = {}
    for room_id in sl_room_ids:
        sl_room_sample_inds[room_id] = np.stack(np.nonzero(sl_free_room_id_vox_cc==room_id),1)[0]
    # 3. compute shortest paths between any two rooms, check if they do not touch another room 
    #   -> rooms are connected
    room_adj_mat = np.zeros((len(sl_room_ids), len(sl_room_ids)),dtype=bool)
    for idx_a, room_id_a in enumerate(sl_room_ids):
        for idx_b, room_id_b in enumerate(sl_room_ids[idx_a+1:]):
            # get source and target as graph indices
            sample_ind_a = sl_room_sample_inds[room_id_a]
            sample_ind_b = sl_room_sample_inds[room_id_b]
            source = int(np.where(np.all(sl_free_vox_inds == sample_ind_a,1))[0])
            target = int(np.where(np.all(sl_free_vox_inds == sample_ind_b,1))[0])
            a_path = nx.astar_path(sl_free_G,source=source,target=target)
            room_ids_visited = sl_free_room_id_vox_cc[tuple(sl_free_vox_inds[a_path].T)]
            directly_connected = np.all(np.isin(room_ids_visited, [room_id_a, room_id_b,-1]))
            room_adj_mat[room_id_a,room_id_b] = directly_connected

    # [dvis(np.stack([sl_room_sample_inds[x[0]],sl_room_sample_inds[x[1]]],0),'line',c=3,vs=5,name="connected/0") for x in np.stack(np.nonzero(room_adj_mat),1)]
    # generate an order of visiting all rooms 
    room_conn_G = nx.from_numpy_array(room_adj_mat)
    tsp_sol = nx.approximation.traveling_salesman_problem(room_conn_G, cycle=False) 
    # remove duplicates, maintain order -> even if intermediate rooms need to be visited, ignore them for a-star
    # for debugging: a-star through all 
    full_path_inds = []
    unique_tsp_sol = list(dict.fromkeys(tsp_sol))
    for idx in range(len(unique_tsp_sol)-1):
        room_id_a, room_id_b = unique_tsp_sol[idx], unique_tsp_sol[idx+1]
        sample_ind_a = sl_room_sample_inds[room_id_a]
        sample_ind_b = sl_room_sample_inds[room_id_b]
        source = int(np.where(np.all(sl_free_vox_inds == sample_ind_a,1))[0])
        target = int(np.where(np.all(sl_free_vox_inds == sample_ind_b,1))[0])
        a_path = nx.astar_path(sl_free_G,source=source,target=target)
        path_inds = sl_free_vox_inds[a_path]
        full_path_inds.append(path_inds)
    full_path_inds = np.concatenate(full_path_inds)








    dvis(np.concatenate([free_pts_scene, free_room_id_mask[:,None]],1),vs=0.1)


    free_vox_slices = ~dil_dense_vox_grid[...,:min_height_v,max_height_v]
    free_vox_slices = free_vox_slices
    free_vox_slices_inds = np.stack(np.nonzero(free_vox_slices),1)
    free_pts_scene = dot(tm_vox.transform, _vox_inds)


    room_id_mask = get_room_id_mask(floor_meshes, free_pts_scene)

    room_id_mask = get_room_id_mask(occ_pts_scene[:,[0,1]],scene_layout)


    sparse_adj = grid_to_graph(free_vox_slices.shape[0], free_vox_slices.shape[1], free_vox_slices.shape[2], mask=free_vox_slices)


    # free_vox_inds = np.stack(np.nonzero(~dil_dense_vox_grid[:,10:11]),1)
    # dvis(free_vox_inds)
    free_vox_slices = ~dense_vox_grid[:,min_height_v,max_height_v]
    free_vox_slices = free_vox_slices.transpose(0,2,1)
    free_vox_slices_inds = np.stack(np.nonzero(free_vox_slices),1)
    sparse_adj = grid_to_graph(free_vox_slices.shape[0], free_vox_slices.shape[1], free_vox_slices.shape[2], mask=free_vox_slices)
    G = nx.from_scipy_sparse_array(sparse_adj)
    source = int(np.where(np.all(free_vox_slices_inds == np.array([6,18,1]),1))[0])
    target = int(np.where(np.all(free_vox_slices_inds == np.array([90,60,0]),1))[0])
    a_path = nx.astar_path(G,source=source,target=target)
    dvis(free_vox_slices_inds[a_path],'xyz',c=6,vs=2)

    # back into original voxel grid coordinates
    a_path_vox_pos = (free_vox_slices_inds[a_path] + np.array([0,0,min_height_v]))[:,[0,2,1]]
    a_path_scene = dot(tm_vox.transform, a_path_vox_pos)
    

    # test tsp
    pts = np.random.rand(200,3)
    dist_mat = distance_matrix(pts,pts)
    G_test = nx.from_numpy_array(dist_mat)
    tsp_sol = nx.approximation.traveling_salesman_problem(G_test)
    smoothness = 1
    tck, u = interpolate.splprep(tuple(pts[tsp_sol].T),s=smoothness)
    num_samples = 1000
    sample_u = np.linspace(0,1,num_samples,endpoint=True)
    inter_pts = np.stack(interpolate.splev(sample_u,tck),1)
    sample_dir_u  = np.concatenate([inter_pts[1:] - inter_pts[:-1], (inter_pts[-1]-inter_pts[0])[None,:]])
    sample_dir_u = sample_dir_u/ np.linalg.norm(sample_dir_u,axis=1)[:,None]
    up_vec = np.array([0,1,0])
    # project sample_dir_u onto plane orthogonal to up_vec [xz here]
    sample_dir_u_proj = np.stack([sample_dir_u[:,0], np.zeros(num_samples), sample_dir_u[:,2]],1)
    sample_dir_u_proj = sample_dir_u_proj / np.linalg.norm(sample_dir_u_proj,axis=1)[:,None]
    left_vec = np.cross(sample_dir_u_proj, up_vec)
    sample_rots = np.stack([left_vec, np.tile(up_vec[None,:], (num_samples,1)), sample_dir_u_proj],-1)
    sample_tranfs = np.tile(np.eye(4)[None,...], (num_samples,1,1))
    sample_tranfs[:,:3,3] = inter_pts
    sample_tranfs[:,:3,:3] = sample_rots
    dvis(inter_pts,"line",c=6,vs=3)
    dvis(pts, vs=0.03,c=3)
    dvis(pts[tsp_sol],"line",c=2)
    [dvis(sample_tranfs[i],t=i,vs=0.5, name=f"trs/{i}") for i in range(20)]


    dvis(free_vox_inds)
    dd = distance_matrix(free_vox_inds,free_vox_inds,p=1)
    adj = (dd<=1)
    G = nx.from_numpy_matrix(adj)
    a_path = nx.astar_path(G,source=2000,target=8000)
    dvis(free_vox_inds[a_path],'xyz',c=6,vs=2)
    sliced_dense_vox_grid = np.zeros_like(dense_vox_grid)
    sliced_dense_vox_grid[:,10:12] = dense_vox_grid[:,10:12]
    dvis(dense_vox_grid,c=2,l=1)
    dvis(sliced_dense_vox_grid,c=10,l=2)
    print('lel')



if __name__ == "__main__":
    # testing inside polygon functionality
    # scene_layout = get_scene_layout("/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1")
    # testing voxelization
    get_all("/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1",  vox_size=0.15, min_height_v = 2, max_height_v = 13)
    query_pts = np.random.rand(200000,2)*20-10
    room_id_mask = get_room_id_mask(query_pts, scene_layout)
    dvis(np.concatenate([query_pts,np.zeros((len(query_pts),1)),room_id_mask[:,None]],1),vs=0.1)
    get_layout_graph(scene_layout)
    room_polygon = room_floor2polygon("/home/normanm/Downloads/room_floor/MasterBedroom-5863.stl")
    in_room_mask = is_inside_sm_parallel(query_pts,room_polygon) 
    
    dvis(np.concatenate([query_pts,np.zeros((len(query_pts),1))],1)[in_room_mask],c=-2,vs=0.1)
    dvis(np.concatenate([query_pts,np.zeros((len(query_pts),1))],1)[~in_room_mask],c=-1,vs=0.1)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(query_pts[in_room_mask,0], query_pts[in_room_mask,1], 'o')
    plt.plot(query_pts[~in_room_mask,0], query_pts[~in_room_mask,1], 'x')
    plt.show()


    polygon_points = np.array([[-1,-1],[-1,3], [1,1], [2,-4], [0,0], [-1,-1]])
    query_pts = np.random.rand(20000,2)*4-2
    
    in_poly_mask = is_inside_sm_parallel(query_pts,polygon_points )

    hull = points2convex_hull(polygon_points)
    in_hull_mask = point_in_hull(query_pts,hull)
    

    # visualize
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(query_pts[in_hull_mask,0], query_pts[in_hull_mask,1], 'o')
    plt.plot(query_pts[~in_hull_mask,0], query_pts[~in_hull_mask,1], 'x')
    plt.show()
    
    plt.clf()
    plt.plot(query_pts[in_poly_mask,0], query_pts[in_poly_mask,1], 'bo')
    plt.plot(query_pts[~in_poly_mask,0], query_pts[~in_poly_mask,1], 'r+')
    plt.show()


    plot_in_hull(query_pts,hull)

