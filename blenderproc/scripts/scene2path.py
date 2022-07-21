from scipy.interpolate import interp1d
from dutils import rot_mat, trans_mat, scale_mat, dot, hmg
from trimesh.proximity import ProximityQuery
from scipy import interpolate
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.feature_extraction.image import grid_to_graph
from scipy.spatial import distance_matrix
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import trimesh
from dvis import dvis
from pathlib import Path
import networkx as nx
import pickle
from tqdm import tqdm
from coverage_helper import int_hash, get_cam_obs_coords, load_mesh_py3d, greedy_select, fov2K, get_render_conf_py3d, get_max_view_state
import torch
from nx_convert_matrix import from_scipy_sparse_array

###
# 1. Load scene_mesh
# 2. Voxelize mesh
# 3. Dilate vox cloud for real free space
# 4. Get occ mask per room
# 5. Get free mask per room
# 6. Do all calcs on sliced free mask
# 7. Get largest connected component of free space per room
# 8. Get room connectivity graph
# 9. Get shortest path connecting all rooms
# 10. For each room:
#     ## Create trajectory inside room
#     11. Get min distances of free voxels to occupied
#     12. Sample points based on distance scores
#     13. TSP for shortest connecting traj
#     14. Sample points on smoothed trajectory
#     14. Create trajectory using different logics
# 15. For each room transition:
#     16. Get trajectory from a-star connection
# 16. Fuse room + transition trajectories for complete trajectory

#### ---> functions for processing room info
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

#### <---


def load_scene_volumes(scene_mesh_fn: str, vox_size: float = 0.10, min_dist: float = 0.10):
    # loads a scene mesh and creates voxel representations
    scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')
    # NOTE: Trimesh does some weird transformation, this seems to fix it
    scene_mesh.vertices = np.column_stack(
        (scene_mesh.vertices[:, 0], -scene_mesh.vertices[:, 2], scene_mesh.vertices[:, 1]))

    tm_vox = scene_mesh.voxelized(vox_size)
    vox2scene = np.array(tm_vox.transform)

    occ_vox = np.zeros(tm_vox.shape, dtype=bool)
    occ_vox[tuple(np.array(tm_vox.sparse_indices).T)] = True

    dil_size = int(np.ceil(min_dist/vox_size))
    dil_occ_vox = binary_dilation(occ_vox, iterations=dil_size)

    return scene_mesh, occ_vox, dil_occ_vox, vox2scene

def vox2nx_graph(vox):
    # efficient conversion of occupancy mask to nx graph
    vox_inds = np.argwhere(vox)
    adj_mat = grid_to_graph(vox.shape[0], vox.shape[1], vox.shape[2], mask=vox)
    return vox_inds, from_scipy_sparse_array(adj_mat)


def create_room_trajectory(sl_room_occ_mask, sl_room_free_mask, init_sample_mode="w_random", cam_min_dist_v=3, mesh_py3d=None):
    # get padded free space based on distance to occ
    sl_room_occ_inds = np.argwhere(sl_room_occ_mask)
    sl_room_free_inds = np.argwhere(sl_room_free_mask)
    free2occ_dist_mat = distance_matrix(
        sl_room_free_inds, sl_room_occ_inds)
    free_dists = np.min(free2occ_dist_mat, 1)
    valid_room_inds = sl_room_free_inds[free_dists >= cam_min_dist_v]
    valid_dists = free_dists[free_dists >= cam_min_dist_v]
    # num of samples based on valid region size
    num_valid_samples = valid_room_inds.shape[0]

    # get initial camera key poses
    if init_sample_mode == "w_random":
        # weighted random selection based on distance to occ
        # TODO: Hard-coded parameters here
        min_samples = 30
        max_samples = 500
        sample_devisor = 40
        num_cam_samples = 2*max(min_samples, min(
            int(num_valid_samples/sample_devisor), max_samples))
        cam_key_indices = np.random.choice(np.arange(
            num_valid_samples), num_cam_samples, p=valid_dists/np.sum(valid_dists))
 

def create_room_traj_by_coverage(room_occ_mask,  room_free_mask, mesh_py3d, raster_settings, fov, K, scene2vox, cam_min_dist_v=3, method="dim_ret", num_cand_samples=1000, num_score_samples=None, global_view_obs_state=None):
    # get padded free space based on distance to occ
    device = room_occ_mask.device
    room_occ_inds = torch.nonzero(room_occ_mask)
    room_free_inds = torch.nonzero(room_free_mask)
    free2occ_dist_mat = torch.cdist(room_free_inds.float(), room_occ_inds.float(),p=1)
    free_dists = torch.min(free2occ_dist_mat, 1).values
    valid_room_inds = room_free_inds[free_dists >= cam_min_dist_v]
    valid_dists = free_dists[free_dists >= cam_min_dist_v]
    # num of samples based on valid region size
    num_valid_samples = valid_room_inds.shape[0]


    if num_score_samples is None:
        # use heuristic to determine how many scored sample should be used
        min_samples = 30
        max_samples = 500
        sample_devisor = 30
        num_score_samples = max(min_samples, min(
            int(num_valid_samples/sample_devisor), max_samples))

    # get best coverage poses from random poses
    # 1. sample random valid camera poses
    pitch_deg_range = [-20,10]
    pitch_ang_range = [x*np.pi/180 for x in pitch_deg_range]
    cand_locs = dot(torch.inverse(scene2vox), valid_room_inds[np.random.randint(0, num_valid_samples,num_cand_samples)].float())
    cand_rots = [rot_mat([1,0,0],pitch_ang).to(device) for pitch_ang in np.random.uniform(pitch_ang_range[0], pitch_ang_range[1], num_cand_samples)]
    cand_rots = [rot_mat([0,0,1],yaw_ang).to(device) @ cand_rots[rot_idx] for rot_idx, yaw_ang in enumerate(np.random.uniform(-np.pi,np.pi, num_cand_samples))]
    cand_c2ws = []
    for cand_loc, cand_rot in zip(cand_locs, cand_rots):
        cand_c2ws.append(trans_mat(cand_loc)@hmg(cand_rot))
    cand_c2ws = torch.stack(cand_c2ws,0)
    # mesh/vox has z and y flipped ->
    cand_c2ws = cand_c2ws[:,:,[0,2,1,3]]
    # get the cam_obs_coords of all cands
    cand_cam_obs_coords = []
    for cand_idx, cam2world in tqdm(enumerate(cand_c2ws)):
        cand_cam_obs_coord = get_cam_obs_coords(mesh_py3d,cam2world, fov, K,scene2vox,raster_settings,room_occ_mask)
        cand_cam_obs_coords.append(cand_cam_obs_coord)
    # greedily select cameras based on score
    view_bin_th = 4

    if global_view_obs_state is None:
        global_view_obs_state = torch.zeros((*room_occ_mask.shape,8),device=room_occ_mask.device)
    else:
        # for validation we use the train obs state to evaluate candidate val poses
        pass 
    best_cand_inds = greedy_select(global_view_obs_state,cand_cam_obs_coords, num_score_samples,method,view_bin_th=view_bin_th)
    best_c2ws = cand_c2ws[best_cand_inds]
    # compute obs_th=1 overage 
    best_cand_cam_obs_coords = [cand_cam_obs_coords[best_cand_idx].cpu() for best_cand_idx in best_cand_inds]
    # best_cov_ratio = torch.any(global_view_obs_state,dim=-1).sum() / room_occ_mask.sum()

    # max_global_view_obs_state = torch.zeros((*room_occ_mask.shape,8),device=room_occ_mask.device)
    # max_global_view_obs_state = get_max_view_state(max_global_view_obs_state,cand_cam_obs_coords, view_bin_th=view_bin_th)

    # # maximum possible (thresholded) based on all views vs actual observation level
    # coverage_ratio = 1- (max_global_view_obs_state - torch.clamp(global_view_obs_state,0,view_bin_th)).sum() / max_global_view_obs_state.sum()
    # obs_th = 4
    # obs_coverage_ratio = 1 - (torch.clamp(max_global_view_obs_state.sum(-1),0,obs_th) - torch.clamp(global_view_obs_state.sum(-1),0,obs_th)).sum() / torch.clamp(max_global_view_obs_state.sum(-1),0,obs_th).sum()
    # return best_c2ws, best_cand_cam_obs_coords, global_view_obs_state.cpu(), coverage_ratio, obs_coverage_ratio
    return best_c2ws, best_cand_cam_obs_coords, global_view_obs_state


def transform_dist_mat(T1, T2, trans_th=0.1, rot_th=5):
    trans1, trans2 = T1[:,:3,3], T2[:,:3,3]
    f1, f2 = T1[:,:3,2], T2[:,:3,2]
    if isinstance(T1, torch.Tensor):
        trans_dist = torch.cdist(trans1, trans2)
        f1 = f1/torch.norm(f1,dim=1,p=2)[:,None]
        f2 = f2/torch.norm(f2,dim=1, p=2)[:,None]
        rot_dist = torch.acos(torch.clamp(torch.einsum('ik,jk->ij', f1, f2),-1,1))*180/np.pi
        total_dist = trans_dist/trans_th + rot_dist/rot_th
    return total_dist


def get_best_order(T, trans_th=0.1, rot_th=5):
    # trans_th in m (scene), rot_th in degrees 
    dist_mat = transform_dist_mat(T,T, trans_th, rot_th=rot_th)
    G = nx.from_numpy_array(dist_mat.cpu().numpy()).to_undirected()
    tsp_sol = nx.approximation.traveling_salesman_problem(G, method=nx.approximation.greedy_tsp)
    return tsp_sol




def create_complete_trajectory(vfront_root: str, vox_size=0.15, min_dist=0.1, min_height=0.2, max_height=2.0, cam_min_dist=0.2, cov_vox_size=None):
    ### load scene and voxelize it
    scene_name = Path(vfront_root).stem
    scene_mesh_fn = Path(vfront_root, "mesh", "mesh.obj")
    scene_mesh, occ_vox, dil_occ_vox, vox2scene = load_scene_volumes(scene_mesh_fn, vox_size, min_dist)
 
    ### --->>> assign room information to free and occupied
    floor_meshes = get_floor_meshes(vfront_root)
    # for distance computation get room ids of occupied
    occ_vox_inds = np.argwhere(occ_vox)
    occ_pts_scene = dot(vox2scene, occ_vox_inds)
    occ_room_id_mask = get_room_id_mask(floor_meshes, occ_pts_scene)
    occ_room_id_vox = np.full(occ_vox.shape, -1)
    occ_room_id_vox[occ_vox] = occ_room_id_mask
    # get room assignments for each free space vox
    free_vox = ~dil_occ_vox
    free_vox_inds = np.argwhere(free_vox)
    free_pts_scene = dot(vox2scene, free_vox_inds)
    free_room_id_mask = get_room_id_mask(floor_meshes, free_pts_scene)
    free_room_id_vox = np.full(free_vox.shape, -1)
    free_room_id_vox[free_vox] = free_room_id_mask

    ### <<<---

    ### --->> Limit the search space to certain heights
    # To all calculations on sl_<...> to ensure reasonable trajectories
    min_height_v, max_height_v = int(np.ceil(min_height/vox_size)), int(max_height/vox_size)
    # limit the height and only work with the the slice
    sl_free_vox = free_vox[..., min_height_v:max_height_v+1]
    sl_free_room_id_vox = free_room_id_vox[..., min_height_v:max_height_v+1]

    sl_occ_vox = occ_vox[..., min_height_v:max_height_v+1]
    sl_occ_room_id_vox = occ_room_id_vox[..., min_height_v:max_height_v+1]
    ### <<<---

    ### ---> Connected space inside each room
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
        room_G = from_scipy_sparse_array(room_sparse_adj)
        # compute the biggest connected component
        conn_comps = list(nx.algorithms.connected_components(room_G))
        largest_cc = max(conn_comps, key=len)
        room_vox_cc_inds = room_vox_inds[list(largest_cc)]
        sl_free_room_id_vox_cc[tuple(room_vox_cc_inds.T)] = room_id
        # dvis(room_vox_cc_inds,c=int(room_id), name=f"cc_room/{int(room_id)}")
    ### <<<---

    ### --->>> Build scene graph of room connectivity
    # 1. convert sliced free space into a graph
    sl_free_vox_inds, sl_free_G = vox2nx_graph(sl_free_vox)
    # 2. pick (first) points from the cc of all rooms
    sl_room_sample_inds = {}
    for room_id in sl_room_ids:
        sl_room_sample_inds[room_id] = np.argwhere(sl_free_room_id_vox_cc == room_id)[0]
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
            try:
                a_path = nx.astar_path(sl_free_G, source=source, target=target)
                room_ids_visited = sl_free_room_id_vox_cc[tuple(
                    sl_free_vox_inds[a_path].T)]
                directly_connected = np.all(
                    np.isin(room_ids_visited, [room_id_a, room_id_b, -1]))
            except:
                # room completely seperated from the rest
                directly_connected = False
            room_adj_mat[room_id_a, room_id_b] = directly_connected
            room_adj_mat[room_id_b,room_id_a] = directly_connected
    # remove all non-reachable rooms
    reachable_idx = np.argwhere(room_adj_mat.sum(1)>0)[:,0]
    sl_room_ids = np.array(sl_room_ids)[reachable_idx]
    room_adj_mat = room_adj_mat[reachable_idx][:,reachable_idx]
    # [dvis(np.stack([sl_room_sample_inds[x[0]],sl_room_sample_inds[x[1]]],0),'line',c=3,vs=5,name="connected/0") for x in np.stack(np.nonzero(room_adj_mat),1)]
    # generate an order of visiting reachable rooms
    room_conn_G = nx.from_numpy_array(room_adj_mat)
    scene_tsp_sol = nx.approximation.traveling_salesman_problem(
        room_conn_G, cycle=False)
    # remove duplicates, maintain order -> even if intermediate rooms need to be visited, ignore them for a-star
    room_conn_order = list(dict.fromkeys(scene_tsp_sol))

    room_trajectories = dict()
    cam_min_dist_v = np.ceil(cam_min_dist/vox_size)
    init_sample_mode = "random_coverage"
    if "coverage" in init_sample_mode:
        H = 256
        fov = 89
        device = "cuda"
        # load mesh_py3d to re-use for all rooms
        mesh_py3d = load_mesh_py3d(scene_mesh_fn,device=device, blender_fmt=True)
        scene2vox = np.linalg.inv(vox2scene)
        if cov_vox_size != vox_size:
            # compute coverage different res compared to path finding
            cov2vox_scale = vox_size/cov_vox_size
            vox2cov_vox = scale_mat(torch.tensor([cov2vox_scale,cov2vox_scale,cov2vox_scale],device=device))
            _, cov_occ_vox, _, cov_vox2scene = load_scene_volumes(scene_mesh_fn, cov_vox_size, min_dist)
            cov_occ_vox_inds = np.argwhere(cov_occ_vox)
            cov_occ_pts_scene = dot(cov_vox2scene, cov_occ_vox_inds)
            cov_occ_room_id_mask = get_room_id_mask(floor_meshes, cov_occ_pts_scene)
            cov_occ_room_id_mask = torch.from_numpy(cov_occ_room_id_mask).to(device)
            # move to torch
            cov_vox2scene = torch.from_numpy(cov_vox2scene).to(device)
            cov_occ_vox = torch.from_numpy(cov_occ_vox).to(device)
            cov_occ_vox_inds = torch.from_numpy(cov_occ_vox_inds)
            cov_occ_room_id_vox = torch.full(cov_occ_vox.shape, -1, device=device)
            cov_occ_room_id_vox[cov_occ_vox] = cov_occ_room_id_mask
            # upscale free space but remove the sl for simplicity
            free_room_id_vox_cc = torch.full(occ_vox.shape,-1,dtype=torch.long, device=device)
            free_room_id_vox_cc[tuple((np.argwhere(sl_free_room_id_vox_cc>=0) + np.array([0,0,min_height_v])).T)] = torch.from_numpy(sl_free_room_id_vox_cc[sl_free_room_id_vox_cc>=0]).to(device)
            cov_free_room_id_vox_cc = torch.nn.Upsample(size=cov_occ_vox.shape,mode='nearest')(free_room_id_vox_cc.float().unsqueeze(0).unsqueeze(0)).long()[0,0]
        else:
            cov_occ_room_id_vox = torch.from_numpy(occ_room_id_vox).to(device)
            free_room_id_vox_cc = torch.full(occ_vox.shape,-1,dtype=torch.long, device=device)
            free_room_id_vox_cc[tuple((np.argwhere(sl_free_room_id_vox_cc>=0) + np.array([0,0,min_height_v])).T)] = torch.from_numpy(sl_free_room_id_vox_cc[sl_free_room_id_vox_cc>=0]).to(device)
            cov_free_room_id_vox_cc = torch.from_numpy(free_room_id_vox_cc).to(device)

        scene2cov_vox = torch.inverse(cov_vox2scene)
        raster_settings, K = get_render_conf_py3d(H,fov,device)
        # compute coverage at different voxel size compared to path etc calc

        room_sample_data = dict()
        for sl_room_id in sl_room_ids:
            room_occ_mask = cov_occ_room_id_vox == sl_room_id
            room_free_mask = cov_free_room_id_vox_cc == sl_room_id
            torch.manual_seed(int_hash(f"train_{scene_name}/{sl_room_id}"))
            train_c2ws, train_cam_obs_coords, train_global_view_obs_state = create_room_traj_by_coverage(room_occ_mask, room_free_mask,  mesh_py3d, raster_settings, fov, K, scene2cov_vox, cam_min_dist_v=3,num_cand_samples=1000)
            # re-order based on transform distance
            train_best_order = get_best_order(train_c2ws,trans_th=0.1, rot_th=5)
            train_c2ws = train_c2ws[train_best_order]
            train_cam_obs_coords = [train_cam_obs_coords[i] for i in train_best_order]

            num_val_score_samples = int(0.2*len(train_c2ws))
            torch.manual_seed(int_hash(f"val_{scene_name}/{sl_room_id}"))
            val_c2ws, val_cam_obs_coords, val_global_view_obs_state = create_room_traj_by_coverage(room_occ_mask, room_free_mask,  mesh_py3d, raster_settings, fov, K, scene2cov_vox, cam_min_dist_v=3,num_cand_samples=max(200,2*num_val_score_samples),num_score_samples=num_val_score_samples, global_view_obs_state=train_global_view_obs_state.clone())
            # re-order based on transform distance
            val_best_order = get_best_order(val_c2ws,trans_th=0.1, rot_th=5)
            val_c2ws = val_c2ws[val_best_order]
            val_cam_obs_coords = [val_cam_obs_coords[i] for i in val_best_order]

            room_sample_data[sl_room_id] = {
                "train": {"c2w": train_c2ws.cpu(), "cam_obs_coords": train_cam_obs_coords},
                "val": {"c2w": val_c2ws.cpu(), "cam_obs_coords": val_cam_obs_coords},
            }
            #[dvis(train_global_view_obs_state.sum(-1)>i,t=i,c=2,name=f'train/{i}') for i in range(0,200,10)]
            #[dvis(val_cam_obs_coords[i],t=i,c=3,name=f'val/{i}') for i in range(40)]
            # [dvis(x.cpu()@ torch.from_numpy(np.diag([1,1,-1.0,1])).float(),'obj_kf',t=i, name='RenderCamera') for i,x in enumerate(train_c2ws)]
    else:
        raise NotImplementedError()

    return room_sample_data
  


if __name__ == "__main__":
    # testing inside polygon functionality
    # scene_layout = get_scene_layout("/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1")
    # testing voxelization
    # scene_name = "/home/normanm/fb_data/renders_front3d_debug/0003d406-5f27-4bbf-94cd-1cff7c310ba1"
    scene_name ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"
    room_sample_data = create_complete_trajectory(vfront_root=scene_name, vox_size=0.2, min_dist=0.1, min_height=0.2, max_height=2.0, cam_min_dist=0.2, cov_vox_size=0.12)
    
    pickle.dump(room_sample_data, open(
        f"{scene_name}/room_sample_data.pkl", 'wb'))
    print('lel')
    # compl_trajectory, compl_meta = create_complete_trajectory(
    #     scene_name, vox_size=0.15, min_height_v=2, max_height_v=13, traj_mode=traj_mode, conn_traj_mode="follow_2d")
    # dvis(compl_trajectory[:, :3, 3], 'line', vs=2)
    # dvis(trs2f_vec(compl_trajectory), "vec", c=-1, vs=3)
    # pickle.dump(compl_trajectory, open(
    #     f"{scene_name}/compl_trajectory_2d_{suffix}.pkl", 'wb'))
    # pickle.dump(compl_meta, open(f"{scene_name}/compl_meta_2d_{suffix}.pkl", 'wb'))
