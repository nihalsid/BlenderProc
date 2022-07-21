import torch
import numpy as np
from torch_scatter import scatter
from tqdm import tqdm
from dutils import rot_mat, trans_mat, dot, hmg

from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, FoVPerspectiveCameras,  \
    MeshRenderer, HardPhongShader, PointLights, HardFlatShader, TexturesVertex
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from dvis import dvis



def unproject_2d_3d(cam2world, K, d, uv=None, th=None):
    if uv is None and len(d.shape) >= 2:
        # create mesh grid according to d
        uv = np.stack(np.meshgrid(np.arange(d.shape[1]), np.arange(d.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        d = d.reshape(-1)
        if not isinstance(d, np.ndarray):
            uv = torch.from_numpy(uv).to(d)
        if th is not None:
            uv = uv[d > th]
            d = d[d > th]
    if isinstance(uv, np.ndarray):
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
        cam_point = dot(np.linalg.inv(K), uvh) * d[:, None]
    else:
        uvh = torch.cat([uv, torch.ones(len(uv), 1).to(uv)], -1)
        cam_point = dot(torch.inverse(K), uvh) * d[:, None]

    world_point = dot(cam2world, cam_point)
    return world_point

def project_3d_2d(cam2world, K, world_point, with_z=False, discrete=True,round=True):
    from dutils import dot

    if isinstance(world_point, np.ndarray):
        cam_point = dot(np.linalg.inv(cam2world), world_point)
        img_point = dot(K, cam_point)
        uv_point = img_point[:, :2] / img_point[:, 2][:, None]
        if discrete:
            if round:
                uv_point = np.round(uv_point)
            uv_point = uv_point.astype(np.int)
        if with_z:
            return uv_point, img_point[:, 2]
        return uv_point

    else:
        cam_point = dot(torch.inverse(cam2world), world_point)
        img_point = dot(K, cam_point)
        uv_point = img_point[:, :2] / img_point[:, 2][:, None]
        if discrete:
            if round:
                uv_point = torch.round(uv_point)
                uv_point = uv_point.int()
        if with_z:
            return uv_point, img_point[:, 2]

        return uv_point

def load_mesh_py3d(obj_filename: str, device=None, blender_fmt=False):
    verts, faces_idx, _ = load_obj(obj_filename)
    faces = faces_idx.verts_idx
    if blender_fmt:
        # flip blender uses, also see scene_mesh generation
        verts = torch.stack([verts[:,0],-verts[:,2],verts[:,1]],1)        
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )
    return mesh


def fov2K(fov, H):
    # quadratic images only atm
    W = H
    fl_factor= 1/(np.tan(fov/2*np.pi/180)*2)
    K = torch.tensor([[fl_factor*W, 0, W/2],[0,fl_factor*H,H/2],[0,0,1]])
    return K


def get_render_conf_py3d(H,fov=89, device=None):
    raster_settings = RasterizationSettings(
        image_size=H,#512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    ) 
    K = fov2K(fov, H).to(device)
    return raster_settings, K


def convert_py3d(cam2world):
    # weird camera model used by py3d
    R = cam2world[:3,:3].unsqueeze(0)
    T = dot(torch.inverse(cam2world[:3,:3]), - cam2world[:3,3]).unsqueeze(0)
    return R, T


def render_rgb(mesh, cam2world, raster_settings):
    device = cam2world.device
    R, T = convert_py3d(cam2world)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh)
    return images[0,...,:3]

def render_depth(mesh, cam2world, raster_settings, fov=60.0):
    device = cam2world.device
    R, T = convert_py3d(cam2world)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)
    depth = fragments.zbuf[0,...,0].flip(0).flip(1)
    return depth

def get_depth_pts(mesh, cam2world, fov, K, raster_settings, with_mask=False):
    depth = render_depth(mesh,cam2world,raster_settings, fov=fov)
    pts = unproject_2d_3d(cam2world,K,depth)
    depth_mask = depth.flatten()>0
    if with_mask:
        return pts[depth_mask], depth_mask
    return pts[depth_mask]



def update_global_view_obs_state(global_view_obs_state, cam_obs_coords):
    if cam_obs_coords.shape[0]>0:
        global_view_obs_state[tuple(cam_obs_coords.T)]+=1
    return global_view_obs_state

def get_view_dirs(ij, cam2world, K):
    pp = dot(cam2world, dot(torch.inverse(K),torch.cat([ij, torch.ones(ij.shape[0],1,device=ij.device)],1)[:,[1,0,2]]))
    ray_o = torch.tile(cam2world[:3,3][None,:], (pp.shape[0],1))
    ray_d = pp - ray_o
    ray_d /= torch.norm(ray_d,dim=1)[:,None]
    return ray_o, ray_d

def bin2dec(x, bits):
    mask = 2**torch.arange(bits).to(x.device)
    return (mask*x).sum(-1)

def get_cam_obs_coords(mesh, cam2world, fov, K, scene2vox, raster_settings, valid_grid_mask=None):
    cam_pts_scene = get_depth_pts(mesh,cam2world,fov,K,raster_settings)
    #cam_pts_vox = torch.round(dot(scene2vox, cam_pts_scene))
    # Randomly perturb voxel positions by +/-0.5 to account for voxelization errors
    cam_pts_vox = torch.round(dot(scene2vox, cam_pts_scene)+0.5*(torch.rand(cam_pts_scene.shape,device=cam_pts_scene.device)*2-1)).int()
    if valid_grid_mask is not None:
        # only keep obs coords for valid indices in the grid
        in_grid_mask = torch.all(cam_pts_vox>=0,1) & torch.all(cam_pts_vox < torch.tensor(valid_grid_mask.shape).to(cam_pts_vox),1)
        valid_mask = valid_grid_mask[tuple(cam_pts_vox[in_grid_mask].long().T)]
        cam_pts_vox = cam_pts_vox[in_grid_mask][valid_mask]
        cam_pts_scene = cam_pts_scene[in_grid_mask][valid_mask]
    view_bins = bin2dec(cam2world[:3,3][None,:] - cam_pts_scene>0,3)
    cam_obs_coords = torch.unique(torch.cat([cam_pts_vox, view_bins[:,None]],1).long(),dim=0)
    return cam_obs_coords

def get_obs_score(global_view_obs_state, cam_obs_coords, method="view_bin_th", view_bin_th=None, obs_th=None):
    if method == 'view_bin_th':
        curr_state = global_view_obs_state[tuple(cam_obs_coords.T)]
        return int(torch.sum((curr_state + 1) <= view_bin_th))
    elif method == "obs_th":
        global_obs_state = global_view_obs_state.sum(-1)
        curr_state = global_obs_state[tuple(cam_obs_coords[:,:3].T)]
        return int(torch.sum((curr_state + 1) <= obs_th))
    else:
        raise NotImplementedError(f"Unknown method: {method}")


def get_obs_score_batched(global_view_obs_state, cam_obs_coords, batch_inds, method="view_bin_th", view_bin_th=None, obs_th=None):
    if method == 'view_bin_th':
        curr_state = global_view_obs_state[tuple(cam_obs_coords.T)]
        return scatter(1*((curr_state + 1) <= view_bin_th), batch_inds,dim=-1, reduce="sum")
    elif method == "obs_th":
        global_obs_state = global_view_obs_state.sum(-1)
        curr_state = global_obs_state[tuple(cam_obs_coords[:,:3].T)]
        return scatter(1*((curr_state + 1) <= obs_th), batch_inds,dim=-1, reduce="sum")
    else:
        raise NotImplementedError(f"Unknown method: {method}")


def score_based_update(global_view_obs_state, mesh,K,scene2vox, cand_cam2world,method="view_bin_th",view_bin_th=None, obs_th=None):
    cand_scores = []
    cand_cam_obs_coords = []
    for cand_idx, cam2world in enumerate(cand_cam2world):
        cam_obs_coords = get_cam_obs_coords(mesh,cam2world,K,scene2vox)
        cand_score = get_obs_score(global_view_obs_state,cam_obs_coords,method,view_bin_th=view_bin_th, obs_th=obs_th)
        cand_scores.append(cand_score)
        cand_cam_obs_coords.append(cam_obs_coords)
    if method in ['view_bin_th', "obs_th"]:
        # try maximizing
        best_cand_idx = np.argmax(cand_scores)
        update_global_view_obs_state(global_view_obs_state,cand_cam_obs_coords[best_cand_idx])
    return cand_cam2world[best_cand_idx], cand_scores[best_cand_idx]


def greedy_select(global_view_obs_state, cand_cam_obs_coords, num_score_samples: int, method: str="view_bin_th",view_bin_th=None, obs_th=None):
    # greedily selects best camera candidates based on score
    device = global_view_obs_state.device
    # batch all first, compute scores by reducing based on batch_idx
    b_cand_cam_obs_coords = torch.cat(cand_cam_obs_coords)
    batch_size = len(cand_cam_obs_coords)
    batch_inds = torch.cat([i*torch.ones(len(x), device=device) for i,x in enumerate(cand_cam_obs_coords)]).int()
    best_cand_inds = []
    for i in tqdm(range(num_score_samples)):
        # go over all remaining again 
        cand_scores = get_obs_score_batched(global_view_obs_state, b_cand_cam_obs_coords, batch_inds.long(),method,view_bin_th=view_bin_th, obs_th=obs_th)
        # NOTE: It still has a score for all batch_idx from 0 -> num_score_samples
        if method in ['view_bin_th', "obs_th"]:
            # try maximizing
            best_cand_idx = int(torch.argmax(cand_scores))
        else:
            raise NotImplementedError(f"Unknown method: {method}")
        if best_cand_idx in best_cand_inds:
            # in case scores are already satured
            best_cand_idx = list(set(np.arange(batch_size)).difference(best_cand_inds))[0]
        update_global_view_obs_state(global_view_obs_state,cand_cam_obs_coords[best_cand_idx])
        # delete the best
        b_cand_cam_obs_coords= b_cand_cam_obs_coords[batch_inds!=best_cand_idx]
        batch_inds = batch_inds[batch_inds!=best_cand_idx]
        best_cand_inds.append(best_cand_idx)

    return best_cand_inds

def get_max_view_state(max_view_obs_state, cand_cam_obs_coords, view_bin_th):
    for cand_cam_obs_coord in cand_cam_obs_coords:
        max_view_obs_state = update_global_view_obs_state(max_view_obs_state,cand_cam_obs_coord)
    
    max_view_obs_state = torch.clamp(max_view_obs_state,0,view_bin_th)
    return max_view_obs_state
    