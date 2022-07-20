from ast import Not
import MinkowskiEngine as ME 
import numpy as np
import trimesh
from pathlib import Path
from dutils import rot_mat, trans_mat, dot, hmg
import torch
from MinkowskiEngine.utils import batch_sparse_collate
from dvis import dvis
from vfront_helper import visualize_depth
from vfront_helper import unproject_2d_3d
from tqdm import tqdm 

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



def convert_py3d(cam2world):
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


def render_depth(mesh, cam2world, raster_settings):
    device = cam2world.device
    R, T = convert_py3d(cam2world)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)
    depth = fragments.zbuf[0,...,0].flip(0).flip(1)
    return depth

def get_depth_pts(mesh, cam2world, K, raster_settings, with_mask=False):
    depth = render_depth(mesh,cam2world,raster_settings)
    pts = unproject_2d_3d(cam2world,K,depth)
    depth_mask = depth.flatten()>0
    if with_mask:
        return pts[depth_mask], depth_mask
    return pts[depth_mask]

def idx2d_1d(idx_2d, W):
    return idx_2d[:,0]*W + idx_2d[:,1]

def idx1d_2d(idx_2d,W):
    return torch.stack([idx_2d//W, idx_2d%W],1)


device = "cuda"

scene_name ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"
min_height_v = 2
vox_size = 0.10
vfront_root = scene_name

scene_mesh_fn = Path(vfront_root, "mesh", "mesh_0.10.obj")
scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')

tm_vox = scene_mesh.voxelized(vox_size)
vox2scene = np.array(tm_vox.transform)
scene2vox = np.linalg.inv(vox2scene)
vox_inds = torch.from_numpy(tm_vox.sparse_indices).to(device)

dvis(scene_mesh)
from vfront_helper import visualize_depth

from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, FoVPerspectiveCameras,  \
    MeshRenderer, HardPhongShader, PointLights, HardFlatShader, TexturesVertex
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes

obj_filename = scene_mesh_fn
mesh = load_objs_as_meshes([obj_filename], device=device)
# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(obj_filename)
faces = faces_idx.verts_idx
# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))
mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)

raster_settings = RasterizationSettings(
    image_size=256,#512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

def fov2K(fov, H):
    # quadratic images only atm
    W = H
    fl_factor= 1/(np.tan(fov/2*np.pi/180)*2)
    K = torch.tensor([[fl_factor*W, 0, W/2],[0,fl_factor*H,H/2],[0,0,1]])
    return K

### global counter
def to_b(x):
    return torch.cat([torch.zeros((x.shape[0],1),device=x.device),x],1).int()


K = fov2K(60, raster_settings.image_size).to(device)
H=W=256
ij = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)),-1).to(device)


def update_global_view_obs_state(global_view_obs_state, cam_obs_coords):
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

def get_cam_obs_coords(mesh, cam2world, K, scene2vox):
    cam_pts_scene = get_depth_pts(mesh,cam2world,K,raster_settings)
    cam_pts_vox = dot(scene2vox, cam_pts_scene).int()
    view_bins = bin2dec(cam2world[:3,3][None,:] - cam_pts_scene>0,3)
    cam_obs_coords = torch.cat([cam_pts_vox, view_bins[:,None]],1).long()
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


## randomly sample some views first

global_view_obs_state = torch.zeros((*tm_vox.shape,8),device=device)
num_samples = 200# 1000
method = "view_bin_th"
method = "obs_th"
view_bin_th = 3
obs_th = 4
num_rand_samples = 100
for i in tqdm(range(num_rand_samples)):
    R = rot_mat([0,1,0], float(torch.rand(1))*2*np.pi).to(device)
    T = torch.tensor([5*(torch.rand(1)-0.5),1,5*(torch.rand(1)-0.5)]).to(device)
    cam2world = torch.eye(4,device=device)
    cam2world[:3,:3] = R
    cam2world[:3,3] = T
    cam_obs_coords = get_cam_obs_coords(mesh,cam2world,K,scene2vox)
    update_global_view_obs_state(global_view_obs_state,cam_obs_coords)
### now select based on score
num_score_samples = 300
num_cand = 20
for i in tqdm(range(num_score_samples)):
    cand_cam2world = []
    for j in range(num_cand):
        R = rot_mat([0,1,0], float(torch.rand(1))*2*np.pi).to(device)
        T = torch.tensor([5*(torch.rand(1)-0.5),1,5*(torch.rand(1)-0.5)]).to(device)
        cam2world = torch.eye(4,device=device)
        cam2world[:3,:3] = R
        cam2world[:3,3] = T
        cand_cam2world.append(cam2world)
    score_based_update(global_view_obs_state,mesh,K,scene2vox,cand_cam2world, method=method,view_bin_th=view_bin_th,obs_th=obs_th)



# dvis(torch.cat([ray_o, ray_o+ray_d],1),'vec',c=7)

global_view_obs_state = torch.zeros((*tm_vox.shape,8),device=device)
for i in tqdm(range(num_samples)):
    R = rot_mat([0,1,0], float(torch.rand(1))*2*np.pi).to(device)
    T = torch.tensor([5*(torch.rand(1)-0.5),1,5*(torch.rand(1)-0.5)]).to(device)
    cam2world = torch.eye(4,device=device)
    cam2world[:3,:3] = R
    cam2world[:3,3] = T
    cam_obs_coords = get_cam_obs_coords(mesh,cam2world,K,scene2vox)
    cam_score = get_obs_score(global_view_obs_state,cam_obs_coords,method,view_bin_th=3, obs_th=4)
    if cam_score < 300:
        # dvis(cam2world,t=i, name=f"cam/{i}")
        dvis(global_view_obs_state.sum(-1)>=obs_th, t=i, name=f"gl/{i}", l=1)
        dvis(cam_obs_coords[:,:3], t=i, name=f"coords/{i}",l=2,c=-1)
    print(cam_score)
    update_global_view_obs_state(global_view_obs_state,cam_obs_coords)




global_view_obs_state = torch.zeros((*tm_vox.shape,8),device=device)

cam2world =  torch.tensor([[1,0,0,-2],[0,1,0,1],[0,0,1,-2],[0,0,0,1]],device=device).float() # @hmg(rot_mat([1,0,0],np.pi/5)).to(device) 
ray_o, ray_d = get_view_dirs(ij.reshape(-1,2), cam2world,K)
i =0 
dvis(-ray_d,'vec',t=i+1)
dvis(cam2world,name=f"c2w/{i}")
update_global_view_obs_state(global_view_obs_state, mesh, ij, cam2world, K, scene2vox)
cam2world =  torch.tensor([[1,0,0,-2],[0,1,0,1],[0,0,1,-2],[0,0,0,1]],device=device).float() @hmg(rot_mat([0,1,0],np.pi/4)).to(device) 
ray_o, ray_d = get_view_dirs(ij.reshape(-1,2), cam2world,K)
i =1 
dvis(-ray_d,'vec',t=i+1)
dvis(cam2world,name=f"c2w/{i}")
update_global_view_obs_state(global_view_obs_state, mesh, ij, cam2world, K, scene2vox)

global_view_obs_state = torch.zeros((*tm_vox.shape,8),device=device)
num_samples = 20# 1000
for i in tqdm(range(num_samples)):
    R = rot_mat([0,1,0], float(torch.rand(1))*np.pi).to(device)
    T = 3*torch.rand(1,3,device=device)
    cam2world = torch.eye(4,device=device)
    cam2world[:3,:3] = R
    cam2world[:3,3] = T
    
    ray_o, ray_d = get_view_dirs(ij.reshape(-1,2), cam2world,K)
    view_bins = ray_d2view_bin(ray_d)
    # dvis(-ray_d,'vec',t=i+1)
    dvis(cam2world,name=f"c2w/{i}")
    update_global_view_obs_state(global_view_obs_state, mesh, ij, cam2world, K, scene2vox)
    # dvis(dot(vox2scene,torch.nonzero(global_view_obs_state>0).float()),t=i+1,name=f"gs/{i}",vs=vox_size)
# [dvis(dot(vox2scene,torch.nonzero(global_view_obs_state[...,i]>0).float()),t=i+1,c=i, name=f"gs/{i}",vs=vox_size) for i in range(8)]
# dvis(dot(vox2scene, torch.nonzero(global_view_obs_state).float()),vs=vox_size)

[dvis(dot(vox2scene,torch.nonzero(global_view_obs_state[...,i]>0).float()),c=i, name=f"gs/{i}",vs=vox_size) for i in range(8)]

for R,T in tqdm(zip(Rs,Ts)):
    # cam2world=torch.eye(4,device=device)
    # cam2world[:3,:3] = R
    # cam2world[:3,3] = T
    cameras = FoVPerspectiveCameras(device=device, R=R.unsqueeze(0), T=T.unsqueeze(0))
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)
    depth = fragments.zbuf[0,...,0]#.flip(0).flip(1)
    pts = unproject_2d_3d(cam2world,K,depth)
    dvis(pts,vs=0.03,c=3,ms=20000,name="pts/as")
    # dvis(visualize_depth(depth.cpu().numpy()),'img')



vox_sparse_count = ME.SparseTensor(coordinates=torch.cat([torch.zeros((vox_inds.shape[0],1),device=device),vox_inds],1).int(), features=counts)

voxel_idx = torch.arange(vox_inds.shape[0],device=device)
cam2world = torch.tensor([[1,0,0,10],[0,0,1,10],[0,-1,0,6],[0,0,0,1]],device=device).float()
cam2world = torch.tensor([[1,0,0,1],[0,0,1,-40],[0,-1,0,-6],[0,0,0,1]],device=device).float()






K = torch.tensor([[150,0,150],[0,150,150],[0,0,1]],device=device)
H,W = 300,300
vox_2d,z_vals = project_3d_2d(cam2world, K, vox_inds.float(),with_z=True)
cam_points = dot(torch.inverse(cam2world), vox_inds.float())
depth = torch.norm(cam_points,dim=1)
valid_mask = torch.all(vox_2d>=0,1) & (vox_2d[:,0]<H) & (vox_2d[:,1]<W) & (z_vals>0)
valid_voxel_idx = voxel_idx[valid_mask]
valid_vox_2d = vox_2d[valid_mask]
valid_vox_1d = idx2d_1d(valid_vox_2d,W)
#voxel: [0,1,2,3,4]
#vox_2d: [(0,0),(3,3),(2,1),(3,3)]
#vox_1d: [0,2,5,9,2]
#dense_vox_1d: [0,1,2,3,1]

# dense_vox_1d,dense_valid_depth 
# -> min dense_vox_1d   [by dense_valid_vox_1d]
# [0 -> 0,1,2,3]

unique_ids, dense_valid_vox_1d = torch.unique(valid_vox_1d,return_inverse=True)
valid_depth = depth[valid_mask]
dense_valid_depth = valid_depth[dense_valid_vox_1d]

from torch_scatter import scatter
sc = scatter(torch.stack([dense_valid_vox_1d, dense_valid_depth]),dense_valid_vox_1d.long(),reduce="min")

vis_vox_1d = unique_ids[sc[0].long()].long()
vis_vox_2d = idx1d_2d(vis_vox_1d,W)
vis_depth = sc[1]

## depth image
depth = torch.zeros(H,W,device=device)
depth[tuple(vis_vox_2d[:,[1,0]].T)] = vis_depth


vis_vox_inds = vox_inds[valid_voxel_idx[sc[0].long()]]

dvis(visualize_depth(depth.cpu().numpy()),'img')

### debugging
mm = torch.zeros(H,W,device=device)
mm[tuple(valid_vox_2d[:,[1,0]].long().T)] = valid_depth
dvis(visualize_depth(mm.cpu().numpy()),'img')
dvis(mm,'img')

tt = torch.stack([torch.arange(len(valid_z_vals),device=device), valid_z_vals])
sc = scatter(tt,dense_valid_vox_1d.long(), reduce='min')

vis_pix = torch.stack([sc[0]//W, sc[0]%W],1)

# sliced room based on z min height
sl_vox2scene = np.copy(vox2scene) @ trans_mat([0, 0, min_height_v])




