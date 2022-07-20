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


def idx2d_1d(idx_2d, W):
    return idx_2d[:,0]*W + idx_2d[:,1]

def idx1d_2d(idx_2d,W):
    return torch.stack([idx_2d//W, idx_2d%W],1)


def blender2py3d(c2w):
    swap_y_z = np.array([[1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])

    deg180 = np.deg2rad(180)
    rot_z = np.array([[np.cos(deg180), -np.sin(deg180), 0],
            [np.sin(deg180), np.cos(deg180), 0],
            [0, 0, 1]])

    c2w = swap_y_z @ c2w

    t = c2w[:3,-1]  # Extract translation of the camera
    r = c2w[:3, :3] @ rot_z # Extract rotation matrix of the camera

    t = t @ r # Make rotation local
    return r,t


device = "cuda"

scene_name ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"
min_height_v = 2
vox_size = 0.10
vfront_root = scene_name

scene_mesh_fn = Path(vfront_root, "mesh", "mesh.obj")
scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')
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

swap_x_y = torch.tensor([[0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1.0]],device=device)

cam2world =  torch.tensor([[1,0,0,-2],[0,1,0,1],[0,0,1,-2],[0,0,0,1]],device=device).float() # @hmg(rot_mat([1,0,0],np.pi/5)).to(device) 

R = cam2world[:3,:3].unsqueeze(0)
T = dot(torch.inverse(cam2world[:3,:3]), - cam2world[:3,3]).unsqueeze(0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
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
dvis(images[0,:,:,:3],'img')
dvis(cam2world,name="org/a")
dvis(cam2world_py3d,name="py/a")


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

def get_depth_pts(mesh, cam2world, K, raster_settings):
    depth = render_depth(mesh,cam2world,raster_settings)
    pts = unproject_2d_3d(cam2world,K,depth)
    pts = pts[depth.flatten()>0]
    return pts


fl_factor= 1/(np.tan(float(cameras.fov)/2*np.pi/180)*2)
H = raster_settings.image_size
W = H
K = torch.tensor([[fl_factor*W, 0, W/2],[0,fl_factor*H,H/2],[0,0,1]],device=device)


from dutils import rot_mat, hmg

cam2world =  torch.tensor([[1,0,0,2],[0,1,0,1],[0,0,1,0],[0,0,0,1]],device=device).float() @hmg(rot_mat([1,0,0],np.pi/5)).to(device) 
R, T = convert_py3d(cam2world)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
rasterizer = MeshRasterizer(
    cameras=cameras, 
    raster_settings=raster_settings
)
fragments = rasterizer(mesh)
depth = fragments.zbuf[0,...,0].flip(0).flip(1)

pts = unproject_2d_3d(cam2world,K,depth)
pts = pts[depth.flatten()>0]
dvis(pts,vs=0.03,c=3,ms=20000,name="pts/as")
dvis(cam2world)
dvis(visualize_depth(depth.cpu().numpy()),'img')




Rs = []
Ts = []
for i in range(100):
    Rs.append(R)
    Ts.append(torch.rand(1,3,device=device))
Rs = torch.cat(Rs)
Ts = torch.cat(Ts)

from tqdm import tqdm
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


fragments = rasterizer(mesh.extend(len(Rs)))
depth = fragments.zbuf[0,...,0].flip(0).flip(1)
dvis(visualize_depth(depth.cpu().numpy()),'img')

pts = unproject_2d_3d(cam2world,K,depth)
dvis(pts,vs=0.03,c=3)



scene_mesh = trimesh.load(scene_mesh_fn, force='mesh')
# NOTE: Trimesh does some weird transformation, this seems to fix it
scene_mesh.vertices = np.column_stack(
(scene_mesh.vertices[:, 0], -scene_mesh.vertices[:, 2], scene_mesh.vertices[:, 1]))
# for visualization
sl_scene_mesh = scene_mesh.slice_plane(
np.array([0, 0, 1.8]), np.array([0, 0, -1]))

tm_vox = scene_mesh.voxelized(vox_size)
vox2scene = np.array(tm_vox.transform)


vox_inds = torch.from_numpy(tm_vox.sparse_indices).to(device)
counts = torch.zeros((vox_inds.shape[0],1),device=device)

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




