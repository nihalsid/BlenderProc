from matplotlib import image
import numpy as np
import imageio
import pickle
from dvis import dvis
import glob
from dutils import dot

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

root_dir = "test_single"
# get frame_inds for now like this
# TODO scene_annotation.pkl
frame_inds = [int(fn.split('/')[-1].split('.')[0]) for fn in glob.glob(f"{root_dir}/annotation/*.pkl")]
for frame_idx in frame_inds:
    anno = pickle.load(open(f"{root_dir}/annotation/{frame_idx:05}.pkl","rb"))
    rgb = imageio.imread(f"{root_dir}/rgb/{frame_idx:05}.jpg")
    depth = imageio.imread(f"{root_dir}/depth/{frame_idx:05}.exr")
    inst = imageio.imread(f"{root_dir}/inst/{frame_idx:05}.png")
    sem = imageio.imread(f"{root_dir}/sem/{frame_idx:05}.png")
    room = imageio.imread(f"{root_dir}/room/{frame_idx:05}.png")
    
    cam2world = np.array(anno['cam2world_matrix']) @ np.diag([1, -1, -1, 1])
    cam_K = np.array(anno['cam_K'])
    
    pts_world = unproject_2d_3d(cam2world,cam_K,depth)
    dvis(np.concatenate([pts_world,rgb.reshape(-1,3)],1),vs=0.01, ms=10000, name=f"col_pts/{frame_idx}",l=1)
    dvis(np.concatenate([pts_world,room.reshape(-1,1)],1),vs=0.01, ms=10000, name=f"room_pts/{frame_idx}",l=2)
    #dvis(np.concatenate([pts_world,inst.reshape(-1,1)],1),vs=0.01, ms=100000, name=f"inst_pts/{frame_idx}",l=2)
    #dvis(np.concatenate([pts_world,sem.reshape(-1,1)],1),vs=0.01, ms=100000, name=f"sem_pts/{frame_idx}",l=3)