from re import I
from matplotlib import image
import numpy as np
import imageio
import pickle
from dvis import dvis
import glob
from dutils import dot
from vfront_helper import unproject_2d_3d, visualize_label, visualize_depth

def remap_labeled_img(lab_img, mapping, default=-1):
    new_lab_img = np.full(lab_img.shape,default, dtype=lab_img.dtype)
    for label in np.unique(lab_img):
        label_mask = lab_img == label
        new_lab_img[label_mask] = mapping[label]
    return new_lab_img

# root_dir = "/home/normanm/fb_data/renders_front3d_tiktok2/0003d406-5f27-4bbf-94cd-1cff7c310ba1"
# root_dir = "/home/normanm/fb_data/renders_front3d_tiktok2/00154c06-2ee2-408a-9664-b8fd74742897"
root_dir = "/home/normanm/fb_data/renders_front3d_tiktok3/00154c06-2ee2-408a-9664-b8fd74742897"
# get frame_inds for now like this
# TODO scene_annotation.pkl
frame_inds = [int(fn.split('/')[-1].split('.')[0]) for fn in glob.glob(f"{root_dir}/annotation/*.pkl")]
for frame_idx in frame_inds:
    anno = pickle.load(open(f"{root_dir}/annotation/{frame_idx:05}.pkl","rb"))
    rgb = imageio.imread(f"{root_dir}/rgb/{frame_idx:05}.png")
    depth = imageio.imread(f"{root_dir}/depth/{frame_idx:05}.exr")
    normal = imageio.imread(f"{root_dir}/normal/{frame_idx:05}.exr")
    inst = imageio.imread(f"{root_dir}/inst/{frame_idx:05}.png")
    sem = imageio.imread(f"{root_dir}/sem/{frame_idx:05}.png")
    room = imageio.imread(f"{root_dir}/room/{frame_idx:05}.png")
    asset = imageio.imread(f"{root_dir}/asset/{frame_idx:05}.png")
  
    cam2world = np.array(anno['cam2world_matrix']) @ np.diag([1, -1, -1, 1])
    # fast debug
    # np.linalg.norm(cam2world[:3,:3],axis=1)
    cam_K = np.array(anno['cam_K'])
    pts_world = unproject_2d_3d(cam2world,cam_K,depth)
    dvis(np.concatenate([pts_world,rgb.reshape(-1,3)],1),vs=0.03, ms=100000, name=f"col_pts/{frame_idx}",l=1)
    dvis(np.concatenate([pts_world,room.reshape(-1,1)],1),vs=0.03, ms=100000, name=f"room_pts/{frame_idx}",l=2)
    dvis(np.concatenate([pts_world,asset.reshape(-1,1)],1),vs=0.03, ms=100000, name=f"inst_pts/{frame_idx}",l=3)
    dvis(np.concatenate([pts_world,sem.reshape(-1,1)],1),vs=0.03, ms=100000, name=f"sem_pts/{frame_idx}",l=4)
    # dvis(np.array(rgb),'img')
    # dvis(visualize_label(sem),'img')
    # dvis(visualize_label(inst),'img')
    # dvis(visualize_label(room),'img')
    # dvis(visualize_label(asset),'img')
    # dvis(np.array(normal),'img')
    # dvis(visualize_depth(depth),'img')
    # masked_rgb = np.copy(rgb)
    # masked_rgb[room!=np.argmax(np.bincount(room.flatten()))] = 255
    # dvis(visualize_label(sem),'img')