from re import A
from moviepy.editor import ImageSequenceClip
import glob
import pickle
import imageio
import numpy as np
import os
from tqdm import tqdm

root_dir ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"
data_traj_fn = f"{root_dir}/compl_trajectory_2d_data.pkl"
data_split_fn = f"{root_dir}/compl_trajectory_split.pkl"
data_room_fn = f"{root_dir}/compl_trajectory_room.pkl"

compl_trajectory = pickle.load(open(data_traj_fn,'rb'))
split_idx = pickle.load(open(data_split_fn,'rb'))
room_ids = pickle.load(open(data_room_fn,'rb'))

target_dir = "/home/normanm/fb_data/renders_front3d_data0/00154c06-2ee2-408a-9664-b8fd74742897"

vis_path = f"{target_dir}/visualizations"
os.makedirs(vis_path, exist_ok=True)

train_fps = 8
for target_room_id in tqdm(np.unique(room_ids)):
    target_frame_inds = np.argwhere(room_ids == target_room_id)
    room_rgb_train = []
    room_rgb_val = []
    for frame_idx in target_frame_inds:
        frame_idx = int(frame_idx)
        rgb = imageio.imread(f"{target_dir}/rgb/{frame_idx:05}.png")
        if split_idx[frame_idx]:
            room_rgb_val.append(rgb)
        else:
            room_rgb_train.append(rgb)
    train_clip = ImageSequenceClip(room_rgb_train, fps=train_fps)
    train_clip.write_gif(f"{vis_path}/train_clip_room_{target_room_id}.gif", fps=train_fps, verbose=False, logger=None)
    val_fps = len(room_rgb_val) / len(room_rgb_train) * train_fps
    val_clip = ImageSequenceClip(room_rgb_val, fps=val_fps)
    val_clip.write_gif(f"{vis_path}/val_clip_room_{target_room_id}.gif", fps=val_fps, verbose=False, logger=None)
    