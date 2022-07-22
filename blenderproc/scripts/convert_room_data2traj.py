import pickle
import torch
import numpy as np

scene_name ="/home/normanm/fb_data/renders_front3d_debug/00154c06-2ee2-408a-9664-b8fd74742897"

room_sample_data = pickle.load(open(f"{scene_name}/room_sample_data.pkl", 'rb'))

# concatenate all rooms, train and val
compl_trajectory = []
split_idx = []
room_ids = []
for room_id, room_data in room_sample_data.items(): 
    compl_trajectory += room_data['train']['c2w']
    split_idx.append(np.zeros(len(room_data['train']['c2w'])))
    compl_trajectory += room_data['val']['c2w']
    split_idx.append(np.ones(len(room_data['val']['c2w'])))
    room_ids.append(room_id * np.ones(len(room_data['train']['c2w']) + len(room_data['val']['c2w'])))

compl_trajectory = np.stack(compl_trajectory)
split_idx = np.concatenate(split_idx).astype(int)
room_ids = np.concatenate(room_ids).astype(int)
    
pickle.dump(compl_trajectory, open(
   f"{scene_name}/compl_trajectory_2d_data.pkl", 'wb'))
pickle.dump(split_idx, open(
   f"{scene_name}/compl_trajectory_split.pkl", 'wb'))
pickle.dump(room_ids, open(
   f"{scene_name}/compl_trajectory_room.pkl", 'wb'))